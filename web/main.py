from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from supabase import create_client, Client

# Load environment variables
load_dotenv()

# Initialize Supabase client
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in .env file")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

app = FastAPI()


# Event model
class Event(BaseModel):
    id: Optional[int] = None
    timestamp: Optional[str] = None
    event_type: str
    description: str
    severity: str


class EventCreate(BaseModel):
    event_type: str
    description: str
    severity: str = "info"


@app.get("/health")
def health():
    """Health check endpoint"""
    return {"status": "ok"}


@app.get("/events", response_model=List[Event])
def get_events(limit: Optional[int] = 10, event_type: Optional[str] = None):
    """
    Get security events from Supabase.

    Parameters:
    - limit: Maximum number of events to return (default: 10)
    - event_type: Filter by event type (optional)
    """
    try:
        # Build query
        query = supabase.table("events").select("*").order("timestamp", desc=True).limit(limit)

        # Add filter if event_type specified
        if event_type:
            query = query.eq("event_type", event_type)

        # Execute query
        response = query.execute()

        return response.data

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching events: {str(e)}")


@app.post("/events", response_model=Event)
def create_event(event: EventCreate):
    """
    Create a new security event.

    This endpoint allows the sentry system to log events to the database.
    """
    try:
        # Add timestamp
        event_data = event.model_dump()
        event_data["timestamp"] = datetime.now().isoformat()

        # Insert into Supabase
        response = supabase.table("events").insert(event_data).execute()

        if response.data:
            return response.data[0]
        else:
            raise HTTPException(status_code=500, detail="Failed to create event")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating event: {str(e)}")


app.mount("/", StaticFiles(directory="web/static/dist", html=True), name="frontend")