from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from supabase import create_client, Client
import tempfile
import sys
import cv2
import numpy as np
import subprocess
from pathlib import Path

sys.path.append(os.path.dirname(__file__))  # ensures local imports work

# ✅ Import alert function
from alerts import send_discord_alert

# Add gemini and sentry modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from gemini.gemini_description import analyze_security_image

# Import sentry service
try:
    from sentry.sentry_service import SentryService
    SENTRY_AVAILABLE = True
except Exception as e:
    print(f"[WARN] Sentry service not available: {e}")
    SENTRY_AVAILABLE = False

# Load environment variables
load_dotenv()

# Initialize Supabase client
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in .env file")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

app = FastAPI()

# Add CORS middleware to allow frontend to communicate with backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174", "http://localhost:3000"],  # Frontend dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global sentry instance
sentry: Optional[SentryService] = None


# Startup event - initialize sentry
@app.on_event("startup")
async def startup_event():
    """Initialize the sentry service on startup."""
    global sentry
    if SENTRY_AVAILABLE:
        try:
            print("[STARTUP] Initializing sentry service...")
            sentry = SentryService()
            sentry.start()
            print("[STARTUP] Sentry service started successfully")
        except Exception as e:
            print(f"[ERROR] Failed to start sentry: {e}")
            sentry = None
    else:
        print("[STARTUP] Sentry not available - video streaming disabled")


# Shutdown event - cleanup sentry
@app.on_event("shutdown")
async def shutdown_event():
    """Stop the sentry service on shutdown."""
    global sentry
    if sentry:
        print("[SHUTDOWN] Stopping sentry service...")
        sentry.stop()
        print("[SHUTDOWN] Sentry stopped")


# Event model
class Event(BaseModel):
    id: Optional[int] = None
    timestamp: Optional[str] = None
    event_type: str
    description: str
    severity: str
    image_url: Optional[str] = None


class EventCreate(BaseModel):
    event_type: str
    description: str
    severity: str = "info"


class FrameAnalysisResponse(BaseModel):
    event_id: int
    timestamp: str
    analysis: str
    severity: str
    status: str


@app.get("/health")
def health():
    """Health check endpoint"""
    return {"status": "ok"}


@app.get("/anomalies")
def get_anomalies(limit: Optional[int] = 50):
    """
    Get anomaly events (alias for /events endpoint).
    Used by the frontend AnomalyLog component.
    """
    return get_events(limit=limit)


class ControlCommand(BaseModel):
    command: str


@app.post("/control")
def camera_control(command: ControlCommand):
    """
    Handle camera control commands from the frontend.

    Commands:
    - toggle_lock: Toggle auto-tracking on/off
    - center: Center the camera servos
    - pan_left, pan_right: Pan camera left or right
    - tilt_up, tilt_down: Tilt camera up or down
    """
    global sentry

    if not sentry:
        return {"status": "error", "message": "Sentry not available"}

    print(f"[CONTROL] Received command: {command.command}")
    sentry.send_command(command.command)

    return {"status": "ok", "command": command.command}


@app.get("/snapshots/stats")
def get_snapshot_stats():
    """
    Get statistics about snapshots and Gemini analysis.
    """
    global sentry
    
    if not sentry:
        return {"status": "error", "message": "Sentry not available"}
    
    return sentry.get_snapshot_stats()


@app.post("/system/start")
def start_sentry_system():
    """
    Start the sentry service (camera + tracking).
    This restarts just the backend sentry, not the entire system.
    """
    global sentry
    
    try:
        if sentry and sentry.running:
            return {"status": "already_running", "message": "Sentry is already running"}
        
        if not SENTRY_AVAILABLE:
            return {"status": "error", "message": "Sentry service not available"}
        
        # Reinitialize sentry if it was stopped (camera was released)
        # We need to create a fresh instance because the camera device was closed
        sentry = SentryService()
        sentry.start()
        
        return {
            "status": "success",
            "message": "Sentry service started",
            "running": True
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start sentry: {str(e)}")


@app.post("/system/stop")
def stop_sentry_system():
    """
    Stop the sentry service (camera + tracking).
    Keeps the backend API running, just stops the camera/tracking.
    """
    global sentry
    
    try:
        if not sentry or not sentry.running:
            return {"status": "already_stopped", "message": "Sentry is not running"}
        
        sentry.stop()
        
        return {
            "status": "success",
            "message": "Sentry service stopped",
            "running": False
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to stop sentry: {str(e)}")


@app.post("/system/restart")
def restart_sentry_system():
    """
    Restart the sentry service (camera + tracking).
    """
    global sentry
    
    try:
        # Stop if running
        if sentry and sentry.running:
            sentry.stop()
        
        # Reinitialize and start
        sentry = SentryService()
        sentry.start()
        
        return {
            "status": "success",
            "message": "Sentry service restarted",
            "running": True
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to restart sentry: {str(e)}")


@app.get("/system/status")
def get_system_status():
    """
    Get current status of the sentry system.
    """
    global sentry
    
    return {
        "sentry_available": SENTRY_AVAILABLE,
        "sentry_initialized": sentry is not None,
        "sentry_running": sentry.running if sentry else False,
        "stats": sentry.get_stats() if (sentry and sentry.running) else None
    }


def generate_frames():
    """
    Generator function that yields MJPEG frames from the sentry.
    """
    global sentry

    if not sentry:
        # Return a placeholder frame if sentry not available
        placeholder_path = os.path.join(os.path.dirname(__file__), "static", "placeholder.jpg")
        placeholder = cv2.imread(placeholder_path)
        if placeholder is None:
            # Create a simple placeholder
            placeholder = cv2.putText(
                cv2.rectangle(np.zeros((320, 320, 3), dtype=np.uint8), (0, 0), (320, 320), (50, 50, 50), -1),
                "Camera Not Available",
                (50, 160),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (200, 200, 200),
                2
            )

        ret, buffer = cv2.imencode('.jpg', placeholder)
        frame_bytes = buffer.tobytes()

        while True:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            import time
            time.sleep(0.1)
    else:
        # Stream frames from sentry
        while True:
            frame = sentry.get_latest_frame()

            if frame is not None:
                # Encode frame as JPEG with reduced quality for performance
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

            import time
            time.sleep(0.01)  # Minimal sleep, let sentry control frame rate


@app.get("/video_feed")
def video_feed():
    """
    Stream video feed from the camera using MJPEG.
    This works directly with HTML <img> tags.
    """
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/sentry/stats")
def get_sentry_stats():
    """
    Get current sentry statistics (FPS, tracking status, servo angles, etc.)
    """
    global sentry

    if not sentry:
        return {"status": "unavailable"}

    return sentry.get_stats()


@app.get("/events", response_model=List[Event])
def get_events(limit: Optional[int] = 10, event_type: Optional[str] = None):
    """
    Get security events from Supabase.
    """
    try:
        query = supabase.table("events").select("*").order("timestamp", desc=True).limit(limit)
        if event_type:
            query = query.eq("event_type", event_type)
        response = query.execute()
        return response.data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching events: {str(e)}")


@app.post("/events", response_model=Event)
def create_event(event: EventCreate):
    """
    Create a new security event.
    """
    try:
        event_data = event.model_dump()
        event_data["timestamp"] = datetime.now().isoformat()

        response = supabase.table("events").insert(event_data).execute()

        # ✅ Send Discord alert (only warnings/critical)
        if event_data["severity"] in ["warning", "critical"]:
            send_discord_alert(
                event_type=event_data["event_type"],
                description=event_data["description"],
                severity=event_data["severity"],
                image_url=event_data.get("image_url")
            )

        if response.data:
            return response.data[0]
        else:
            raise HTTPException(status_code=500, detail="Failed to create event")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating event: {str(e)}")


@app.delete("/events/{event_id}")
def delete_event(event_id: int):
    """
    Delete a security event by ID.
    """
    try:
        response = supabase.table("events").delete().eq("id", event_id).execute()

        if response.data:
            return {"status": "success", "message": f"Event {event_id} deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail=f"Event {event_id} not found")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting event: {str(e)}")


@app.post("/analyze-frame", response_model=FrameAnalysisResponse)
async def analyze_frame(file: UploadFile = File(...)):
    """
    Analyze a frame with Gemini Vision, upload to Supabase Storage,
    log results, and send alerts to Discord if severity is high.
    """
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_path = temp_file.name
            content = await file.read()
            temp_file.write(content)

        result = analyze_security_image(temp_path)

        if result["status"] != "success":
            raise HTTPException(status_code=500, detail=f"Gemini analysis failed: {result.get('error', 'Unknown error')}")

        analysis_text = result["analysis"]
        severity = result.get("severity", "info")

        if not severity or severity not in ["info", "warning", "critical"]:
            severity = "info"
            lower = analysis_text.lower()
            if any(k in lower for k in ["alert", "suspicious", "unusual", "concern"]):
                severity = "warning"
            elif any(k in lower for k in ["danger", "threat", "emergency", "critical"]):
                severity = "critical"

        timestamp = datetime.now()
        timestamp_str = timestamp.isoformat()
        file_extension = os.path.splitext(file.filename)[1] if file.filename else ".jpg"
        storage_filename = f"frame_{timestamp.strftime('%Y%m%d_%H%M%S')}_{timestamp.microsecond}{file_extension}"

        supabase.storage.from_("security-frames").upload(
            path=storage_filename,
            file=content,
            file_options={"content-type": file.content_type or "image/jpeg"}
        )

        image_url = supabase.storage.from_("security-frames").get_public_url(storage_filename)

        event_data = {
            "event_type": "vision_analysis",
            "description": analysis_text,
            "severity": severity,
            "timestamp": timestamp_str,
            "image_url": image_url
        }

        db_response = supabase.table("events").insert(event_data).execute()

        # ✅ Send Discord alert for warning/critical results
        if severity in ["warning", "critical"]:
            send_discord_alert(
                event_type=event_data["event_type"],
                description=event_data["description"],
                severity=event_data["severity"],
                image_url=event_data.get("image_url")
            )

        if not db_response.data or len(db_response.data) == 0:
            raise HTTPException(status_code=500, detail="Failed to store analysis in database")

        event_record = db_response.data[0]
        event_id = event_record.get("id", 0)

        return FrameAnalysisResponse(
            event_id=event_id,
            timestamp=timestamp_str,
            analysis=analysis_text,
            severity=severity,
            status="success"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing frame: {str(e)}")

    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


app.mount("/", StaticFiles(directory="web/static/dist", html=True), name="frontend")
