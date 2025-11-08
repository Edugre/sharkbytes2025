from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

app = FastAPI()

@app.get("/health")
def health(): return {"status": "ok"}

app.mount("/", StaticFiles(directory="web/static/dist", html=True), name="frontend")