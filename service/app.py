"""FastAPI app entrypoint.

Run with:
    venv/bin/uvicorn service.app:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

from fastapi import FastAPI

from .routers import checkin, enroll, events

app = FastAPI(
    title="Employee Activity Tracking",
    description="Endpoints for enrollment, check-in, and event extraction.",
    version="0.1.0",
)

app.include_router(enroll.router)
app.include_router(checkin.router)
app.include_router(events.router)


@app.get("/health")
def health():
    return {"status": "ok"}
