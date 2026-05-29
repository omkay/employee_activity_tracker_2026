"""POST /enroll — register a person's face/body fingerprint from image paths."""
from __future__ import annotations

from typing import List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..gallery import enroll_person
from ..models import get_face_embedder, get_reid_embedder

router = APIRouter(prefix="/enroll", tags=["enroll"])


class EnrollRequest(BaseModel):
    name: str = Field(..., description="Employee folder name (used as their ID)")
    face_images: List[str] = Field(default_factory=list,
                                   description="Server-side paths to face images")
    body_images: List[str] = Field(default_factory=list,
                                   description="Server-side paths to full-body images")


@router.post("")
def enroll(req: EnrollRequest):
    if not req.face_images and not req.body_images:
        raise HTTPException(400, "Provide at least one face or body image path.")
    try:
        summary = enroll_person(
            req.name, req.face_images, req.body_images,
            get_face_embedder(), get_reid_embedder(),
        )
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(400, str(e))
    return summary
