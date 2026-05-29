"""POST /checkin — identify the person in an image or video via face/ReID match."""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..config import DEFAULT_FACE_THR, DEFAULT_REID_THR, GALLERY_PATH
from ..gallery import EmployeeGallery
from ..pipeline import checkin, checkin_video

router = APIRouter(prefix="/checkin", tags=["checkin"])


class CheckinRequest(BaseModel):
    image_path: str = Field(..., description="Server-side path to an image")
    face_thr: float = DEFAULT_FACE_THR
    reid_thr: float = DEFAULT_REID_THR


class CheckinVideoRequest(BaseModel):
    source: str = Field(
        ...,
        description=(
            "Server-side path to a video file, an RTSP URL, or a webcam index (e.g. '0'). "
            "Frames are filtered through a motion gate, blur gate, and face-detection gate "
            "before any expensive inference runs."
        ),
    )
    face_thr: float = DEFAULT_FACE_THR
    reid_thr: float = DEFAULT_REID_THR
    stride: int = Field(5, description="Process every Nth frame.")
    motion_thr: float = Field(
        0.01,
        description="Min fraction of changed pixels to consider a frame active (0.01 = 1%).",
    )
    blur_thr: float = Field(
        80.0,
        description="Min Laplacian variance for a frame to be considered sharp.",
    )
    early_exit_conf: float = Field(
        0.90,
        description="Stop scanning as soon as a match exceeds this confidence score.",
    )
    max_frames: int = Field(
        500,
        description="Max number of frames to inspect (safety cap for long streams).",
    )


def _load_gallery() -> EmployeeGallery:
    if not GALLERY_PATH.exists():
        raise HTTPException(400, "No gallery enrolled yet. Call /enroll first.")
    return EmployeeGallery.load(GALLERY_PATH)


@router.post("")
def checkin_endpoint(req: CheckinRequest):
    """Identify the person in a single image."""
    gallery = _load_gallery()
    try:
        return checkin(req.image_path, gallery,
                       face_thr=req.face_thr, reid_thr=req.reid_thr)
    except FileNotFoundError as e:
        raise HTTPException(400, str(e))


@router.post("/video")
def checkin_video_endpoint(req: CheckinVideoRequest):
    """Identify the person in a video file or camera stream.

    Applies motion, blur, and face-detection gates so only useful frames
    reach the expensive face/ReID models. Returns the best match together
    with frame-level statistics so you can tune the thresholds.
    """
    gallery = _load_gallery()
    # Validate file-path sources exist up front (streams/webcams are validated by cv2)
    if not req.source.startswith("rtsp://") and not req.source.isdigit():
        if not Path(req.source).exists():
            raise HTTPException(400, f"Video source not found: {req.source}")
    try:
        return checkin_video(
            req.source, gallery,
            face_thr=req.face_thr,
            reid_thr=req.reid_thr,
            stride=req.stride,
            motion_thr=req.motion_thr,
            blur_thr=req.blur_thr,
            early_exit_conf=req.early_exit_conf,
            max_frames=req.max_frames,
        )
    except RuntimeError as e:
        raise HTTPException(400, str(e))
