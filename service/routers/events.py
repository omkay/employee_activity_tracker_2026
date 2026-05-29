"""POST /events/run + GET /events/{job_id} — async pipeline job + results."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..config import (
    DEFAULT_DET_CONF, DEFAULT_DET_IOU, DEFAULT_FACE_THR, DEFAULT_FUSE_WIN,
    DEFAULT_MAX_FRAMES, DEFAULT_PROX_PX, DEFAULT_REID_THR, DEFAULT_STRIDE, GALLERY_PATH,
)
from ..gallery import EmployeeGallery
from ..jobs import store
from ..pipeline import run_pipeline
from ..schemas import ZoneDefinition

router = APIRouter(prefix="/events", tags=["events"])


class EventsRequest(BaseModel):
    video_paths: List[str] = Field(..., description="Server-side video paths (one per camera).")
    camera_ids: Optional[List[str]] = Field(
        None, description="Optional camera IDs (must match length of video_paths)."
    )
    zones: Optional[List[Optional[List[ZoneDefinition]]]] = Field(
        None,
        description=(
            "Per-video zone definitions. Length must equal video_paths when provided. "
            "Each element is a list of ZoneDefinition objects for the corresponding video, "
            "or null to use the full frame as a single work_area for that video. "
            "Omit the field entirely to use the full frame for all videos."
        ),
    )
    det_conf: float = DEFAULT_DET_CONF
    det_iou: float = DEFAULT_DET_IOU
    face_thr: float = DEFAULT_FACE_THR
    reid_thr: float = DEFAULT_REID_THR
    fuse_win: int = DEFAULT_FUSE_WIN
    stride: int = DEFAULT_STRIDE
    max_frames: int = DEFAULT_MAX_FRAMES
    prox_px: int = DEFAULT_PROX_PX
    write_video: bool = False


def _run_all(video_paths, camera_ids, **kwargs):
    """Sequential per-camera pipeline run. Returns merged events JSON."""
    progress_cb = kwargs.pop("progress", None)
    # zones is a per-video list (or None). Pop it so we can index into it per iteration.
    per_video_zones = kwargs.pop("zones", None)
    all_events = []
    annotated = []
    n = len(video_paths)
    for i, (vp, cid) in enumerate(zip(video_paths, camera_ids)):
        def _p(done, total, _i=i, _n=n):
            if progress_cb is not None:
                # Each video contributes 1/n of total progress.
                progress_cb(int((_i + done / max(total, 1)) * 100), n * 100)
        video_zones = per_video_zones[i] if per_video_zones is not None else None
        df, ann = run_pipeline(vp, camera_id=cid, progress=_p, zones=video_zones, **kwargs)
        # Replace NaN/inf with None so the result is JSON-serialisable.
        records = df.where(df.notna(), other=None).to_dict(orient="records")
        all_events.extend(records)
        if ann is not None:
            annotated.append(str(ann))
    return {
        "events": all_events,
        "event_count": len(all_events),
        "annotated_videos": annotated,
    }


@router.post("/run")
def run(req: EventsRequest):
    for vp in req.video_paths:
        if not Path(vp).exists():
            raise HTTPException(400, f"Video not found: {vp}")
    if req.camera_ids and len(req.camera_ids) != len(req.video_paths):
        raise HTTPException(400, "camera_ids length must match video_paths length.")
    if req.zones is not None and len(req.zones) != len(req.video_paths):
        raise HTTPException(400, "zones length must match video_paths length.")

    camera_ids = req.camera_ids or [f"cam{i}" for i in range(len(req.video_paths))]
    gallery = EmployeeGallery.load(GALLERY_PATH) if GALLERY_PATH.exists() else None

    job = store.submit(
        _run_all,
        req.video_paths,
        camera_ids,
        gallery=gallery,
        zones=req.zones,
        det_conf=req.det_conf,
        det_iou=req.det_iou,
        face_thr=req.face_thr,
        reid_thr=req.reid_thr,
        fuse_win=req.fuse_win,
        stride=req.stride,
        max_frames=req.max_frames,
        prox_px=req.prox_px,
        write_video=req.write_video,
    )
    return {"job_id": job.id, "status": job.status}


@router.get("/{job_id}")
def get_job(job_id: str):
    job = store.get(job_id)
    if job is None:
        raise HTTPException(404, f"Unknown job_id: {job_id}")
    return job.to_dict()
