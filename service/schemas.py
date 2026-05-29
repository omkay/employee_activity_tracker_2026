"""Shared Pydantic schemas used by the API layer and the pipeline."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class ZoneDefinition(BaseModel):
    """A named, typed zone to apply to a video frame.

    Coordinates are in pixels relative to the top-left corner of the frame.
    Any coordinate left as *None* is replaced by the corresponding frame edge
    at runtime, so omitting all four coords makes the zone cover the whole frame.

    ``zone_type`` controls which events are reported inside the zone:
    - ``"work_area"``   → presence, working (laptop/monitor), phone_use
    - ``"common_area"`` → presence, interaction (proximity between employees)
    """

    label: str = Field(
        ...,
        description="Human-readable label shown in event output, e.g. 'omar_desk' or 'reception'.",
    )
    zone_type: Literal["work_area", "common_area"] = Field(
        "work_area",
        description="Activity class: 'work_area' or 'common_area'.",
    )
    x1: Optional[int] = Field(None, description="Left edge in pixels (default: 0).")
    y1: Optional[int] = Field(None, description="Top edge in pixels (default: 0).")
    x2: Optional[int] = Field(None, description="Right edge in pixels (default: frame width).")
    y2: Optional[int] = Field(None, description="Bottom edge in pixels (default: frame height).")
