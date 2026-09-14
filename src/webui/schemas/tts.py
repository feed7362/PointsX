"""Request body for POST /api/tts."""
from __future__ import annotations

from pydantic import BaseModel, Field


class TtsRequest(BaseModel):
    """Short Ukrainian phrase for pose hints / countdown (synthesized via edge-tts)."""

    text: str = Field(..., min_length=1, max_length=600)
