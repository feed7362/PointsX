"""Small helpers shared by every storage backend."""
from __future__ import annotations

def _sanitise_request_id(request_id: str) -> str:
    rid = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in request_id)[:96]
    return rid or "anon"


def _extension_for(content_type: str) -> str:
    ct = (content_type or "").lower().strip()
    if "jpeg" in ct or "jpg" in ct:
        return "jpg"
    if "png" in ct:
        return "png"
    if "webp" in ct:
        return "webp"
    return "bin"
