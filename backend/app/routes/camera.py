"""Camera streaming & ROI configuration routes."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["camera"])

# ── Paths ────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
VIDEO_PATH = _PROJECT_ROOT / "test.mp4"
ROI_PATH = _PROJECT_ROOT / "ai_module" / "config" / "roi.json"


# ── Schemas ──────────────────────────────────────────────────
class ROIPayload(BaseModel):
    points: list[list[int]]


# ── Video feed (AI-processed) ────────────────────────────────
@router.get("/video_feed")
def video_feed() -> StreamingResponse:
    """Stream AI-processed video (YOLO bounding boxes + ROI overlay) as MJPEG."""
    if not VIDEO_PATH.exists():
        raise HTTPException(status_code=404, detail=f"Video not found: {VIDEO_PATH}")

    # Lazy import to avoid loading YOLO model at module-import time
    from app.services.ai_pipeline import get_pipeline

    pipeline = get_pipeline()
    return StreamingResponse(
        pipeline.generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# ── Cropped video dimensions ────────────────────────────────
@router.get("/video_info")
def video_info() -> dict[str, Any]:
    """Return the cropped frame dimensions so the frontend can match the aspect ratio."""
    import cv2

    from app.services.ai_pipeline import CROP_BOTTOM_RATIO, CROP_TOP_RATIO

    if not VIDEO_PATH.exists():
        raise HTTPException(status_code=404, detail=f"Video not found: {VIDEO_PATH}")

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    try:
        ok, frame = cap.read()
        if not ok or frame is None:
            raise HTTPException(status_code=500, detail="Failed to read video frame")

        h_raw = frame.shape[0]
        cropped = frame[int(h_raw * CROP_TOP_RATIO) : int(h_raw * CROP_BOTTOM_RATIO), :]
        ch, cw = cropped.shape[:2]
        return {"width": cw, "height": ch}
    finally:
        cap.release()


# ── ROI configuration ───────────────────────────────────────
@router.get("/roi")
def get_roi() -> dict[str, Any]:
    """Return the current ROI polygon from roi.json."""
    if not ROI_PATH.exists():
        return {"points": []}

    try:
        data = json.loads(ROI_PATH.read_text(encoding="utf-8"))
        return {"points": data if isinstance(data, list) else []}
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to read ROI file: %s", exc)
        return {"points": []}


@router.post("/roi")
def save_roi(payload: ROIPayload) -> dict[str, Any]:
    """Validate and save 4 ROI coordinates to roi.json."""
    if len(payload.points) != 4:
        raise HTTPException(
            status_code=422,
            detail=f"Exactly 4 points required, got {len(payload.points)}.",
        )

    for idx, pt in enumerate(payload.points):
        if len(pt) != 2:
            raise HTTPException(
                status_code=422,
                detail=f"Point {idx} must be [x, y], got {pt}.",
            )

    ROI_PATH.parent.mkdir(parents=True, exist_ok=True)
    ROI_PATH.write_text(
        json.dumps(payload.points, indent=2) + "\n", encoding="utf-8"
    )

    # Force the running AI pipeline to pick up the new ROI immediately
    # (instead of waiting for the periodic mtime check).
    try:
        from app.services.ai_pipeline import get_pipeline, _pipeline

        if _pipeline is not None:
            _pipeline.invalidate_roi()
            logger.info("ROI invalidated in running pipeline")
    except Exception as exc:
        logger.warning("Could not invalidate pipeline ROI (pipeline may not be running): %s", exc)

    logger.info("ROI saved: %s", payload.points)
    return {"success": True, "points": payload.points}
