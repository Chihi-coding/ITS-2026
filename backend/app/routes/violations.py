"""Violation API routes backed by Supabase."""

from __future__ import annotations

import logging
import re
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.core.config import DEBUG
from app.core.supabase_client import get_supabase_client

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/violations", tags=["violations"])

VIOLATION_BUCKET = "violation-images"


def _violation_time_from_record(record: dict[str, Any]) -> str:
    """Read the violation time from whichever timestamp column Supabase returned."""
    for key in ("violation_started_at", "created_at", "detected_at", "violation_time", "timestamp"):
        if record.get(key) is not None:
            return str(record[key])
    return ""


def _normalize_violation(record: dict[str, Any]) -> dict[str, Any]:
    """Map Supabase column names to the API shape used by the frontend."""
    normalized = dict(record)
    normalized["plate_number"] = record.get("license_plate", record.get("plate_number"))
    normalized["image_url"] = record.get("evidence_image_path", record.get("image_url"))
    normalized["timestamp"] = _violation_time_from_record(record)
    return normalized


def _error_detail(message: str, exc: Exception | None = None) -> str:
    if DEBUG and exc is not None:
        return f"{message}: {exc}"
    return message


def _safe_filename(timestamp: str, plate_number: str) -> str:
    """Build a storage-safe object name."""
    ts = re.sub(r"[^\w\-]", "_", timestamp)
    plate = re.sub(r"[^\w\-]", "_", plate_number)
    return f"{ts}_{plate}.jpg"


def _upload_image(image_bytes: bytes, filename: str) -> str:
    """Upload image bytes to Supabase Storage and return the public URL."""
    supabase = get_supabase_client()
    storage = supabase.storage.from_(VIOLATION_BUCKET)

    try:
        storage.upload(
            filename,
            image_bytes,
            file_options={"content-type": "image/jpeg", "upsert": "true"},
        )
    except Exception as exc:
        logger.exception("Supabase storage upload failed for %s", filename)
        raise HTTPException(
            status_code=500,
            detail=_error_detail(
                f"Failed to upload image to storage bucket '{VIOLATION_BUCKET}'",
                exc,
            ),
        ) from exc

    return storage.get_public_url(filename)


def _insert_violation(
    plate_number: str,
    timestamp: str,
    image_url: str,
    duration_seconds: int = 0,
    status: str = "Pending",
    camera_id: int = 1,
    zone_id: int = 1,
) -> dict[str, Any]:
    """Insert a violation row into Supabase and return the created record.

    Supabase table 'violations' columns:
        license_plate, evidence_image_path, detected_at, duration_seconds,
        status, telegram_sent, camera_id, zone_id
    """
    supabase = get_supabase_client()
    payload = {
        "license_plate": plate_number,
        "detected_at": timestamp,
        "evidence_image_path": image_url,
        "duration_seconds": duration_seconds,
        "telegram_sent": False,
        "status": status,
        "camera_id": camera_id,
        "zone_id": zone_id,
    }

    try:
        result = supabase.table("violations").insert(payload).execute()
    except Exception as exc:
        logger.exception("Supabase insert failed for plate %s", plate_number)
        print(f"ERROR: Database insert failed — {exc}")
        raise HTTPException(
            status_code=500,
            detail=_error_detail("Failed to save violation record to table 'violations'", exc),
        ) from exc

    if not result.data:
        print("ERROR: Database insert returned no data")
        raise HTTPException(status_code=500, detail="Violation insert returned no data")

    print(f"SUCCESS: Database insert complete — id={result.data[0].get('id')}, plate={plate_number}")
    logger.info("SUCCESS: inserted violation id=%s", result.data[0].get("id"))
    return result.data[0]


def _mark_telegram_sent(violation_id: int) -> None:
    """Mark a violation row as having a successful Telegram alert."""
    try:
        supabase = get_supabase_client()
        supabase.table("violations").update({"telegram_sent": True}).eq("id", violation_id).execute()
    except Exception:
        logger.debug("Could not update telegram_sent column (may not exist in table)")


@router.post("")
async def create_violation(
    plate_number: str = Form(...),
    timestamp: str = Form(...),
    image: UploadFile = File(...),
    duration_seconds: float = Form(0),
) -> dict[str, Any]:
    """Receive a violation image, store it in Supabase, and save metadata.

    Telegram alerts are NOT sent automatically — they are triggered
    manually from the frontend dashboard.
    """
    try:
        image_bytes = await image.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Uploaded image is empty")

        filename = _safe_filename(timestamp, plate_number)
        image_url = _upload_image(image_bytes, filename)

        record = _insert_violation(
            plate_number=plate_number.strip(),
            timestamp=timestamp,
            image_url=image_url,
            duration_seconds=max(0, int(round(duration_seconds))),
            status="Pending",
        )

        logger.info(
            "Violation created: id=%s plate=%s",
            record.get("id"),
            record.get("license_plate"),
        )
        return {
            "success": True,
            "message": "Violation recorded",
            "data": _normalize_violation(record),
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unexpected error while creating violation")
        print(f"ERROR: Unexpected failure in create_violation — {type(exc).__name__}: {exc}")
        raise HTTPException(status_code=500, detail=f"Unable to process violation: {exc}") from exc


@router.get("")
def list_violations() -> list[dict[str, Any]]:
    """Return all violations ordered by newest first."""
    supabase = get_supabase_client()

    try:
        result = (
            supabase.table("violations")
            .select("*")
            .order("id", desc=True)
            .execute()
        )
        return [_normalize_violation(row) for row in (result.data or [])]
    except Exception as exc:
        logger.exception("Failed to fetch violations from Supabase")
        raise HTTPException(status_code=500, detail="Failed to fetch violations") from exc
