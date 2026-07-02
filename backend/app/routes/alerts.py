"""Manual Telegram alert endpoint."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException

from app.core.supabase_client import get_supabase_client
from app.services.telegram_bot import send_telegram_alert

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/alerts", tags=["alerts"])


@router.post("/telegram/{violation_id}")
async def send_manual_telegram_alert(violation_id: int) -> dict[str, Any]:
    """Manually trigger a Telegram alert for a specific violation.

    Queries Supabase for the violation record, sends the alert, and
    marks ``telegram_sent = True`` in the database.
    """
    logger.info("Manual Telegram alert requested for violation #%s", violation_id)

    # ── Fetch violation record from Supabase ─────────────────
    supabase = get_supabase_client()
    try:
        result = (
            supabase.table("violations")
            .select("*")
            .eq("id", violation_id)
            .single()
            .execute()
        )
    except Exception as exc:
        logger.exception("Failed to query violation #%s", violation_id)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to query violation: {exc}",
        ) from exc

    record = result.data
    if not record:
        raise HTTPException(
            status_code=404,
            detail=f"Violation #{violation_id} not found",
        )

    plate = record.get("license_plate", "UNKNOWN")
    image_url = record.get("evidence_image_path", "")
    timestamp = record.get("detected_at") or record.get("created_at") or "—"

    # ── Send Telegram alert ──────────────────────────────────
    try:
        await send_telegram_alert(
            plate_number=plate,
            timestamp=timestamp,
            image_url=image_url,
        )
    except Exception as exc:
        logger.exception(
            "Manual Telegram alert failed for violation #%s", violation_id
        )
        print(f"[TELEGRAM ERROR] Manual alert failed for violation #{violation_id}: {exc}")
        raise HTTPException(
            status_code=502,
            detail=f"Telegram delivery failed: {exc}",
        ) from exc

    # ── Mark as sent in the database ─────────────────────────
    try:
        supabase.table("violations").update(
            {"telegram_sent": True}
        ).eq("id", violation_id).execute()
    except Exception:
        logger.debug("Could not update telegram_sent column for violation #%s", violation_id)

    return {
        "success": True,
        "message": f"Telegram alert sent for violation #{violation_id}",
        "violation_id": violation_id,
    }
