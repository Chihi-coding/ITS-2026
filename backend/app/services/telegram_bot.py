"""Async Telegram alert helper."""

from __future__ import annotations

import logging

import httpx

from app.core.config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

logger = logging.getLogger(__name__)


async def send_telegram_alert(plate_number: str, timestamp: str, image_url: str) -> None:
    """Send a photo alert with violation details to the configured Telegram chat."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("Telegram credentials missing; skipping alert for plate %s", plate_number)
        return

    caption = (
        "Illegal Parking Violation\n"
        f"Plate: {plate_number}\n"
        f"Time: {timestamp}\n"
        f"Image: {image_url}"
    )

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "photo": image_url,
        "caption": caption,
    }

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.post(url, data=payload)
            response.raise_for_status()
        logger.info("Telegram alert sent for plate %s", plate_number)
    except httpx.HTTPError:
        logger.exception("Failed to send Telegram alert for plate %s", plate_number)
