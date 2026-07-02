"""Async Telegram alert helper.

Downloads the evidence image from Supabase Storage and uploads it
directly to the Telegram Bot API as a multipart file upload.  This is
more reliable than passing a URL (which may require public access).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta

import httpx

from app.core.config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

logger = logging.getLogger(__name__)

# GMT+7 (Indochina Time / Asia Ho Chi Minh)
_GMT7 = timezone(timedelta(hours=7))


class TelegramSendError(Exception):
    """Raised when a Telegram alert cannot be delivered."""


def _parse_telegram_error(response: httpx.Response) -> str:
    """Extract a human-readable error from the Telegram API JSON body."""
    try:
        body = response.json()
        description = body.get("description", response.text)
        return f"Telegram API {response.status_code}: {description}"
    except Exception:
        return f"Telegram API {response.status_code}: {response.text[:300]}"


def _format_timestamp_gmt7(raw_timestamp: str) -> str:
    """Convert an ISO-8601 timestamp string to a human-readable GMT+7 format.

    Examples:
        '2026-07-02T12:00:00+00:00' → '02 Jul 2026, 19:00:00 (GMT+7)'
        '2026-07-02T12:00:00Z'      → '02 Jul 2026, 19:00:00 (GMT+7)'
    """
    try:
        # Handle Z suffix
        normalized = raw_timestamp.replace("Z", "+00:00")
        dt = datetime.fromisoformat(normalized)
        # Convert to GMT+7
        dt_gmt7 = dt.astimezone(_GMT7)
        return dt_gmt7.strftime("%d %b %Y, %H:%M:%S") + " (GMT+7)"
    except Exception:
        # If parsing fails, return the raw value
        return raw_timestamp


async def _download_image(image_url: str) -> bytes | None:
    """Download image bytes from a URL (e.g. Supabase public URL)."""
    if not image_url:
        return None
    try:
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            response = await client.get(image_url)
            if response.is_success:
                return response.content
            logger.warning("Failed to download evidence image: HTTP %s", response.status_code)
    except Exception as exc:
        logger.warning("Failed to download evidence image: %s", exc)
    return None


async def send_telegram_alert(
    plate_number: str, timestamp: str, image_url: str
) -> None:
    """Send a photo alert with violation details to the configured Telegram chat.

    The function:
    1. Downloads the evidence image from Supabase Storage
    2. Formats the timestamp in GMT+7
    3. Uploads the image as a multipart file to Telegram's sendPhoto API

    Falls back to sending the URL if the image download fails.

    Raises:
        TelegramSendError: on missing credentials or failed HTTP request.
    """
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        msg = (
            "Telegram credentials missing — "
            f"TELEGRAM_BOT_TOKEN={'set' if TELEGRAM_BOT_TOKEN else 'EMPTY'}, "
            f"TELEGRAM_CHAT_ID={'set' if TELEGRAM_CHAT_ID else 'EMPTY'}"
        )
        logger.error(msg)
        print(f"[TELEGRAM ERROR] {msg}")
        raise TelegramSendError(msg)

    # Format timestamp in GMT+7
    formatted_time = _format_timestamp_gmt7(timestamp)

    caption = (
        "🚨 Illegal Parking Violation\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        f"🚗 Plate: {plate_number}\n"
        f"🕐 Time: {formatted_time}\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        "⚠️ Action Required"
    )

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"

    try:
        # Try to download the image and upload as file (more reliable)
        image_bytes = await _download_image(image_url)

        async with httpx.AsyncClient(timeout=20.0) as client:
            if image_bytes:
                # Upload image as multipart file
                response = await client.post(
                    url,
                    data={"chat_id": TELEGRAM_CHAT_ID, "caption": caption},
                    files={"photo": ("violation.jpg", image_bytes, "image/jpeg")},
                )
            else:
                # Fallback: send URL directly
                response = await client.post(
                    url,
                    data={
                        "chat_id": TELEGRAM_CHAT_ID,
                        "photo": image_url,
                        "caption": caption,
                    },
                )

            if not response.is_success:
                error_detail = _parse_telegram_error(response)
                logger.error("Telegram alert failed for plate %s: %s", plate_number, error_detail)
                print(f"[TELEGRAM ERROR] {error_detail}")
                raise TelegramSendError(error_detail)

        logger.info("Telegram alert sent for plate %s", plate_number)
        print(f"[TELEGRAM] Alert sent successfully for plate {plate_number}")

    except TelegramSendError:
        raise
    except httpx.TimeoutException as exc:
        msg = f"Telegram request timed out after 20s: {exc}"
        logger.error(msg)
        print(f"[TELEGRAM ERROR] {msg}")
        raise TelegramSendError(msg) from exc
    except httpx.HTTPError as exc:
        msg = f"Telegram HTTP error: {type(exc).__name__}: {exc}"
        logger.exception("Failed to send Telegram alert for plate %s", plate_number)
        print(f"[TELEGRAM ERROR] {msg}")
        raise TelegramSendError(msg) from exc
