"""Violation API routes."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.database import get_db
from app.models.violation import Violation
from app.schemas.violation import ViolationCreateResponse, ViolationRead
from app.services.telegram_bot import TelegramNotifier

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/violations", tags=["violations"])


def _parse_timestamp(value: str | None) -> datetime:
    if not value:
        return datetime.now(timezone.utc)

    normalized = value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="Invalid timestamp format") from exc

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


async def _persist_image(upload: UploadFile, storage_dir: Path) -> Path:
    suffix = Path(upload.filename or "violation.jpg").suffix or ".jpg"
    filename = f"{uuid.uuid4().hex}{suffix}"
    destination = storage_dir / filename

    content = await upload.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded image is empty")

    destination.write_bytes(content)
    return destination


@router.post("", response_model=ViolationCreateResponse, status_code=201)
async def create_violation(
    background_tasks: BackgroundTasks,
    plate_number: str = Form(...),
    timestamp: str | None = Form(None),
    image: UploadFile = File(...),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> ViolationCreateResponse:
    try:
        saved_path = await _persist_image(image, settings.violation_storage_dir)
        violation_time = _parse_timestamp(timestamp)

        web_image_path = f"/storage/{saved_path.name}"
        violation = Violation(
            plate_number=plate_number.strip() or "UNKNOWN",
            timestamp=violation_time,
            image_path=web_image_path,
            status="Pending",
        )
        db.add(violation)
        db.commit()
        db.refresh(violation)

        notifier = TelegramNotifier(settings)
        background_tasks.add_task(
            notifier.send_violation_alert,
            plate_number=violation.plate_number,
            timestamp=violation.timestamp.isoformat(),
            image_path=saved_path,
            status=violation.status,
        )

        logger.info("Stored violation %s for plate %s", violation.id, violation.plate_number)
        return ViolationCreateResponse(
            id=violation.id,
            plate_number=violation.plate_number,
            timestamp=violation.timestamp,
            image_path=violation.image_path,
            status=violation.status,
        )
    except HTTPException:
        raise
    except Exception as exc:
        db.rollback()
        logger.exception("Failed to create violation")
        raise HTTPException(status_code=500, detail="Unable to save violation") from exc


@router.get("", response_model=list[ViolationRead])
def list_violations(db: Session = Depends(get_db)) -> list[Violation]:
    return (
        db.query(Violation)
        .order_by(Violation.timestamp.desc())
        .all()
    )
