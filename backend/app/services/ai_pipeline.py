"""Unified AI pipeline: YOLO tracking + ROI dwell + violation reporting.

Runs YOLOv8 vehicle detection with persistent tracking on each video frame,
crops the blurred vertical-video padding, monitors dwell time inside the ROI
polygon, and saves annotated evidence (Supabase Storage + DB record) when the
threshold is exceeded.  Telegram alerts are triggered manually from the
frontend dashboard, NOT from this pipeline.

All AI annotations (bounding boxes, ROI overlay, dwell timers) are drawn on
the cropped frames before they are JPEG-encoded and yielded as an MJPEG stream.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator

import cv2
import numpy as np
from ultralytics import YOLO

# ── Ensure project root is on sys.path for ai_module imports ─
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.core.supabase_client import get_supabase_client

logger = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────
VIDEO_PATH = _PROJECT_ROOT / "test.mp4"
ROI_PATH = _PROJECT_ROOT / "ai_module" / "config" / "roi.json"
MODEL_PATH = str(_PROJECT_ROOT / "yolov8n.pt")

# ── Detection / streaming constants ─────────────────────────
VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck
MAX_DWELL_SECONDS = 5
JPEG_QUALITY = 70
TARGET_FPS = 20
FRAME_INTERVAL = 1.0 / TARGET_FPS
ROI_CHECK_EVERY = 5  # check roi.json mtime every N frames
VIOLATION_BUCKET = "violation-images"
USE_MOCK_PLATE = True
MOCK_PLATE = "30F-12345"

# ── Frame cropping ratios ────────────────────────────────────
# Remove blurred vertical-video padding: keep only the clear middle portion.
CROP_TOP_RATIO = 0.25
CROP_BOTTOM_RATIO = 0.80


# ── Helpers ──────────────────────────────────────────────────

def _safe_filename(timestamp: str, plate: str) -> str:
    """Build a storage-safe object name from timestamp and plate."""
    ts = re.sub(r"[^\w\-]", "_", timestamp)
    pl = re.sub(r"[^\w\-]", "_", plate)
    return f"{ts}_{pl}.jpg"


# ── Pipeline ─────────────────────────────────────────────────

class SmartParkingPipeline:
    """Unified YOLO + ROI + violation pipeline for MJPEG streaming.

    Usage::

        pipeline = get_pipeline()
        return StreamingResponse(
            pipeline.generate_frames(),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )
    """

    def __init__(self) -> None:
        logger.info("Initializing SmartParkingPipeline...")
        print("[AI PIPELINE] Loading YOLO model...")
        self._model = YOLO(MODEL_PATH)
        print(f"[AI PIPELINE] YOLO model loaded: {MODEL_PATH}")

        # OCR engine (optional, may be heavy)
        self._ocr = self._init_ocr()

        # ROI state
        self._roi_polygon: np.ndarray | None = self._load_roi()
        self._roi_mtime: float = self._get_roi_mtime()

        # Per-track dwell state:  track_id → {"first_seen": float, "is_reported": bool}
        self._tracked: dict[int, dict[str, float | bool]] = {}
        self._frame_idx = 0

        # Prevent two concurrent generators from sharing YOLO tracker state
        self._lock = threading.Lock()

        print("[AI PIPELINE] Pipeline ready ✓")

    # ── Initialization helpers ───────────────────────────────

    @staticmethod
    def _init_ocr():
        """Try to load the EasyOCR engine; return None on failure."""
        try:
            from ai_module.core.ocr_engine import OCREngine

            engine = OCREngine(languages=["en"])
            print("[AI PIPELINE] OCR engine ready")
            return engine
        except Exception as exc:
            logger.warning("OCR engine unavailable: %s", exc)
            print(f"[AI PIPELINE] OCR unavailable ({exc}), will use mock plates")
            return None

    @staticmethod
    def _load_roi() -> np.ndarray | None:
        """Load ROI polygon from roi.json (supports both list and dict formats)."""
        if not ROI_PATH.exists():
            logger.warning("ROI file not found: %s", ROI_PATH)
            return None
        try:
            raw = json.loads(ROI_PATH.read_text(encoding="utf-8"))
            points = raw.get("polygon", raw) if isinstance(raw, dict) else raw
            if isinstance(points, list) and len(points) >= 3:
                polygon = np.array(points, dtype=np.int32)
                print(f"[AI PIPELINE] ROI loaded: {len(polygon)} points")
                return polygon
        except Exception:
            logger.exception("Failed to load ROI from %s", ROI_PATH)
        return None

    @staticmethod
    def _get_roi_mtime() -> float:
        try:
            return os.path.getmtime(ROI_PATH) if ROI_PATH.exists() else 0.0
        except OSError:
            return 0.0

    # ── ROI hot-reload ───────────────────────────────────────

    def invalidate_roi(self) -> None:
        """Force an immediate reload of roi.json.

        Called by the /api/roi POST endpoint so the stream picks up
        the new polygon without waiting for the periodic mtime check.
        """
        try:
            self._roi_polygon = self._load_roi()
            self._roi_mtime = self._get_roi_mtime()
            self._tracked.clear()
            coords = self._roi_polygon.tolist() if self._roi_polygon is not None else None
            print(f"[AI PIPELINE] ROI force-reloaded: {coords}")
            logger.info("ROI invalidated and reloaded: %s", coords)
        except Exception as exc:
            logger.exception("ROI forced reload failed: %s", exc)

    def _maybe_reload_roi(self) -> None:
        """Check roi.json mtime every ROI_CHECK_EVERY frames; reload if changed."""
        if self._frame_idx % ROI_CHECK_EVERY != 0 or not ROI_PATH.exists():
            return
        try:
            mtime = os.path.getmtime(ROI_PATH)
            if mtime != self._roi_mtime:
                self._roi_polygon = self._load_roi()
                self._roi_mtime = mtime
                self._tracked.clear()
                coords = self._roi_polygon.tolist() if self._roi_polygon is not None else None
                print(f"[AI PIPELINE] ROI hot-reloaded (mtime={mtime}): {coords}")
        except Exception as exc:
            logger.warning("ROI reload failed (file may be mid-write): %s", exc)

    # ── Drawing helpers ──────────────────────────────────────

    def _draw_roi(self, frame: np.ndarray) -> None:
        """Draw the red ROI polygon with semi-transparent fill."""
        if self._roi_polygon is None:
            return
        cv2.polylines(frame, [self._roi_polygon], True, (0, 0, 255), 2)
        overlay = frame.copy()
        cv2.fillPoly(overlay, [self._roi_polygon], (0, 0, 255))
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

    def _is_inside_roi(self, point: tuple[int, int]) -> bool:
        if self._roi_polygon is None:
            return False
        return cv2.pointPolygonTest(self._roi_polygon, point, False) >= 0

    # ── Violation reporting (background thread) ──────────────

    def _report_violation(
        self, evidence_frame: np.ndarray, plate: str, dwell_seconds: float
    ) -> None:
        """Upload annotated evidence → insert DB record.

        Telegram alerts are NOT sent here — they are triggered manually
        from the frontend dashboard via /api/alerts/telegram/{violation_id}.

        Runs in a daemon thread so it never blocks the video stream.
        """
        try:
            timestamp = datetime.now(timezone.utc).isoformat()

            # Encode full annotated frame as JPEG evidence
            ok, buf = cv2.imencode(".jpg", evidence_frame)
            if not ok:
                logger.error("Failed to encode violation evidence to JPEG")
                return
            image_bytes = buf.tobytes()

            # ── Upload to Supabase Storage ───────────────────
            filename = _safe_filename(timestamp, plate)
            supabase = get_supabase_client()
            storage = supabase.storage.from_(VIOLATION_BUCKET)
            storage.upload(
                filename,
                image_bytes,
                file_options={"content-type": "image/jpeg", "upsert": "true"},
            )
            image_url = storage.get_public_url(filename)
            print(f"[VIOLATION] Evidence uploaded: {filename}")

            # ── Insert violation record (telegram_sent=False) ─
            result = (
                supabase.table("violations")
                .insert(
                    {
                        "license_plate": plate,
                        "detected_at": timestamp,
                        "evidence_image_path": image_url,
                        "duration_seconds": max(0, int(round(dwell_seconds))),
                        "telegram_sent": False,
                        "status": "Pending",
                        "camera_id": 1,
                        "zone_id": 1,
                    }
                )
                .execute()
            )

            if result.data:
                violation_id = result.data[0].get("id")
                print(
                    f"[VIOLATION] DB record created: id={violation_id}, plate={plate}"
                )

        except Exception as exc:
            logger.exception("Violation report pipeline failed")
            print(f"[VIOLATION ERROR] {type(exc).__name__}: {exc}")

    def _handle_violation(
        self,
        frame: np.ndarray,
        track_id: int,
        bbox: tuple[int, int, int, int],
        dwell_seconds: float,
    ) -> None:
        """Run OCR on vehicle crop, capture full annotated frame as evidence,
        and fire background violation report."""
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        x1c, y1c = max(0, x1), max(0, y1)
        x2c, y2c = min(w, x2), min(h, y2)
        crop = frame[y1c:y2c, x1c:x2c]

        # OCR → plate text
        plate = MOCK_PLATE
        if self._ocr is not None:
            try:
                result = self._ocr.read_plate(crop if crop.size else frame)
                if result != "UNKNOWN":
                    plate = result
                elif USE_MOCK_PLATE:
                    plate = MOCK_PLATE
            except Exception:
                logger.warning("OCR failed for track %s, using mock plate", track_id)

        print(
            f"[VIOLATION] Track {track_id} exceeded {MAX_DWELL_SECONDS}s "
            f"in ROI ({dwell_seconds:.1f}s), plate={plate}"
        )

        # Capture the full annotated frame (bounding boxes + ROI already drawn)
        # as evidence instead of just the vehicle crop.
        evidence_frame = frame.copy()

        # Fire report in a background thread so the stream doesn't stall
        threading.Thread(
            target=self._report_violation,
            args=(evidence_frame, plate, dwell_seconds),
            daemon=True,
        ).start()

    # ── Per-frame processing ─────────────────────────────────

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Run YOLO tracking, ROI checks, and violation logic on one frame."""
        current_time = time.time()

        # Draw ROI overlay (red polygon)
        self._draw_roi(frame)

        # YOLO tracking with persistent IDs across frames
        results = self._model.track(
            frame,
            persist=True,
            classes=VEHICLE_CLASSES,
            verbose=False,
        )

        active_ids: set[int] = set()

        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                if box.id is None:
                    continue

                track_id = int(box.id.item())
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                active_ids.add(track_id)

                # Bottom-center of bounding box for ROI point-in-polygon test
                bottom_center = (int((x1 + x2) / 2), int(y2))
                inside = self._is_inside_roi(bottom_center)

                # Bounding box: red if inside ROI, green if outside
                color = (0, 0, 255) if inside else (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.circle(frame, bottom_center, 4, (255, 255, 0), -1)

                if inside:
                    # Start / continue dwell timer
                    if track_id not in self._tracked:
                        self._tracked[track_id] = {
                            "first_seen": current_time,
                            "is_reported": False,
                        }
                        print(f"[ROI] Track {track_id} entered restricted zone")

                    state = self._tracked[track_id]
                    elapsed = current_time - float(state["first_seen"])

                    label = f"ID {track_id} | {elapsed:.1f}s"
                    cv2.putText(
                        frame,
                        label,
                        (x1, max(y1 - 8, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 0, 255),
                        2,
                    )

                    # Violation threshold reached
                    if elapsed > MAX_DWELL_SECONDS and not state["is_reported"]:
                        state["is_reported"] = True
                        self._handle_violation(
                            frame, track_id, (x1, y1, x2, y2), elapsed
                        )
                else:
                    # Vehicle outside ROI — green label, clear dwell state
                    label = f"ID {track_id}"
                    cv2.putText(
                        frame,
                        label,
                        (x1, max(y1 - 8, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 0),
                        2,
                    )
                    if track_id in self._tracked:
                        print(f"[ROI] Track {track_id} left restricted zone")
                        del self._tracked[track_id]

        # Purge tracks that disappeared from YOLO detections
        stale = [tid for tid in self._tracked if tid not in active_ids]
        for tid in stale:
            print(f"[TRACK] Track {tid} lost, clearing state")
            del self._tracked[tid]

        # Periodic status log
        if self._frame_idx % 30 == 0:
            print(
                f"[FRAME {self._frame_idx}] "
                f"active_tracks={len(active_ids)} "
                f"tracked_in_roi={len(self._tracked)}"
            )

        return frame

    # ── MJPEG generator ──────────────────────────────────────

    def generate_frames(self) -> Generator[bytes, None, None]:
        """Yield AI-annotated MJPEG frames with full violation logic.

        Acquires a lock so concurrent callers wait rather than corrupt
        the YOLO tracker's internal state.
        """
        if not self._lock.acquire(blocking=False):
            logger.warning("Pipeline already active — waiting for previous stream to end")
            self._lock.acquire()

        try:
            while True:
                cap = cv2.VideoCapture(str(VIDEO_PATH))
                if not cap.isOpened():
                    logger.error("Cannot open video: %s", VIDEO_PATH)
                    break

                print(f"[AI PIPELINE] Streaming from {VIDEO_PATH}")

                while True:
                    ok, raw_frame = cap.read()
                    if not ok:
                        break  # EOF → loop video

                    # ── Crop blurred padding from vertical video ──
                    h_raw = raw_frame.shape[0]
                    frame = raw_frame[
                        int(h_raw * CROP_TOP_RATIO) : int(h_raw * CROP_BOTTOM_RATIO), :
                    ]

                    self._frame_idx += 1
                    self._maybe_reload_roi()

                    frame = self._process_frame(frame)

                    _, buffer = cv2.imencode(
                        ".jpg",
                        frame,
                        [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
                    )
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n"
                        + buffer.tobytes()
                        + b"\r\n"
                    )
                    time.sleep(FRAME_INTERVAL)

                cap.release()
                # Reset dwell state for the next video loop iteration
                self._tracked.clear()
                print("[AI PIPELINE] Video ended, looping...")

        finally:
            self._lock.release()


# ── Module-level singleton (lazy, thread-safe) ───────────────
_pipeline: SmartParkingPipeline | None = None
_init_lock = threading.Lock()


def get_pipeline() -> SmartParkingPipeline:
    """Return (or create) the singleton pipeline instance."""
    global _pipeline
    if _pipeline is None:
        with _init_lock:
            if _pipeline is None:  # double-checked locking
                _pipeline = SmartParkingPipeline()
    return _pipeline
