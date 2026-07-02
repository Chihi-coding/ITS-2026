"""
Smart Parking AI orchestrator.

Detects vehicles with YOLOv8 tracking, monitors ROI dwell time,
runs OCR on violation crops, and posts alerts to the FastAPI backend.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import requests
from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai_module.core.ocr_engine import OCREngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("ai_module.main")

ROI_PATH = ROOT_DIR / "config" / "roi.json"
MODEL_PATH = "yolov8n.pt"
API_URL = "http://localhost:8000/api/violations"
MAX_DWELL_SECONDS = 5
VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck
USE_MOCK_PLATE_IF_UNKNOWN = True
MOCK_PLATE = "30F-12345"
WINDOW_NAME = "Smart Parking Monitor"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smart Parking AI pipeline with YOLO tracking and ROI dwell monitoring."
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Webcam index (e.g. 0) or path to a video file (default: 0).",
    )
    return parser.parse_args()


def resolve_video_source(source: str) -> str | int:
    """Convert CLI source string to an OpenCV-compatible capture target."""
    if source.isdigit():
        return int(source)

    path = Path(source)
    if path.exists():
        return str(path.resolve())

    raise FileNotFoundError(f"Video source not found: {source}")


def load_roi_polygon(roi_path: Path) -> np.ndarray:
    """Load ROI polygon coordinates from JSON."""
    if not roi_path.exists():
        raise FileNotFoundError(f"ROI config not found: {roi_path}")

    with roi_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    # Support both plain list [[x,y],...] and dict {"polygon": [[x,y],...]} formats
    if isinstance(data, dict):
        data = data.get("polygon", [])

    if not isinstance(data, list) or len(data) < 3:
        raise ValueError("roi.json must contain a polygon array with at least 3 points")

    polygon = np.array(data, dtype=np.int32)
    logger.info("Loaded ROI polygon with %s points", len(polygon))
    print(f"[ROI] Polygon coordinates: {polygon.tolist()}")
    return polygon


def bottom_center_point(x1: int, y1: int, x2: int, y2: int) -> tuple[int, int]:
    """Return bottom-center point of a bounding box."""
    return int((x1 + x2) / 2), int(y2)


def is_inside_roi(point: tuple[int, int], polygon: np.ndarray) -> bool:
    """Return True when point is inside or on the ROI boundary."""
    return cv2.pointPolygonTest(polygon, point, False) >= 0


def draw_roi(frame: np.ndarray, polygon: np.ndarray) -> None:
    """Draw ROI polygon on frame in red."""
    cv2.polylines(frame, [polygon], isClosed=True, color=(0, 0, 255), thickness=2)
    overlay = frame.copy()
    cv2.fillPoly(overlay, [polygon], color=(0, 0, 255))
    cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)


def crop_vehicle(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    """Safely crop a vehicle region from the frame."""
    h, w = frame.shape[:2]
    x1 = max(0, min(w, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h, y1))
    y2 = max(0, min(h, y2))
    return frame[y1:y2, x1:x2]


def encode_jpeg(image: np.ndarray) -> bytes:
    """Encode an OpenCV image as JPEG bytes."""
    success, buffer = cv2.imencode(".jpg", image)
    if not success:
        raise RuntimeError("Failed to encode image to JPEG")
    return buffer.tobytes()


def report_violation(
    plate_number: str,
    image_bytes: bytes,
    timestamp: datetime,
    duration_seconds: float = 0,
) -> bool:
    """POST violation payload to FastAPI backend (triggers Supabase DB insert)."""
    payload = {
        "plate_number": plate_number,
        "timestamp": timestamp.isoformat(),
        "duration_seconds": duration_seconds,
    }
    files = {
        "image": ("violation.jpg", BytesIO(image_bytes), "image/jpeg"),
    }

    try:
        logger.info("Sending violation to backend: plate=%s", plate_number)
        response = requests.post(API_URL, data=payload, files=files, timeout=20)
        response.raise_for_status()
        result = response.json()
        logger.info("Backend accepted violation: %s", result)
        print(f"DB Insert Success: plate={plate_number}, response={result}")
        return True
    except requests.RequestException as exc:
        logger.exception("Failed to send violation to backend")
        print(f"DB Error: {exc}")
        return False
    except Exception as exc:
        logger.exception("Unexpected error during violation report")
        print(f"DB Error: unexpected failure — {exc}")
        return False


def resolve_plate_text(ocr_engine: OCREngine, crop: np.ndarray) -> str:
    """Run OCR on crop, optionally fall back to mock plate for demo speed."""
    plate_text = ocr_engine.read_plate(crop)
    if plate_text != "UNKNOWN":
        return plate_text
    if USE_MOCK_PLATE_IF_UNKNOWN:
        logger.info("OCR returned UNKNOWN, using mock plate: %s", MOCK_PLATE)
        print(f"[OCR] No plate detected, using mock plate: {MOCK_PLATE}")
        return MOCK_PLATE
    return "UNKNOWN"


def open_video_capture(source: str | int) -> cv2.VideoCapture:
    """Open webcam index or video file path."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video source: {source}")
    return cap


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("Smart Parking AI Module - Starting pipeline")
    print("=" * 60)

    roi_polygon = load_roi_polygon(ROI_PATH)
    print(f"[INIT] ROI loaded from {ROI_PATH}")

    logger.info("Loading YOLO model: %s", MODEL_PATH)
    model = YOLO(MODEL_PATH)
    print(f"[INIT] YOLO model loaded: {MODEL_PATH}")

    ocr_engine = OCREngine(languages=["en"])
    print("[INIT] OCR engine ready")

    try:
        video_source = resolve_video_source(args.source)
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)

    cap = open_video_capture(video_source)
    print(f"[INIT] Video source opened: {video_source}")

    tracked_vehicles: dict[int, dict[str, float | bool]] = {}
    frame_index = 0
    roi_last_mtime: float = os.path.getmtime(ROI_PATH) if ROI_PATH.exists() else 0.0

    # Create a resizable window (scaled to fit screen on first frame)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    window_sized = False

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("[STOP] End of video stream reached")
                break

            frame_index += 1
            current_time = time.time()

            # ── Hot-reload ROI every 30 frames ──────────────────
            if frame_index % 30 == 0 and ROI_PATH.exists():
                try:
                    current_mtime = os.path.getmtime(ROI_PATH)
                    if current_mtime != roi_last_mtime:
                        roi_polygon = load_roi_polygon(ROI_PATH)
                        roi_last_mtime = current_mtime
                        tracked_vehicles.clear()
                        print(f"[ROI] Hot-reloaded polygon from {ROI_PATH} (mtime={current_mtime})")
                except Exception as reload_exc:
                    logger.warning("ROI hot-reload failed (file may be mid-write): %s", reload_exc)

            # Auto-scale window to fit screen on first frame
            if not window_sized:
                fh, fw = frame.shape[:2]
                max_h = 800
                scale = max_h / fh
                cv2.resizeWindow(WINDOW_NAME, int(fw * scale), max_h)
                print(f"[INIT] Display scaled to {int(fw * scale)}x{max_h} (scale={scale:.3f})")
                window_sized = True

            draw_roi(frame, roi_polygon)

            results = model.track(
                frame,
                persist=True,
                classes=VEHICLE_CLASSES,
                verbose=False,
            )

            active_track_ids: set[int] = set()

            if results and results[0].boxes is not None:
                boxes = results[0].boxes
                for box in boxes:
                    if box.id is None:
                        continue

                    track_id = int(box.id.item())
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    active_track_ids.add(track_id)

                    point = bottom_center_point(x1, y1, x2, y2)
                    inside = is_inside_roi(point, roi_polygon)

                    if frame_index % 30 == 1:
                        print(
                            f"[DEBUG] Track {track_id}: "
                            f"bbox=({x1},{y1},{x2},{y2}) "
                            f"bottom_center={point} "
                            f"inside_roi={inside}"
                        )

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(frame, point, 4, (255, 255, 0), -1)

                    if inside:
                        if track_id not in tracked_vehicles:
                            tracked_vehicles[track_id] = {
                                "first_seen": current_time,
                                "is_reported": False,
                            }
                            print(f"[ROI] Track {track_id} entered restricted zone")

                        state = tracked_vehicles[track_id]
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

                        if elapsed > MAX_DWELL_SECONDS and not state["is_reported"]:
                            print(
                                f"[VIOLATION] Track {track_id} exceeded "
                                f"{MAX_DWELL_SECONDS}s in ROI ({elapsed:.1f}s)"
                            )

                            crop = crop_vehicle(frame, x1, y1, x2, y2)
                            plate_number = resolve_plate_text(ocr_engine, crop)
                            image_bytes = encode_jpeg(crop if crop.size else frame)
                            timestamp = datetime.now(timezone.utc)

                            if report_violation(
                                plate_number,
                                image_bytes,
                                timestamp,
                                duration_seconds=elapsed,
                            ):
                                state["is_reported"] = True
                                tracked_vehicles[track_id] = state
                    else:
                        if track_id in tracked_vehicles:
                            print(f"[ROI] Track {track_id} left restricted zone")
                            del tracked_vehicles[track_id]

            stale_ids = [
                track_id
                for track_id in list(tracked_vehicles.keys())
                if track_id not in active_track_ids
            ]
            for track_id in stale_ids:
                print(f"[TRACK] Track {track_id} lost, clearing state")
                del tracked_vehicles[track_id]

            if frame_index % 30 == 0:
                print(
                    f"[FRAME {frame_index}] active_tracks={len(active_track_ids)} "
                    f"tracked_in_roi={len(tracked_vehicles)}"
                )

            cv2.imshow(WINDOW_NAME, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[STOP] Exit requested by user (q pressed)")
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[DONE] AI pipeline stopped")


if __name__ == "__main__":
    main()
