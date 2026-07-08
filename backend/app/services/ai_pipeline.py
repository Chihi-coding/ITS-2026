"""Unified AI pipeline: YOLO tracking + ROI dwell + violation reporting."""

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
import collections

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
DEBUG_DIR = _PROJECT_ROOT / "backend" / "debug_images"
DEBUG_DIR.mkdir(parents=True, exist_ok=True)

# ── Detection / streaming constants ─────────────────────────
VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck
MAX_DWELL_SECONDS = 5
JPEG_QUALITY = 70
TARGET_FPS = 20
FRAME_INTERVAL = 1.0 / TARGET_FPS
ROI_CHECK_EVERY = 5  # check roi.json mtime every N frames
VIOLATION_BUCKET = "violation-images"
USE_MOCK_PLATE = False
MOCK_PLATE = "30F-12345"

# ── Helpers ──────────────────────────────────────────────────

def _safe_filename(timestamp: str, plate: str) -> str:
    ts = re.sub(r"[^\w\-]", "_", timestamp)
    pl = re.sub(r"[^\w\-]", "_", plate)
    return f"{ts}_{pl}.jpg"


# ── Pipeline ─────────────────────────────────────────────────

class SmartParkingPipeline:
    def __init__(self) -> None:
        logger.info("Initializing SmartParkingPipeline...")
        print("[AI PIPELINE] Loading YOLO model...")
        self._model = YOLO(MODEL_PATH)
        print(f"[AI PIPELINE] YOLO model loaded: {MODEL_PATH}")

        self._ocr = self._init_ocr()

        self._roi_polygon: np.ndarray | None = self._load_roi()
        self._roi_mtime: float = self._get_roi_mtime()

        # track_id → {"first_seen": float, "is_reported": bool, "crops": list}
        self._tracked: dict[int, dict] = {}
        self._frame_idx = 0

        self._lock = threading.Lock()
        print("[AI PIPELINE] Pipeline ready ✓")

    @staticmethod
    def _init_ocr():
        try:
            from ai_module.core.ocr_engine import OCREngine
            engine = OCREngine(plate_country="CN") # Enable Chinese
            print("[AI PIPELINE] OCR engine ready")
            return engine
        except Exception as exc:
            logger.warning("OCR engine unavailable: %s", exc)
            print(f"[AI PIPELINE] OCR unavailable ({exc})")
            return None

    @staticmethod
    def _load_roi() -> np.ndarray | None:
        if not ROI_PATH.exists():
            return None
        try:
            raw = json.loads(ROI_PATH.read_text(encoding="utf-8"))
            points = raw.get("polygon", raw) if isinstance(raw, dict) else raw
            if isinstance(points, list) and len(points) >= 3:
                polygon = np.array(points, dtype=np.int32)
                return polygon
        except Exception:
            pass
        return None

    @staticmethod
    def _get_roi_mtime() -> float:
        try:
            return os.path.getmtime(ROI_PATH) if ROI_PATH.exists() else 0.0
        except OSError:
            return 0.0

    def invalidate_roi(self) -> None:
        try:
            self._roi_polygon = self._load_roi()
            self._roi_mtime = self._get_roi_mtime()
            self._tracked.clear()
        except Exception as exc:
            pass

    def _maybe_reload_roi(self) -> None:
        if self._frame_idx % ROI_CHECK_EVERY != 0 or not ROI_PATH.exists():
            return
        try:
            mtime = os.path.getmtime(ROI_PATH)
            if mtime != self._roi_mtime:
                self._roi_polygon = self._load_roi()
                self._roi_mtime = mtime
                self._tracked.clear()
        except Exception as exc:
            pass

    def _draw_roi(self, frame: np.ndarray) -> None:
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

    def _report_violation(
        self, evidence_frame: np.ndarray, plate: str, dwell_seconds: float, track_id: int, status: str
    ) -> None:
        try:
            timestamp = datetime.now(timezone.utc).isoformat()

            ok, buf = cv2.imencode(".jpg", evidence_frame)
            if not ok:
                return
            image_bytes = buf.tobytes()

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

            result = (
                supabase.table("violations")
                .insert(
                    {
                        "license_plate": plate,
                        "detected_at": timestamp,
                        "evidence_image_path": image_url,
                        "duration_seconds": max(0, int(round(dwell_seconds))),
                        "telegram_sent": False,
                        "status": status,
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
    ) -> str:
        crops_list = self._tracked[track_id].get("crops", [])
        crops_list.sort(key=lambda x: x['sharpness'], reverse=True)
        top_crops = crops_list[:5]
        
        best_plate = "OCR FAILED"
        highest_conf = 0.0
        plate_status = "OCR FAILED"
        
        votes = collections.defaultdict(float)
        
        for idx, item in enumerate(top_crops):
            plate_crop = item["plate_crop"]
            if plate_crop is None: continue
            
            if self._ocr:
                try:
                    text, conf, status = self._ocr.read_plate(plate_crop)
                    print(f"[OCR_RAW] track={track_id}, text={text}, conf={conf:.2f}, status={status}")
                    if text != "UNKNOWN" and text:
                        votes[text] += conf
                        if conf > highest_conf:
                            highest_conf = conf
                            best_plate = text
                            plate_status = status
                            
                    # Save debug image
                    cv2.imwrite(str(DEBUG_DIR / f"track_{track_id}_crop_{idx}.jpg"), plate_crop)
                    if item.get("v_crop") is not None:
                        cv2.imwrite(str(DEBUG_DIR / f"track_{track_id}_vcrop_{idx}.jpg"), item["v_crop"])
                except Exception as e:
                    logger.error(f"OCR error: {e}")
                    
        if votes:
            best_plate = max(votes.items(), key=lambda x: x[1])[0]
            # re-validate the winning plate
            plate_status = self._ocr.validate_format(best_plate) if self._ocr else "OCR FAILED"
            print(f"[OCR_NORMALIZED] track={track_id}, normalized_text={best_plate}")
            print(f"[PLATE_ACCEPTED] track={track_id}, plate={best_plate}, status={plate_status}")
        else:
            print(f"[PLATE_REJECTED] track={track_id}, reason='No valid text found'")
            
        # Draw on frame before upload
        evidence_frame = frame.copy()
        
        threading.Thread(
            target=self._report_violation,
            args=(evidence_frame, best_plate, dwell_seconds, track_id, plate_status),
            daemon=True,
        ).start()
        
        return best_plate

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        current_time = time.time()
        self._draw_roi(frame)

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
                confidence = float(box.conf.item())
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                active_ids.add(track_id)
                
                bottom_center = (int((x1 + x2) / 2), int(y2))
                inside = self._is_inside_roi(bottom_center)

                if track_id not in self._tracked:
                    self._tracked[track_id] = {
                        "first_seen": current_time,
                        "is_reported": False,
                        "crops": [],
                        "final_plate": None
                    }
                    print(f"[VEHICLE] track_id={track_id}, conf={confidence:.2f}, bbox=({x1},{y1},{x2},{y2})")

                state = self._tracked[track_id]
                color = (0, 0, 255) if inside else (0, 255, 0)
                
                # --- Plate Localization per frame ---
                h, w = frame.shape[:2]
                x1c, y1c = max(0, x1), max(0, y1)
                x2c, y2c = min(w, x2), min(h, y2)
                
                if x2c > x1c and y2c > y1c and not state["is_reported"] and self._ocr:
                    v_crop = frame[y1c:y2c, x1c:x2c]
                    try:
                        # detect horizontal text boxes
                        det_res = self._ocr.reader.detect(v_crop)
                        bboxes = det_res[0][0]
                        if bboxes and len(bboxes) > 0:
                            p_xmin, p_xmax, p_ymin, p_ymax = map(int, bboxes[0])
                            p_xmin, p_ymin = max(0, p_xmin), max(0, p_ymin)
                            p_xmax, p_ymax = min(v_crop.shape[1], p_xmax), min(v_crop.shape[0], p_ymax)
                            
                            if p_xmax > p_xmin and p_ymax > p_ymin:
                                plate_crop = v_crop[p_ymin:p_ymax, p_xmin:p_xmax].copy()
                                gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
                                sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
                                
                                # Log it only once a second roughly to not spam the console
                                if self._frame_idx % 20 == 0:
                                    print(f"[PLATE_DETECT] track={track_id}, bbox=({p_xmin},{p_ymin},{p_xmax},{p_ymax}), sharpness={sharpness:.1f}")
                                    print(f"[PLATE_CROP] dims={plate_crop.shape}")
                                
                                state["crops"].append({
                                    "sharpness": sharpness,
                                    "plate_crop": plate_crop,
                                    "v_crop": v_crop.copy()
                                })
                                
                                # Draw plate bbox on video
                                px1 = x1c + p_xmin
                                py1 = y1c + p_ymin
                                px2 = x1c + p_xmax
                                py2 = y1c + p_ymax
                                cv2.rectangle(frame, (px1, py1), (px2, py2), (255, 0, 0), 2)
                    except Exception as e:
                        pass
                
                elapsed = current_time - float(state["first_seen"])
                
                if inside:
                    if not state["is_reported"] and elapsed > MAX_DWELL_SECONDS:
                        state["is_reported"] = True
                        final_plate = self._handle_violation(frame, track_id, (x1, y1, x2, y2), elapsed)
                        state["final_plate"] = final_plate

                # Determine label
                disp_plate = state.get("final_plate")
                if disp_plate:
                    label = f"ID {track_id} | {disp_plate} | {elapsed:.1f}s"
                else:
                    label = f"ID {track_id} | {elapsed:.1f}s"
                    
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.circle(frame, bottom_center, 4, (255, 255, 0), -1)
                cv2.putText(frame, label, (x1, max(y1 - 8, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        stale = [tid for tid in self._tracked if tid not in active_ids]
        for tid in stale:
            del self._tracked[tid]

        return frame

    def generate_frames(self) -> Generator[bytes, None, None]:
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

                    self._frame_idx += 1
                    self._maybe_reload_roi()

                    # Removed cropping logic to keep full 1280x720 video
                    frame = self._process_frame(raw_frame.copy())

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
                self._tracked.clear()
                print("[AI PIPELINE] Video ended, looping...")

        finally:
            self._lock.release()

_pipeline: SmartParkingPipeline | None = None
_init_lock = threading.Lock()

def get_pipeline() -> SmartParkingPipeline:
    global _pipeline
    if _pipeline is None:
        with _init_lock:
            if _pipeline is None:
                _pipeline = SmartParkingPipeline()
    return _pipeline
