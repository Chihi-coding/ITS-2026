"""YOLOv8 vehicle and license plate detection."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from ultralytics import YOLO

logger = logging.getLogger(__name__)

# COCO vehicle classes used by default YOLOv8 weights.
VEHICLE_CLASS_IDS = {2, 3, 5, 7}  # car, motorcycle, bus, truck


@dataclass
class Detection:
    bbox: tuple[int, int, int, int]
    confidence: float
    class_id: int
    class_name: str


class VehicleDetector:
    """Handles YOLOv8 inference for vehicles and optional license plates."""

    def __init__(
        self,
        vehicle_model_path: str | Path = "yolov8n.pt",
        plate_model_path: str | Path | None = None,
        confidence_threshold: float = 0.45,
        device: str | None = None,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.vehicle_model = YOLO(str(vehicle_model_path))
        self.plate_model = YOLO(str(plate_model_path)) if plate_model_path else None
        self.device = device
        logger.info("Vehicle detector initialized (vehicle=%s, plate=%s)", vehicle_model_path, plate_model_path)

    def _parse_results(self, results: Any, allowed_classes: set[int] | None = None) -> list[Detection]:
        detections: list[Detection] = []
        if not results:
            return detections

        result = results[0]
        if result.boxes is None:
            return detections

        names = result.names or {}
        for box in result.boxes:
            class_id = int(box.cls.item())
            if allowed_classes is not None and class_id not in allowed_classes:
                continue

            confidence = float(box.conf.item())
            if confidence < self.confidence_threshold:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            detections.append(
                Detection(
                    bbox=(x1, y1, x2, y2),
                    confidence=confidence,
                    class_id=class_id,
                    class_name=str(names.get(class_id, class_id)),
                )
            )
        return detections

    def detect_vehicles(self, frame: np.ndarray) -> list[Detection]:
        """Run vehicle detection on a BGR frame."""
        try:
            results = self.vehicle_model.predict(
                source=frame,
                verbose=False,
                device=self.device,
            )
            return self._parse_results(results, allowed_classes=VEHICLE_CLASS_IDS)
        except Exception:
            logger.exception("Vehicle detection failed")
            return []

    def detect_plates(self, frame: np.ndarray) -> list[Detection]:
        """Run license plate detection when a dedicated model is configured."""
        if self.plate_model is None:
            return []

        try:
            results = self.plate_model.predict(
                source=frame,
                verbose=False,
                device=self.device,
            )
            return self._parse_results(results)
        except Exception:
            logger.exception("License plate detection failed")
            return []

    def detect_plates_in_vehicle_crop(
        self,
        frame: np.ndarray,
        vehicle_bbox: tuple[int, int, int, int],
    ) -> list[Detection]:
        """Detect plates inside a vehicle crop and map coordinates back to the full frame."""
        x1, y1, x2, y2 = vehicle_bbox
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return []

        plates = self.detect_plates(crop)
        mapped: list[Detection] = []
        for plate in plates:
            px1, py1, px2, py2 = plate.bbox
            mapped.append(
                Detection(
                    bbox=(px1 + x1, py1 + y1, px2 + x1, py2 + y1),
                    confidence=plate.confidence,
                    class_id=plate.class_id,
                    class_name=plate.class_name,
                )
            )
        return mapped
