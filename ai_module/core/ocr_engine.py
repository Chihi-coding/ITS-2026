"""License plate OCR using EasyOCR."""

from __future__ import annotations

import logging
import re

import easyocr
import numpy as np

logger = logging.getLogger(__name__)


class OCREngine:
    """Extract plate text from cropped vehicle or plate images."""

    def __init__(self, languages: list[str] | None = None) -> None:
        lang = languages or ["en"]
        logger.info("Initializing EasyOCR reader for languages: %s", lang)
        self.reader = easyocr.Reader(lang, gpu=False)

    @staticmethod
    def _normalize_text(raw_text: str) -> str:
        cleaned = raw_text.upper().strip()
        cleaned = re.sub(r"[^A-Z0-9\-.\s]", "", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned

    def read_plate(self, image_crop: np.ndarray) -> str:
        """
        Read plate characters from a cropped image.

        Returns extracted text, or "UNKNOWN" when nothing is detected.
        """
        if image_crop is None or image_crop.size == 0:
            logger.warning("OCR received an empty crop")
            return "UNKNOWN"

        try:
            results = self.reader.readtext(image_crop, detail=0, paragraph=True)
            combined = " ".join(results).strip()
            plate_text = self._normalize_text(combined)
            if plate_text:
                logger.info("OCR result: %s", plate_text)
                return plate_text
            logger.warning("OCR found no readable text")
            return "UNKNOWN"
        except Exception:
            logger.exception("OCR failed on crop")
            return "UNKNOWN"
