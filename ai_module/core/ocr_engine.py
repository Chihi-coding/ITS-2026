"""License plate OCR using EasyOCR."""

from __future__ import annotations

import logging
import re
import cv2
import numpy as np
import easyocr

logger = logging.getLogger(__name__)

class OCREngine:
    """Extract plate text from cropped vehicle or plate images."""

    def __init__(self, plate_country: str = "AUTO") -> None:
        self.plate_country = plate_country.upper()
        # "ch_sim" for simplified Chinese, "en" for alphanumeric
        lang = ["en", "ch_sim"] if self.plate_country in ["CN", "AUTO"] else ["en"]
        logger.info("Initializing EasyOCR reader for languages: %s", lang)
        self.reader = easyocr.Reader(lang, gpu=False)

    @staticmethod
    def _normalize_text(raw_text: str, allow_chinese: bool) -> str:
        cleaned = raw_text.upper().strip()
        if allow_chinese:
            # allow Chinese characters, A-Z, 0-9, and dash/dot
            cleaned = re.sub(r"[^\u4e00-\u9fa5A-Z0-9\-.\s]", "", cleaned)
        else:
            cleaned = re.sub(r"[^A-Z0-9\-.\s]", "", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned

    def validate_format(self, text: str) -> str:
        """Returns the status string (e.g. valid, UNSUPPORTED FORMAT, FOREIGN PLATE)."""
        if not text or text == "UNKNOWN":
            return "OCR FAILED"

        is_vn_format = bool(re.match(r"^[0-9]{2}[A-Z]-[0-9]{3,5}$", text.replace(" ", ""))) or \
                       bool(re.match(r"^[0-9]{2}[A-Z][0-9]{4,5}$", text.replace(" ", "")))
        
        has_chinese = bool(re.search(r"[\u4e00-\u9fa5]", text))

        if self.plate_country == "VN":
            if has_chinese or not is_vn_format:
                return "FOREIGN PLATE" if has_chinese else "UNSUPPORTED FORMAT"
            return "VALID"
        elif self.plate_country == "CN":
            if not has_chinese:
                return "UNSUPPORTED FORMAT"
            return "VALID"
        
        # AUTO accepts anything that looks roughly like a plate
        if len(text) < 4:
            return "UNSUPPORTED FORMAT"
        return "VALID"

    def _preprocess_crop(self, image_crop: np.ndarray) -> list[np.ndarray]:
        """Generate different preprocessed versions of the crop for OCR."""
        crops = [image_crop]
        
        # Grayscale
        gray = cv2.cvtColor(image_crop, cv2.COLOR_BGR2GRAY)
        crops.append(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
        
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl1 = clahe.apply(gray)
        crops.append(cv2.cvtColor(cl1, cv2.COLOR_GRAY2BGR))
        
        # Mild sharpening
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(image_crop, -1, kernel)
        crops.append(sharpened)
        
        return crops

    def read_plate(self, image_crop: np.ndarray) -> tuple[str, float, str]:
        """
        Read plate characters from a cropped image.
        Returns: (plate_text, confidence, status)
        """
        if image_crop is None or image_crop.size == 0:
            logger.warning("OCR received an empty crop")
            return "UNKNOWN", 0.0, "EMPTY CROP"

        allow_chinese = self.plate_country in ["CN", "AUTO"]
        best_text = "UNKNOWN"
        best_conf = 0.0
        
        crops_to_test = self._preprocess_crop(image_crop)
        
        for crop in crops_to_test:
            try:
                results = self.reader.readtext(crop, detail=1, paragraph=False)
                # results is a list of tuples: (bbox, text, prob)
                if results:
                    # Sort by probability or just join them
                    combined = " ".join([res[1] for res in results]).strip()
                    avg_prob = sum([res[2] for res in results]) / len(results)
                    
                    plate_text = self._normalize_text(combined, allow_chinese)
                    if plate_text and avg_prob > best_conf:
                        best_text = plate_text
                        best_conf = avg_prob
            except Exception as e:
                logger.debug(f"OCR failed on a preprocessed crop: {e}")

        if best_text != "UNKNOWN":
            status = self.validate_format(best_text)
            logger.info("OCR result: %s (conf: %.2f, status: %s)", best_text, best_conf, status)
            return best_text, best_conf, status
            
        logger.warning("OCR found no readable text")
        return "UNKNOWN", 0.0, "OCR FAILED"
