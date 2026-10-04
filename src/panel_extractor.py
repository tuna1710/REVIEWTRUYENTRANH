"""
Module 2: Manga Panel Extractor & Smart RTL Reading Order
Detects individual manga panels using YOLO (fine-tuned on Manga109) with OpenCV fallback,
and sorts them according to Right-to-Left (RTL) Japanese manga reading order.
"""

import os
import cv2
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from PIL import Image

try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except ImportError:
    HAS_ULTRALYTICS = False


class MangaPanelExtractor:
    def __init__(
        self,
        model_name: str = "leoxs22/manga-panel-detector-yolo26n",
        confidence: float = 0.45,
        confidence_threshold: Optional[float] = None,
        output_dir: str = "./workspace/panels",
        reading_order: str = "RTL"
    ):
        self.model_name = model_name
        self.confidence = confidence_threshold if confidence_threshold is not None else confidence
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.reading_order = reading_order.upper()
        self.model = None

        if HAS_ULTRALYTICS:
            try:
                print(f"[Panel Extractor] Initializing YOLO model: {self.model_name}...")
                model_file = self.model_name
                # Handle Hugging Face repository identifiers
                if "/" in self.model_name and not os.path.exists(self.model_name):
                    try:
                        from huggingface_hub import hf_hub_download
                        if "manga-panel-detector" in self.model_name:
                            model_file = hf_hub_download(
                                repo_id=self.model_name,
                                filename="manga_panel_detector_fp32.pt"
                            )
                        else:
                            model_file = hf_hub_download(
                                repo_id=self.model_name,
                                filename="best.pt"
                            )
                        print(f"[Panel Extractor] Downloaded YOLO weights from HF: {model_file}")
                    except Exception as hf_e:
                        print(f"[Panel Extractor] HuggingFace download note: {hf_e}")

                self.model = YOLO(model_file)
            except Exception as e:
                print(f"[Panel Extractor] Warning: Could not load YOLO weights ({e}). OpenCV fallback enabled.")

    def process_all_pages(self, page_paths: List[str]) -> List[Dict]:
        """
        Processes a list of page images, extracts panels, and returns sorted metadata.
        """
        all_panels = []
        global_panel_idx = 1

        for page_idx, page_path in enumerate(page_paths, start=1):
            print(f"[Panel Extractor] Processing page {page_idx}/{len(page_paths)}: {Path(page_path).name}")
            page_img = cv2.imread(page_path)
            if page_img is None:
                continue

            h, w = page_img.shape[:2]
            boxes = self._detect_panels(page_img)

            # Sort boxes according to manga reading order
            sorted_boxes = self._sort_reading_order(boxes, page_width=w, page_height=h)

            # Crop and save each panel
            for b_idx, box in enumerate(sorted_boxes, start=1):
                x1, y1, x2, y2 = box
                # Add slight margin & clamp to page borders
                x1 = max(0, x1 - 2)
                y1 = max(0, y1 - 2)
                x2 = min(w, x2 + 2)
                y2 = min(h, y2 + 2)

                panel_crop = page_img[y1:y2, x1:x2]
                if panel_crop.size == 0:
                    continue

                panel_filename = f"panel_{global_panel_idx:04d}.png"
                panel_path = self.output_dir / panel_filename
                cv2.imwrite(str(panel_path), panel_crop)

                panel_meta = {
                    "panel_id": global_panel_idx,
                    "file_path": str(panel_path),
                    "file_name": panel_filename,
                    "source_page": Path(page_path).name,
                    "bbox": [x1, y1, x2, y2],
                    "width": x2 - x1,
                    "height": y2 - y1,
                    "aspect_ratio": round((x2 - x1) / max(1, (y2 - y1)), 2)
                }
                all_panels.append(panel_meta)
                global_panel_idx += 1

        # Save metadata index
        meta_file = self.output_dir / "panels_index.json"
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(all_panels, f, ensure_ascii=False, indent=2)

        print(f"[Panel Extractor] Completed. Extracted total {len(all_panels)} panels into {self.output_dir}")
        return all_panels

    def _detect_panels(self, img: np.ndarray) -> List[List[int]]:
        """
        Detects panel bounding boxes using YOLO or OpenCV contour detection.
        """
        h, w = img.shape[:2]
        boxes = []

        if self.model is not None:
            try:
                results = self.model.predict(source=img, conf=self.confidence, verbose=False)
                for box in results[0].boxes:
                    cls_id = int(box.cls[0])
                    label = self.model.names[cls_id]
                    # We only care about panels (ignore standalone bubble / character detections)
                    if "panel" in label.lower() or label == "0":
                        xyxy = list(map(int, box.xyxy[0]))
                        # Filter out tiny artifacts (<3% of total page area)
                        box_area = (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])
                        if box_area >= (h * w * 0.02):
                            boxes.append(xyxy)
                if boxes:
                    return boxes
            except Exception as e:
                print(f"[Panel Extractor] YOLO prediction failed ({e}), using OpenCV fallback.")

        # OpenCV Fallback
        return self._detect_panels_opencv(img)

    def _detect_panels_opencv(self, img: np.ndarray) -> List[List[int]]:
        """
        Contour-based fallback panel detector.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        # Threshold to find dark panel borders
        _, thresh = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY_INV)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        min_area = h * w * 0.03 # At least 3% of the page
        max_area = h * w * 0.96

        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            area = bw * bh
            if min_area <= area <= max_area:
                boxes.append([x, y, x + bw, y + bh])

        if not boxes:
            # If no panel border detected (e.g. splash art), treat full page as single panel
            boxes = [[0, 0, w, h]]

        return boxes

    def _sort_reading_order(self, boxes: List[List[int]], page_width: int, page_height: int) -> List[List[int]]:
        """
        Sorts boxes by Japanese Manga Reading Order:
        1. Top to Bottom (Rows/Tiers)
        2. Right to Left (RTL) within each row
        """
        if len(boxes) <= 1:
            return boxes

        # Sort primarily by vertical center
        boxes_with_center = []
        for b in boxes:
            yc = (b[1] + b[3]) / 2.0
            xc = (b[0] + b[2]) / 2.0
            boxes_with_center.append((b, xc, yc))

        # Group into rows based on vertical overlap tolerance
        row_threshold = page_height * 0.08
        boxes_with_center.sort(key=lambda item: item[2]) # Sort by Y center

        rows = []
        current_row = [boxes_with_center[0]]

        for item in boxes_with_center[1:]:
            prev_y = current_row[-1][2]
            if abs(item[2] - prev_y) <= row_threshold:
                current_row.append(item)
            else:
                rows.append(current_row)
                current_row = [item]
        if current_row:
            rows.append(current_row)

        sorted_boxes = []
        for row in rows:
            if self.reading_order == "RTL":
                # Right to Left: Highest X coordinate comes first
                row.sort(key=lambda item: item[1], reverse=True)
            else:
                # Left to Right: Lowest X coordinate comes first
                row.sort(key=lambda item: item[1], reverse=False)

            for item in row:
                sorted_boxes.append(item[0])

        return sorted_boxes
