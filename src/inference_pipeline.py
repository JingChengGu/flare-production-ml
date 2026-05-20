"""
FLARE Inference Pipeline
========================
Two-stage EV charger fault detection.

Stage 1 — SegFormer B3:
    Segments full charger image into component regions
    (screen, body, cable, plug)

Stage 2 — ViT Classifiers:
    Each component crop is independently classified as healthy/broken.
    Body segment uses the OOS (out-of-service) classifier.
"""

import time
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F
from transformers import (
    SegformerForSemanticSegmentation,
    SegformerImageProcessor,
    AutoModelForImageClassification,
    AutoImageProcessor,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

COMPONENT_MAP: dict[int, str] = {
    1: "screen",
    2: "body",
    3: "cable",
    4: "plug",
}

SEG_PALETTE: dict[int, tuple] = {
    0: (0,   0,   0),
    1: (255, 0,   0),
    2: (255, 255, 0),
    3: (0,   0,   255),
    4: (0,   255, 0),
    5: (128, 0,   128),
}

SEGFORMER_MODEL_ID = "JaesonGu/flare-segformer-mit-b3"

CLASSIFIER_MODEL_IDS: dict[str, str] = {
    "screen": "JaesonGu/flare-screen-vit",
    "body":   "JaesonGu/flare-body-vit",
    "cable":  "JaesonGu/flare-cable-vit",
    "plug":   "JaesonGu/flare-plug-vit",
}

REVIEW_THRESHOLD = 0.70
CROP_PADDING = 8


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class FLAREPipeline:

    def __init__(self):
        logger.info("Loading SegFormer B3 from %s", SEGFORMER_MODEL_ID)
        self.seg_processor = SegformerImageProcessor.from_pretrained(SEGFORMER_MODEL_ID)
        self.seg_model = SegformerForSemanticSegmentation.from_pretrained(SEGFORMER_MODEL_ID)
        self.seg_model.eval()

        logger.info("Loading ViT classifiers...")
        self.classifiers: dict[str, AutoModelForImageClassification] = {}
        self.clf_processors: dict[str, AutoImageProcessor] = {}

        for component, model_id in CLASSIFIER_MODEL_IDS.items():
            logger.info("  Loading %s classifier from %s", component, model_id)
            self.clf_processors[component] = AutoImageProcessor.from_pretrained(model_id)
            self.classifiers[component] = AutoModelForImageClassification.from_pretrained(model_id)
            self.classifiers[component].eval()

        logger.info("All models loaded successfully.")

    def _run_segmentation(self, image: Image.Image) -> np.ndarray:
        inputs = self.seg_processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.seg_model(**inputs)
        logits = outputs.logits
        upsampled = F.interpolate(
            logits,
            size=(image.height, image.width),
            mode="bilinear",
            align_corners=False,
        )
        return upsampled.argmax(dim=1)[0].numpy()

    def _render_seg_mask(self, image: Image.Image, seg_mask: np.ndarray, alpha: float = 0.5) -> Image.Image:
        h, w = seg_mask.shape
        color_mask = np.zeros((h, w, 3), dtype=np.uint8)
        for class_id, color in SEG_PALETTE.items():
            color_mask[seg_mask == class_id] = color
        color_mask_img = Image.fromarray(color_mask, mode="RGB")
        overlay = Image.blend(image.convert("RGB"), color_mask_img, alpha=alpha)

        draw = ImageDraw.Draw(overlay)
        legend = [
            ("screen",     SEG_PALETTE[1]),
            ("body (OOS)", SEG_PALETTE[2]),
            ("cable",      SEG_PALETTE[3]),
            ("plug",       SEG_PALETTE[4]),
            ("background", SEG_PALETTE[5]),
        ]
        x, y, box, gap = 10, 10, 16, 4
        for label, color in legend:
            draw.rectangle([x, y, x + box, y + box], fill=color)
            draw.text((x + box + gap, y), label, fill=(255, 255, 255))
            y += box + gap

        return overlay

    def _extract_crop(self, image: Image.Image, seg_mask: np.ndarray, class_id: int) -> Optional[Image.Image]:
        mask = (seg_mask == class_id)
        if mask.sum() == 0:
            return None
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        rmin = max(0, rmin - CROP_PADDING)
        rmax = min(image.height, rmax + CROP_PADDING)
        cmin = max(0, cmin - CROP_PADDING)
        cmax = min(image.width, cmax + CROP_PADDING)
        return image.crop((cmin, rmin, cmax, rmax))

    def _classify_component(self, component: str, crop: Image.Image) -> tuple[str, float]:
        processor = self.clf_processors[component]
        model = self.classifiers[component]
        inputs = processor(images=crop, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0]
        predicted_idx = probs.argmax().item()
        confidence = probs[predicted_idx].item()
        label = model.config.id2label[predicted_idx]
        return label, round(confidence, 4)

    def _save_debug_output(self, image, seg_mask, crops, results, output_dir):
        output_dir.mkdir(parents=True, exist_ok=True)
        crops_dir = output_dir / "crops"
        crops_dir.mkdir(exist_ok=True)

        self._render_seg_mask(image, seg_mask, alpha=0.5).save(output_dir / "seg_mask_overlay.png")
        self._render_seg_mask(image, seg_mask, alpha=1.0).save(output_dir / "seg_mask_raw.png")
        logger.info("Saved seg masks to %s/", output_dir)

        for component, crop in crops.items():
            if crop is None:
                logger.info("  %s: not detected", component)
                continue
            annotated = crop.copy()
            draw = ImageDraw.Draw(annotated)
            r = results.get(component, {})
            status = r.get("status", "unknown")
            confidence = r.get("confidence", 0)
            color = (255, 0, 0) if status == "broken" else (0, 200, 0)
            draw.rectangle([0, 0, annotated.width, 20], fill=(0, 0, 0))
            draw.text((4, 2), f"{status} ({confidence:.2%})", fill=color)
            annotated.save(crops_dir / f"{component}.png")
            logger.info("  Saved %s crop", component)

    def predict(self, image: Image.Image, charger_id: str = "unknown", save_debug: bool = False, debug_dir: str = "debug_output") -> dict:
        start = time.time()

        if image.mode != "RGB":
            image = image.convert("RGB")

        seg_mask = self._run_segmentation(image)

        components: dict[str, dict] = {}
        crops: dict[str, Optional[Image.Image]] = {}
        flagged = False

        for class_id, component_name in COMPONENT_MAP.items():
            crop = self._extract_crop(image, seg_mask, class_id)
            crops[component_name] = crop

            if crop is None:
                components[component_name] = {"detected": False, "status": None, "confidence": None}
                continue

            label, confidence = self._classify_component(component_name, crop)
            if confidence < REVIEW_THRESHOLD:
                flagged = True

            components[component_name] = {"detected": True, "status": label, "confidence": confidence}

        processing_time_ms = round((time.time() - start) * 1000)

        detected_statuses = [v["status"] for v in components.values() if v["detected"]]
        if not detected_statuses:
            overall_status = "NO_COMPONENTS_DETECTED"
        elif "broken" in detected_statuses:
            overall_status = "FAULT_DETECTED"
        else:
            overall_status = "HEALTHY"

        result = {
            "charger_id": charger_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "processing_time_ms": processing_time_ms,
            "overall_status": overall_status,
            "flagged_for_review": flagged,
            "components": components,
            "model_versions": {
                "segformer": SEGFORMER_MODEL_ID,
                "classifiers": CLASSIFIER_MODEL_IDS,
            },
        }

        if save_debug:
            self._save_debug_output(image, seg_mask, crops, components, Path(debug_dir))

        return result


if __name__ == "__main__":
    import json
    import sys

    image_path = sys.argv[1] if len(sys.argv) > 1 else "semantic_segmentation_models/example_data/example.jpg"
    save_debug = "--debug" in sys.argv

    logger.info("Loading test image from %s", image_path)
    image = Image.open(image_path)

    pipeline = FLAREPipeline()
    logger.info("Running inference...")
    result = pipeline.predict(image, charger_id="TEST-001", save_debug=save_debug, debug_dir="debug_output")

    print("\n" + "=" * 50)
    print("FAULT REPORT")
    print("=" * 50)
    print(json.dumps(result, indent=2))
    print(f"\nTotal processing time: {result['processing_time_ms']}ms")

    if save_debug:
        print("\nDebug output saved to debug_output/")
        print("  seg_mask_overlay.png  — mask blended on original")
        print("  seg_mask_raw.png      — raw color map")
        print("  crops/                — component crops with status labels")