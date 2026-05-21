"""
FLARE ONNX Inference Pipeline
==============================
Drop-in replacement for inference_pipeline.py using ONNX Runtime
instead of PyTorch for inference.

Why ONNX Runtime over PyTorch for inference:
    PyTorch is optimized for training — flexible computation graph,
    autograd tracking, Python overhead. At inference time none of that
    is needed. ONNX Runtime strips it all out and runs a static,
    optimized graph with CPU kernel fusion.

    Benchmark results (Apple M1/Intel Mac, CPU only):
        SegFormer B3:  PyTorch 1524ms → ONNX 793ms  (1.92x)
        ViT (avg):     PyTorch 130ms  → ONNX 105ms  (1.25x)
        Total models:  PyTorch 2044ms → ONNX 1215ms (1.68x)

Trade-offs documented:
    - ViT classifiers exported with fixed 224x224 input — acceptable
      since all crops are resized to this dimension by the processor
    - TracerWarning on ViT input validation is benign — the check is
      hardcoded as a constant in the ONNX graph, which is correct
      behavior for fixed-size inference
    - ONNX graph is static — dynamic control flow not supported,
      but FLARE pipeline has no dynamic branching at inference time

Usage:
    Same as inference_pipeline.py — swap the import in app/main.py:
    from inference_pipeline_onnx import FLAREPipeline
"""

import time
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import onnxruntime as ort
from transformers import (
    SegformerImageProcessor,
    AutoImageProcessor,
)
from huggingface_hub import hf_hub_download

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

SEGFORMER_MODEL_ID = "JaesonGu/flare-segformer-mit-b3"
CLASSIFIER_MODEL_IDS = {
    "screen": "JaesonGu/flare-screen-vit",
    "body":   "JaesonGu/flare-body-vit",
    "cable":  "JaesonGu/flare-cable-vit",
    "plug":   "JaesonGu/flare-plug-vit",
}

# ONNX model paths
ONNX_DIR = Path("models/onnx")
SEGFORMER_ONNX = ONNX_DIR / "segformer_b3.onnx"
CLASSIFIER_ONNX = {
    "screen": ONNX_DIR / "screen_vit.onnx",
    "body":   ONNX_DIR / "body_vit.onnx",
    "cable":  ONNX_DIR / "cable_vit.onnx",
    "plug":   ONNX_DIR / "plug_vit.onnx",
}

# id2label from training notebook
# Used since ONNX models don't carry the HuggingFace config
CLASSIFIER_ID2LABEL = {0: "healthy", 1: "broken"}

REVIEW_THRESHOLD = 0.70
CROP_PADDING = 8
CABLE_BROKEN_MIN_CONFIDENCE = 0.92

# --- ONNX Model Download ---
ONNX_HF_REPO = "JaesonGu/flare-onnx-models"

def ensure_onnx_models():
    """
    Download ONNX models from HuggingFace if not present locally.
    Called once at startup — skipped if files already exist.
    This enables Docker deployment without baking model files into the image.
    """
    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    
    files = {
        "segformer_b3.onnx": SEGFORMER_ONNX,
        "screen_vit.onnx":   CLASSIFIER_ONNX["screen"],
        "body_vit.onnx":     CLASSIFIER_ONNX["body"],
        "cable_vit.onnx":    CLASSIFIER_ONNX["cable"],
        "plug_vit.onnx":     CLASSIFIER_ONNX["plug"],
    }
    
    for filename, local_path in files.items():
        if not local_path.exists():
            logger.info("Downloading %s from HuggingFace...", filename)
            hf_hub_download(
                repo_id=ONNX_HF_REPO,
                filename=filename,
                local_dir=str(ONNX_DIR),
            )
            logger.info("  Downloaded %s", filename)
        else:
            logger.info("  %s already exists, skipping download", filename)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class FLAREPipeline:
    """
    ONNX Runtime version of the FLARE two-stage inference pipeline.
    Identical interface to the PyTorch version — swap the import to switch.
    """

    def __init__(self):
        # Download ONNX models from HuggingFace if not present locally
        ensure_onnx_models()

        # Validate ONNX files exist before trying to load
        for path in [SEGFORMER_ONNX] + list(CLASSIFIER_ONNX.values()):
            if not path.exists():
                raise FileNotFoundError(
                    f"ONNX model not found: {path}\n"
                    f"Run scripts/export_onnx.py first."
                )

        # ONNX Runtime session options
        # CPUExecutionProvider is default — no GPU required
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        logger.info("Loading SegFormer B3 ONNX from %s", SEGFORMER_ONNX)
        self.seg_session = ort.InferenceSession(
            str(SEGFORMER_ONNX),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )

        # Load processors from HuggingFace — these are lightweight config
        # objects only, no model weights
        self.seg_processor = SegformerImageProcessor.from_pretrained(SEGFORMER_MODEL_ID)

        logger.info("Loading ViT classifier ONNX sessions...")
        self.clf_sessions: dict[str, ort.InferenceSession] = {}
        self.clf_processors: dict[str, AutoImageProcessor] = {}

        for component, onnx_path in CLASSIFIER_ONNX.items():
            logger.info("  Loading %s from %s", component, onnx_path)
            self.clf_sessions[component] = ort.InferenceSession(
                str(onnx_path),
                sess_options=sess_options,
                providers=["CPUExecutionProvider"],
            )
            self.clf_processors[component] = AutoImageProcessor.from_pretrained(
                CLASSIFIER_MODEL_IDS[component]
            )

        logger.info("All ONNX models loaded successfully.")

    # -----------------------------------------------------------------------
    # Stage 1: Segmentation
    # -----------------------------------------------------------------------

    def _run_segmentation(self, image: Image.Image) -> np.ndarray:
        """
        Run SegFormer B3 via ONNX Runtime.
        Upsamples logits to original image size before argmax for clean
        bounding box extraction.
        """
        inputs = self.seg_processor(images=image, return_tensors="np")
        pixel_values = inputs["pixel_values"].astype(np.float32)

        # ONNX Runtime inference
        ort_inputs = {"pixel_values": pixel_values}
        logits = self.seg_session.run(None, ort_inputs)[0]  # (1, num_classes, H/4, W/4)

        # Upsample to original size using PyTorch (no model weights needed)
        logits_tensor = torch.from_numpy(logits)
        upsampled = F.interpolate(
            logits_tensor,
            size=(image.height, image.width),
            mode="bilinear",
            align_corners=False,
        )
        return upsampled.argmax(dim=1)[0].numpy()

    # -----------------------------------------------------------------------
    # Bridge: mask → bounding box crop (identical to PyTorch version)
    # -----------------------------------------------------------------------

    def _extract_crop(
        self,
        image: Image.Image,
        seg_mask: np.ndarray,
        class_id: int,
    ) -> Optional[Image.Image]:
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

    # -----------------------------------------------------------------------
    # Stage 2: Classification
    # -----------------------------------------------------------------------

    def _classify_component(
        self,
        component: str,
        crop: Image.Image,
    ) -> tuple[str, float]:
        """
        Run ViT classifier via ONNX Runtime.
        Uses hardcoded id2label since ONNX models don't carry HuggingFace config.
        """
        processor = self.clf_processors[component]
        session = self.clf_sessions[component]

        inputs = processor(images=crop, return_tensors="pt")
        pixel_values = inputs["pixel_values"].numpy().astype(np.float32)

        ort_inputs = {"pixel_values": pixel_values}
        logits = session.run(None, ort_inputs)[0]  # (1, num_classes)

        # Softmax manually since we're outside PyTorch
        exp_logits = np.exp(logits - logits.max())
        probs = exp_logits / exp_logits.sum()

        predicted_idx = int(probs.argmax())
        confidence = float(probs[0][predicted_idx])
        label = CLASSIFIER_ID2LABEL[predicted_idx]

        # Cable-specific override
        if component == "cable" and label == "broken" and confidence < CABLE_BROKEN_MIN_CONFIDENCE: 
            label = "healthy"

        return label, round(confidence, 4)

    # -----------------------------------------------------------------------
    # Public interface — identical to PyTorch version
    # -----------------------------------------------------------------------

    def predict(
        self,
        image: Image.Image,
        charger_id: str = "unknown",
        save_debug: bool = False,
        debug_dir: str = "debug_output",
    ) -> dict:
        start = time.time()

        if image.mode != "RGB":
            image = image.convert("RGB")

        seg_mask = self._run_segmentation(image)

        components: dict[str, dict] = {}
        flagged = False

        for class_id, component_name in COMPONENT_MAP.items():
            crop = self._extract_crop(image, seg_mask, class_id)

            if crop is None:
                components[component_name] = {
                    "detected": False,
                    "status": None,
                    "confidence": None,
                }
                continue

            label, confidence = self._classify_component(component_name, crop)
            if confidence < REVIEW_THRESHOLD:
                flagged = True

            components[component_name] = {
                "detected": True,
                "status": label,
                "confidence": confidence,
            }

        processing_time_ms = round((time.time() - start) * 1000)

        detected_statuses = [v["status"] for v in components.values() if v["detected"]]
        if not detected_statuses:
            overall_status = "NO_COMPONENTS_DETECTED"
        elif "broken" in detected_statuses:
            overall_status = "FAULT_DETECTED"
        else:
            overall_status = "HEALTHY"

        return {
            "charger_id": charger_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "processing_time_ms": processing_time_ms,
            "overall_status": overall_status,
            "flagged_for_review": flagged,
            "components": components,
            "model_versions": {
                "segformer": f"{SEGFORMER_MODEL_ID} (ONNX)",
                "classifiers": {k: f"{v} (ONNX)" for k, v in CLASSIFIER_MODEL_IDS.items()},
            },
        }


# ---------------------------------------------------------------------------
# Quick local test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    import sys

    image_path = sys.argv[1] if len(sys.argv) > 1 else "classifier_models/example_data/brokencharger_example.png"

    logger.info("Loading test image from %s", image_path)
    image = Image.open(image_path)

    pipeline = FLAREPipeline()
    logger.info("Running ONNX inference...")
    result = pipeline.predict(image, charger_id="TEST-ONNX")

    print("\n" + "=" * 50)
    print("FAULT REPORT (ONNX)")
    print("=" * 50)
    print(json.dumps(result, indent=2))
    print(f"\nTotal processing time: {result['processing_time_ms']}ms")