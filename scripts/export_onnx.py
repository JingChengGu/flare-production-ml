"""
ONNX Export Script
==================
Exports all 5 FLARE models from PyTorch to ONNX format.

Why ONNX:
    PyTorch is optimized for training — flexible but slower at inference.
    ONNX Runtime is optimized purely for inference — kernel fusion,
    memory optimization, and platform-specific acceleration on CPU.
    Same results, faster execution. No GPU required.

Output:
    models/onnx/
        segformer_b3.onnx
        screen_vit.onnx
        body_vit.onnx
        cable_vit.onnx
        plug_vit.onnx

Usage:
    python scripts/export_onnx.py
"""

import time
import torch
import numpy as np
from pathlib import Path
from transformers import (
    SegformerForSemanticSegmentation,
    SegformerImageProcessor,
    AutoModelForImageClassification,
    AutoImageProcessor,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SEGFORMER_MODEL_ID = "JaesonGu/flare-segformer-mit-b3"
CLASSIFIER_MODEL_IDS = {
    "screen": "JaesonGu/flare-screen-vit",
    "body":   "JaesonGu/flare-body-vit",
    "cable":  "JaesonGu/flare-cable-vit",
    "plug":   "JaesonGu/flare-plug-vit",
}

OUTPUT_DIR = Path("models/onnx")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Input sizes confirmed from processor configs
SEGFORMER_INPUT_SIZE = (512, 512)   # height, width
VIT_INPUT_SIZE       = (224, 224)   # height, width

OPSET_VERSION = 14  # ONNX opset — 14 is stable for transformer models


# ---------------------------------------------------------------------------
# Benchmark helper
# ---------------------------------------------------------------------------

def benchmark_pytorch(model, dummy_input: dict, n_runs: int = 10) -> float:
    """Run n inference passes and return average latency in ms."""
    model.eval()
    with torch.no_grad():
        # Warmup
        _ = model(**dummy_input)
        # Timed runs
        start = time.time()
        for _ in range(n_runs):
            _ = model(**dummy_input)
    return round((time.time() - start) / n_runs * 1000, 1)


def benchmark_onnx(session, dummy_input: dict, n_runs: int = 10) -> float:
    """Run n ONNX Runtime inference passes and return average latency in ms."""
    import onnxruntime as ort
    inputs = {k: v.numpy() for k, v in dummy_input.items()}
    # Warmup
    _ = session.run(None, inputs)
    # Timed runs
    start = time.time()
    for _ in range(n_runs):
        _ = session.run(None, inputs)
    return round((time.time() - start) / n_runs * 1000, 1)


# ---------------------------------------------------------------------------
# Export SegFormer
# ---------------------------------------------------------------------------

def export_segformer():
    print("\n" + "="*50)
    print("Exporting SegFormer B3")
    print("="*50)

    print("Loading model...")
    model = SegformerForSemanticSegmentation.from_pretrained(SEGFORMER_MODEL_ID)
    model.eval()

    # Dummy input: batch=1, channels=3, height=512, width=512
    dummy_pixel_values = torch.randn(1, 3, *SEGFORMER_INPUT_SIZE)
    dummy_input = {"pixel_values": dummy_pixel_values}

    output_path = OUTPUT_DIR / "segformer_b3.onnx"

    print(f"Benchmarking PyTorch...")
    pytorch_ms = benchmark_pytorch(model, dummy_input)
    print(f"  PyTorch avg latency: {pytorch_ms}ms")

    print(f"Exporting to {output_path}...")
    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy_pixel_values,),
            str(output_path),
            input_names=["pixel_values"],
            output_names=["logits"],
            dynamic_axes={
                "pixel_values": {0: "batch_size"},
                "logits":       {0: "batch_size"},
            },
            opset_version=OPSET_VERSION,
        )

    print(f"Benchmarking ONNX Runtime...")
    import onnxruntime as ort
    session = ort.InferenceSession(str(output_path))
    onnx_ms = benchmark_onnx(session, dummy_input)
    print(f"  ONNX Runtime avg latency: {onnx_ms}ms")

    speedup = round(pytorch_ms / onnx_ms, 2)
    print(f"\nSpeedup: {speedup}x  ({pytorch_ms}ms → {onnx_ms}ms)")
    print(f"Saved: {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")

    return {"pytorch_ms": pytorch_ms, "onnx_ms": onnx_ms, "speedup": speedup}


# ---------------------------------------------------------------------------
# Export ViT classifiers
# ---------------------------------------------------------------------------

def export_vit(component: str, model_id: str):
    print(f"\n{'='*50}")
    print(f"Exporting ViT — {component}")
    print("="*50)

    print("Loading model...")
    model = AutoModelForImageClassification.from_pretrained(model_id)
    model.eval()

    # Dummy input: batch=1, channels=3, height=224, width=224
    dummy_pixel_values = torch.randn(1, 3, *VIT_INPUT_SIZE)
    dummy_input = {"pixel_values": dummy_pixel_values}

    output_path = OUTPUT_DIR / f"{component}_vit.onnx"

    print(f"Benchmarking PyTorch...")
    pytorch_ms = benchmark_pytorch(model, dummy_input)
    print(f"  PyTorch avg latency: {pytorch_ms}ms")

    print(f"Exporting to {output_path}...")
    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy_pixel_values,),
            str(output_path),
            input_names=["pixel_values"],
            output_names=["logits"],
            dynamic_axes={
                "pixel_values": {0: "batch_size"},
                "logits":       {0: "batch_size"},
            },
            opset_version=OPSET_VERSION,
        )

    print(f"Benchmarking ONNX Runtime...")
    import onnxruntime as ort
    session = ort.InferenceSession(str(output_path))
    onnx_ms = benchmark_onnx(session, dummy_input)
    print(f"  ONNX Runtime avg latency: {onnx_ms}ms")

    speedup = round(pytorch_ms / onnx_ms, 2)
    print(f"\nSpeedup: {speedup}x  ({pytorch_ms}ms → {onnx_ms}ms)")
    print(f"Saved: {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")

    return {"pytorch_ms": pytorch_ms, "onnx_ms": onnx_ms, "speedup": speedup}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import onnxruntime as ort
    print(f"ONNX Runtime version: {ort.__version__}")
    print(f"Output directory: {OUTPUT_DIR.resolve()}")

    results = {}

    # Export SegFormer
    results["segformer"] = export_segformer()

    # Export all ViT classifiers
    for component, model_id in CLASSIFIER_MODEL_IDS.items():
        results[component] = export_vit(component, model_id)

    # ---------------------------------------------------------------------------
    # Summary table
    # ---------------------------------------------------------------------------
    print("\n" + "="*50)
    print("ONNX EXPORT SUMMARY")
    print("="*50)
    print(f"{'Model':<20} {'PyTorch':>10} {'ONNX':>10} {'Speedup':>10}")
    print("-"*50)
    for model_name, r in results.items():
        print(f"{model_name:<20} {r['pytorch_ms']:>9}ms {r['onnx_ms']:>9}ms {r['speedup']:>9}x")

    total_pytorch = sum(r["pytorch_ms"] for r in results.values())
    total_onnx    = sum(r["onnx_ms"]    for r in results.values())
    total_speedup = round(total_pytorch / total_onnx, 2)

    print("-"*50)
    print(f"{'TOTAL PIPELINE':<20} {total_pytorch:>9}ms {total_onnx:>9}ms {total_speedup:>9}x")
    print("\nThese numbers are your resume talking point.")
    print(f"Save them: PyTorch p95 baseline was 2784ms end-to-end.")
