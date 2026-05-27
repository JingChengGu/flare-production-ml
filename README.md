# FLARE — EV Charger Fault Detection (Production ML System)

## Live Demo

**Interactive Demo:** [jingchenggu-flare-production-ml-demoapp-pybn7l.streamlit.app](https://jingchenggu-flare-production-ml-demoapp-pybn7l.streamlit.app)

**API Base URL:** `https://flare-api-610805014879.us-central1.run.app`

> Note: First request may take 60–90 seconds (cold start — ONNX models download from HuggingFace). Subsequent requests return in ~5 seconds.

```bash
# Health check
curl https://flare-api-610805014879.us-central1.run.app/health

# Run fault detection
curl -X POST https://flare-api-610805014879.us-central1.run.app/analyze \
  -F "image=@your_charger_image.jpg" \
  -F "charger_id=CH-001"
```

---

## Project Origin

FLARE began as a UCSD capstone project developed by a team including Jason Gu and collaborators. The original research implementation included a SegFormer B3 model for component segmentation and ViT classifiers for fault detection — trained on labeled EV charger images and hosted on HuggingFace.

This repository extends that research prototype into a production-grade ML inference system.

---

## What Was Built

| Component | Description |
|-----------|-------------|
| Two-stage pipeline bridge | Automated handoff from SegFormer segmentation → bounding box crop extraction → ViT classification. This connection did not exist in the original prototype. |
| ONNX inference optimization | Exported all 5 models to ONNX Runtime, reducing p95 latency by 28% and cold start by 41% on CPU — no GPU required |
| FastAPI inference service | REST API with `/analyze`, `/health`, and `/metrics` endpoints |
| Auto model download | ONNX models hosted on HuggingFace, downloaded automatically at startup — keeps Docker image lightweight |
| Structured fault report | JSON output per request with per-component status, confidence scores, and review flags |
| Latency monitoring | Rolling p95/avg latency and error rate tracked via `/metrics` |
| Dockerized deployment | Containerized for consistent deployment across environments |
| CI/CD pipeline | GitHub Actions — every push to main automatically builds, pushes to GCR, and redeploys to Cloud Run |
| Cloud deployment | GCP Cloud Run — scales to zero, free tier sufficient for portfolio traffic |
| Interactive demo | Streamlit app hosted on Streamlit Community Cloud — upload any charger image and see results in real time |

---

## Architecture

```
POST /analyze (image)
        ↓
SegFormer B3 (ONNX)
Segments full charger image into component regions
        ↓
Bounding box crop extraction
One crop per component: screen, body, cable, plug
        ↓
ViT Classifiers x4 (ONNX)
Each crop classified as healthy / broken
        ↓
Fault report JSON
{overall_status, components, confidences, latency}
        ↓
/metrics endpoint
Rolling p95 latency, error rate, request count
```

---

## Performance Benchmarks

### ONNX Optimization (CPU, no GPU)

| Metric | PyTorch | ONNX | Improvement |
|--------|---------|------|-------------|
| Cold start | 2770ms | 1622ms | 41% faster |
| Avg latency | 2376ms | 1598ms | 33% faster |
| p95 latency | 2770ms | 2005ms | 28% faster |
| Error rate | 0% | 0% | — |

SegFormer B3 alone improved 48% (1524ms → 793ms) — the largest single gain.

Benchmarked on CPU (Apple MacBook), 5 labeled test images, warm server.

### API Validation

Tested on 5 labeled examples (broken charger, broken cord, broken plug, broken screen, healthy screen):
- 0 errors across all requests
- Correct fault detection on broken-screen (97.7% confidence) and broken-cord (80.7% confidence)
- Known limitation: cable classifier shows systematic false positives (see Known Limitations)

---

## API Reference

### POST /analyze

Accepts an EV charger image, returns a structured fault report.

```bash
curl -X POST https://flare-api-610805014879.us-central1.run.app/analyze \
  -F "image=@charger.jpg" \
  -F "charger_id=CH-001"
```

Response:
```json
{
  "charger_id": "CH-001",
  "timestamp": "2026-05-21T10:18:10",
  "processing_time_ms": 1464,
  "overall_status": "FAULT_DETECTED",
  "flagged_for_review": false,
  "components": {
    "screen": {"detected": true,  "status": "healthy", "confidence": 0.98},
    "body":   {"detected": true,  "status": "healthy", "confidence": 0.95},
    "cable":  {"detected": true,  "status": "broken",  "confidence": 0.89},
    "plug":   {"detected": true,  "status": "broken",  "confidence": 0.76}
  },
  "model_versions": {
    "segformer": "JaesonGu/flare-segformer-mit-b3 (ONNX)",
    "classifiers": {
      "screen": "JaesonGu/flare-screen-vit (ONNX)",
      "body":   "JaesonGu/flare-body-vit (ONNX)",
      "cable":  "JaesonGu/flare-cable-vit (ONNX)",
      "plug":   "JaesonGu/flare-plug-vit (ONNX)"
    }
  }
}
```

Components with confidence below 0.70 are automatically flagged for human review.

### GET /health
Returns server status and model load confirmation. Used by GCP Cloud Run to verify instance readiness before routing traffic.

### GET /metrics
Returns rolling request statistics: request count, error rate, avg latency, p95 latency.

---

## Models

| Model | HuggingFace | Format | Size |
|-------|-------------|--------|------|
| SegFormer B3 | JaesonGu/flare-segformer-mit-b3 | ONNX | 190MB |
| Screen ViT | JaesonGu/flare-screen-vit | ONNX | 343MB |
| Body ViT | JaesonGu/flare-body-vit | ONNX | 343MB |
| Cable ViT | JaesonGu/flare-cable-vit | ONNX | 343MB |
| Plug ViT | JaesonGu/flare-plug-vit | ONNX | 343MB |

ONNX models: `JaesonGu/flare-onnx-models`

Downloaded automatically at startup via `ensure_onnx_models()`. To regenerate from PyTorch weights: `python scripts/export_onnx.py`

---

## Running Locally

```bash
# Clone and set up environment
git clone https://github.com/JingChengGu/flare-production-ml
cd flare-production-ml
conda env create -f environment.yml
conda activate flare-prod

# Start the API server
uvicorn app.main:app --port 8080
# ONNX models download automatically on first startup (~1-2 min)

# Test the endpoint
curl -X POST http://localhost:8080/analyze \
  -F "image=@classifier_models/example_data/brokenscreen_example.png" \
  -F "charger_id=TEST-001"

# Run the Streamlit demo locally
streamlit run demo/app.py
```

---

## CI/CD

Every push to `main` triggers the GitHub Actions pipeline:

1. Build Docker image
2. Push to Google Container Registry
3. Deploy to GCP Cloud Run

No manual deployment steps required.

---

## Known Limitations & Future Work

**Cable/plug classifier domain gap**
Classifiers were trained on manually cropped component images. In production, crops are auto-generated from SegFormer bounding boxes — a different distribution. Observed effect: cables at non-standard angles are misclassified as broken despite being healthy. Fix requires retraining classifiers on segmentation-derived crops. Deferred due to GPU access requirements.

**ONNX fixed input size**
ViT classifiers exported with fixed 224x224 input. Acceptable since all production crops are resized to this dimension by the image processor. Would require re-export if input size changes.

**Class imbalance**
Screen and body classifiers trained with standard cross-entropy on imbalanced datasets (~65-69% healthy). Weighted loss would improve recall on broken class. Deferred pending retraining infrastructure.

**Active learning pipeline**
Low-confidence predictions (flagged_for_review) are logged but not automatically routed to a labeling queue. Full active learning loop would require labeling infrastructure — identified as highest-value v2 improvement.

**Cold start latency**
GCP Cloud Run scales to zero when idle. First request after idle period triggers model download + load (~30-60 seconds). Subsequent requests return in ~1600ms avg. Set minimum instances to 1 to eliminate cold start at additional cost.
