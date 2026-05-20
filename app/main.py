"""
FLARE FastAPI Inference Server
==============================
Exposes three endpoints:

    POST /analyze   — accepts an image, returns a structured fault report
    GET  /health    — confirms server is live and models are loaded
    GET  /metrics   — returns running request stats and latency averages

The FLAREPipeline is loaded once at server startup via FastAPI's lifespan
context manager. All requests share the same model instance — no reloading.
"""

import time
import logging
from contextlib import asynccontextmanager
from collections import deque

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from PIL import Image
import io

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))
from inference_pipeline import FLAREPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# In-memory metrics store
# Tracks last 1000 requests for rolling averages
# In production this would write to PostgreSQL — covered in Week 3
# ---------------------------------------------------------------------------

class MetricsStore:
    def __init__(self, maxlen: int = 1000):
        self.request_count = 0
        self.error_count = 0
        self.latencies_ms = deque(maxlen=maxlen)  # rolling window

    def record(self, latency_ms: int, error: bool = False):
        self.request_count += 1
        self.latencies_ms.append(latency_ms)
        if error:
            self.error_count += 1

    def summary(self) -> dict:
        if not self.latencies_ms:
            return {
                "request_count": self.request_count,
                "error_count": self.error_count,
                "avg_latency_ms": None,
                "p95_latency_ms": None,
            }

        sorted_latencies = sorted(self.latencies_ms)
        p95_idx = int(len(sorted_latencies) * 0.95)

        return {
            "request_count": self.request_count,
            "error_count": self.error_count,
            "error_rate": round(self.error_count / self.request_count, 4),
            "avg_latency_ms": round(sum(sorted_latencies) / len(sorted_latencies)),
            "p95_latency_ms": sorted_latencies[p95_idx],
        }


# ---------------------------------------------------------------------------
# App state — shared across all requests
# ---------------------------------------------------------------------------

# These are set during startup and read during requests
pipeline: FLAREPipeline = None
metrics: MetricsStore = MetricsStore()


# ---------------------------------------------------------------------------
# Lifespan — runs on server startup and shutdown
# This is where models load — once, before any request is served
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline

    logger.info("Server starting — loading FLARE pipeline...")
    pipeline = FLAREPipeline()
    logger.info("Pipeline ready. Server accepting requests.")

    yield  # server runs here, handling requests

    logger.info("Server shutting down.")
    pipeline = None


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="FLARE — EV Charger Fault Detection API",
    description="Two-stage CV pipeline: SegFormer B3 segmentation + ViT fault classification",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/analyze")
async def analyze(
    image: UploadFile = File(...),
    charger_id: str = Form(default = "unknown"),
):
    """
    Accepts an EV charger image and returns a structured fault report.

    - Runs SegFormer B3 to segment components (screen, body, cable, plug)
    - Runs ViT classifier on each component crop
    - Returns overall fault status + per-component confidence scores

    Args:
        image:      Image file (JPEG, PNG)
        charger_id: Optional charger identifier (e.g. "CH-1234")

    Returns:
        Fault report JSON with overall_status, components, and latency
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    # Validate file type
    if image.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported file type: {image.content_type}. Use JPEG or PNG."
        )

    start = time.time()
    error = False

    try:
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents))
        result = pipeline.predict(pil_image, charger_id=charger_id)
        return JSONResponse(content=result)

    except Exception as e:
        error = True
        logger.error("Inference error for charger %s: %s", charger_id, str(e))
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

    finally:
        latency_ms = round((time.time() - start) * 1000)
        metrics.record(latency_ms=latency_ms, error=error)
        logger.info(
            "charger_id=%s status=%s latency=%dms",
            charger_id,
            "error" if error else "ok",
            latency_ms,
        )


@app.get("/health")
async def health():
    """
    Health check endpoint.

    Used by deployment platforms (GCP Cloud Run, AWS) to verify the
    instance is live and models are loaded before routing traffic to it.

    Returns 200 if ready, 503 if pipeline failed to load.
    """
    if pipeline is None:
        return JSONResponse(
            status_code=503,
            content={"status": "unavailable", "models_loaded": False}
        )

    return JSONResponse(content={
        "status": "ok",
        "models_loaded": True,
        "model_versions": {
            "segformer": "JaesonGu/flare-segformer-mit-b3",
            "classifiers": {
                "screen": "JaesonGu/flare-screen-vit",
                "body":   "JaesonGu/flare-body-vit",
                "cable":  "JaesonGu/flare-cable-vit",
                "plug":   "JaesonGu/flare-plug-vit",
            }
        }
    })


@app.get("/metrics")
async def get_metrics():
    """
    Returns rolling request statistics for the current server instance.

    Includes request count, error rate, avg latency, and p95 latency.
    p95 latency is the primary SLA metric — 95% of requests complete within
    this time. This is what gets reported in load testing results.

    Note: resets when server restarts. Week 3 will persist this to PostgreSQL.
    """
    return JSONResponse(content=metrics.summary())
