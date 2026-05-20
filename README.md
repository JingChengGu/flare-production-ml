## Project Origin

FLARE began as a UCSD capstone project developed by a team including Jason Gu and collaborators. The original research implementation included SegFormer-based component segmentation and ViT-based component classification models.

This repository extends that work into a production-style ML inference system by adding:

- Automated segmentation-to-classification handoff
- FastAPI inference service
- Dockerized deployment
- Structured fault report generation
- MLflow tracking
- Cloud deployment
- Monitoring and latency benchmarking
- ONNX inference optimization experiments


### Pipeline Limitation
Cable classifier was trained on manually cropped images. In production, crops are auto-generated from SegFormer bounding boxes, creating a training/inference distribution mismatch. Observed effect: cables at non-standard angles are misclassified as broken despite being healthy. Fix requires retraining classifiers on segmentation-derived crops — deferred due to GPU access requirements.