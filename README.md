# 🐝 Bee Detection & Segmentation with YOLO + ONNX

This repository contains a fast inference pipeline for bee-related object detection and segmentation using YOLO and ONNX Runtime.

## What this project does

- Detects bees, varroa mites, hive beetles, capped drone cells, and queen appearances.
- Uses quantized ONNX models for faster inference on CPU-friendly environments.
- Exposes a simple FastAPI endpoint for inference.
- Includes notebooks for training and experimentation.

## Repository structure

- `README.md` — Project overview and usage.
- `requirements.txt` — Python dependencies.
- `api.py` — FastAPI application serving detection inference.
- `startup.sh` — Launch helper for production-style deployment.
- `local_test.py` — Quick local API test script.
- `Models/` — ONNX model artifacts.
- `bee-detection.ipynb` — Detection training and experimentation notebook.
- `bee-segmentation.ipynb` — Segmentation training and experimentation notebook.

> The second branch of this repository is responsible for deployment on Azure.

## Models included

- `Models/pest_detection.onnx` — Pest detection model (varroa + beetle + drone cell variants).
- `Models/queen_detection.onnx` — Queen detection model.

## Setup

1. Create a Python environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the API locally

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

Or use `startup.sh`:

```bash
./startup.sh
```

## API usage

### Health check

```bash
curl http://127.0.0.1:8000/
```

Response:

```json
{"status":"Bee Detection API running (Azure safe)"}
```

### Detect endpoint

```bash
curl -X POST "http://127.0.0.1:8000/detect" -F "file=@/path/to/image.jpg" --output result.jpg
```

The endpoint returns a JPEG image with bounding boxes and counts drawn on the frame.

## Local test

Edit `local_test.py` to point at an available test image, then run:

```bash
python local_test.py
```

## Notes

- `api.py` loads both models at startup.
- Detection results are filtered and drawn with color-coded bounding boxes.
- The API is designed to work in containerized or cloud-hosted Linux environments.

## Recommended next steps

1. Replace the placeholder test image path in `local_test.py` with your own image.
2. Extend `api.py` to return structured JSON if you need non-visual outputs.
3. Add a notebook or script for batch image evaluation.

---

## License

Use this repository for research, monitoring, and beehive health analysis.
