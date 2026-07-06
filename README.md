
# 🐝 Bee Detection & Segmentation using YOLO

This repository contains a FastAPI inference app and training notebooks for bee detection and segmentation using YOLO models.

## Repository structure

- [api.py](api.py) — FastAPI app used for local inference.
- [notebooks/bee-detections.ipynb](notebooks/bee-detections.ipynb) — detection workflow notebook.
- [notebooks/bees-segmentation.ipynb](notebooks/bees-segmentation.ipynb) — segmentation workflow notebook.
- [models/](models/) — ONNX model files used by the inference service.
- [CV_Buzz_Deployment/](CV_Buzz_Deployment/) — deployment package for the existing web service configuration.

## Model files

- [models/Beetle_Drone_Varroa.onnx](models/Beetle_Drone_Varroa.onnx)
- [models/ModelIterationDeteection2.onnx](models/ModelIterationDeteection2.onnx)

## Local run

```bash
pip install -r requirements.txt
uvicorn api:app --reload
```

## Deployment note

The deployment service is configured around [CV_Buzz_Deployment/](CV_Buzz_Deployment/), and the runtime entry points were preserved. The repository layout was cleaned up for clarity without changing the existing deployment behavior.

## Notes

- The original top-level notebook files remain available for compatibility.
- A compatibility stub remains at [Readme.md](Readme.md).

