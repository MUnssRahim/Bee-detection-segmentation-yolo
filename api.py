from fastapi import FastAPI, UploadFile, Response
from ultralytics import YOLO
import cv2
import numpy as np
import os

app = FastAPI()

# ---------------------------------------------------------
# 1. MODEL PATHS (CASE-SENSITIVE — AZURE LINUX)
# ---------------------------------------------------------
MODEL_PESTS_PATH = "./Models/Beetle_Drone_Varroa.onnx"
MODEL_QUEEN_PATH = "./Models/ModelIterationDeteection2.onnx"

model_pests = None
model_queen = None

# ---------------------------------------------------------
# 2. LOAD MODELS SAFELY (LAZY LOAD — FIXES AZURE TIMEOUT)
# --------------------------s-------------------------------
@app.on_event("startup")
def load_models():
    global model_pests, model_queen

    print("🚀 FastAPI startup: loading models...")

    if os.path.exists(MODEL_PESTS_PATH):
        print(f"✅ Loading Pest Model: {MODEL_PESTS_PATH}")
        model_pests = YOLO(MODEL_PESTS_PATH, task="detect")
    else:
        print(f"❌ Pest model NOT found at {MODEL_PESTS_PATH}")

    if os.path.exists(MODEL_QUEEN_PATH):
        print(f"✅ Loading Queen Model: {MODEL_QUEEN_PATH}")
        model_queen = YOLO(MODEL_QUEEN_PATH, task="detect")
    else:
        print(f"❌ Queen model NOT found at {MODEL_QUEEN_PATH}")

    print("✅ Model loading complete")

# ---------------------------------------------------------
# 3. VISUAL CONFIG
# ---------------------------------------------------------
COLORS = {
    "varroa-mites":       (0, 0, 255),
    "small-hive-beetle":  (255, 0, 255),
    "capped-drone-cell": (139, 0, 0),
    "queen":              (0, 255, 255),
}

HIDDEN = {"bee"}
IGNORE = {"drone", "drone-bee"}

# ---------------------------------------------------------
# 4. IMAGE PROCESSING
# ---------------------------------------------------------
def process_image(img_bytes: bytes):
    if model_pests is None and model_queen is None:
        print("⚠️ Models not loaded yet")
        return None

    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return None

    # Resize to training resolution
    img = cv2.resize(img, (1280, 1280))

    # Sharpen image
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
    img = cv2.filter2D(img, -1, kernel)

    boxes, names = [], []

    # Pest detection
    if model_pests:
        results = model_pests(img, conf=0.25, iou=0.45, verbose=False)
        if results and results[0].boxes is not None:
            r = results[0]
            boxes.extend(r.boxes)
            names.extend([r.names] * len(r.boxes))

    # Queen detection
    if model_queen:
        results = model_queen(img, conf=0.30, iou=0.45, verbose=False)
        if results and results[0].boxes is not None:
            r = results[0]
            boxes.extend(r.boxes)
            names.extend([r.names] * len(r.boxes))

    counts = {k: 0 for k in COLORS}

    for box, name_map in zip(boxes, names):
        cls_id = int(box.cls[0])
        cname = name_map.get(cls_id)

        if cname in IGNORE or cname in HIDDEN:
            continue

        if cname in counts:
            counts[cname] += 1

        color = COLORS.get(cname, (255, 255, 255))
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)

    # Legend
    y = 50
    for name, count in counts.items():
        color = COLORS[name]
        label = f"{name.replace('-', ' ').title()}: {count}"

        cv2.rectangle(img, (30, y - 20), (55, y + 5), color, -1)
        cv2.rectangle(img, (30, y - 20), (55, y + 5), (255, 255, 255), 2)

        cv2.putText(img, label, (70, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
        cv2.putText(img, label, (70, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        y += 40

    success, encoded = cv2.imencode(".jpg", img)
    if not success:
        return None

    return encoded.tobytes()

# ---------------------------------------------------------
# 5. API ROUTES
# ---------------------------------------------------------
@app.get("/")
def health():
    return {"status": "Bee Detection API running (Azure safe)"}

@app.post("/detect")
async def detect(file: UploadFile):
    img_bytes = await file.read()
    result = process_image(img_bytes)

    if result is None:
        return Response(content="Image processing failed", status_code=500)

    return Response(content=result, media_type="image/jpeg")
