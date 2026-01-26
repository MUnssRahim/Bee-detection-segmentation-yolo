from fastapi import FastAPI, UploadFile, Response
from ultralytics import YOLO
import cv2
import numpy as np
import os

app = FastAPI()

# ---------------------------------------------------------
# 1. LOAD MODELS (Global Context)
# ---------------------------------------------------------
# using relative paths for Vercel structure
model_path_pests = "./models/Beetle_Drone_Varroa.pt"
model_path_queen = "./models/ModelIterationDeteection2.pt"

model_pests = None
model_queen = None

if os.path.exists(model_path_pests):
    print(f"Loading Pest Model from: {model_path_pests}")
    model_pests = YOLO(model_path_pests)
else:
    print(f"WARNING: Pest Model not found at {model_path_pests}")

if os.path.exists(model_path_queen):
    print(f"Loading Queen Model from: {model_path_queen}")
    model_queen = YOLO(model_path_queen)
else:
    print(f"WARNING: Queen Model not found at {model_path_queen}")


# ---------------------------------------------------------
# 2. CONFIGURATION
# ---------------------------------------------------------
COLORS = {
    "varroa-mites":      (0, 0, 255),       # Red
    "small-hive-beetle": (255, 0, 255),     # Purple
    "capped-drone-cell": (139, 0, 0),       # Dark Blue
    "queen":             (0, 255, 255),     # Yellow
}

# Classes to completely hide (No Box, No Count)
HIDDEN = ["bee"]

# Classes to ignore logic (e.g. adult drones)
IGNORE = ["drone", "drone-bee"] 


# ---------------------------------------------------------
# 3. PROCESSING LOGIC
# ---------------------------------------------------------
def process_image(img_bytes):
    # --- Decode & Enhance ---
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Resize to standard size used in training
    img = cv2.resize(img, (1280, 1280))
    
    # Sharpen Filter
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
    img = cv2.filter2D(img, -1, kernel)

    # --- Run Detection ---
    boxes, names = [], []
    
    # Run Pest Model
    if model_pests:
        results = model_pests(img, conf=0.25, iou=0.45, verbose=False)
        if results:
            r = results[0]
            boxes.extend(r.boxes)
            names.extend([r.names] * len(r.boxes))
            
    # Run Queen Model
    if model_queen:
        results = model_queen(img, conf=0.30, iou=0.45, verbose=False)
        if results:
            r = results[0]
            boxes.extend(r.boxes)
            names.extend([r.names] * len(r.boxes))

    # --- Draw Results ---
    # Initialize counts to 0 for specific keys we care about
    counts = {k: 0 for k in COLORS}
    
    for box, name_dict in zip(boxes, names):
        cls_id = int(box.cls[0])
        cname = name_dict[cls_id]
        
        # Check Filters
        if cname in IGNORE: continue
        if cname in HIDDEN: continue

        # Increment Count
        if cname in counts:
            counts[cname] += 1
        elif cname in COLORS: # Safety check if model has extra classes
            counts[cname] = 1

        # Draw Box (Strictly box only, no text)
        color = COLORS.get(cname, (255, 255, 255))
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)

    # --- Draw Legend (Top-Left) ---
    y = 50
    for name, count in counts.items():
        if count >= 0: # Show all listed in COLORS, even if 0
            color = COLORS.get(name, (255, 255, 255))
            
            # 1. Draw Icon
            cv2.rectangle(img, (30, y-20), (55, y+5), color, -1)
            cv2.rectangle(img, (30, y-20), (55, y+5), (255,255,255), 2)
            
            # 2. Draw Text (No Bee Count)
            text = f"{name.replace('-', ' ').title()}: {count}"
            
            # Stroke (Black Outline)
            cv2.putText(img, text, (70, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 4)
            # Fill (Color)
            cv2.putText(img, text, (70, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            y += 40

    # --- Convert to JPG ---
    # This encodes the numpy array into JPEG bytes
    success, encoded_image = cv2.imencode('.jpg', img)
    if not success:
        return None
        
    return encoded_image.tobytes()

# ---------------------------------------------------------
# 4. API ENDPOINTS
# ---------------------------------------------------------
@app.get("/")
def read_root():
    return {"status": "Bee Detection API Active"}

@app.post("/detect")
async def detect(file: UploadFile):
    image_bytes = await file.read()
    processed_bytes = process_image(image_bytes)
    
    if processed_bytes is None:
        return Response(content="Error processing image", status_code=500)
        
    return Response(content=processed_bytes, media_type="image/jpeg")