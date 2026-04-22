import os
import cv2
import numpy as np
import base64
import logging
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from ultralytics import YOLO

# --- 1. Configure Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# --- 2. Initialize FastAPI ---
app = FastAPI(title="BuzzGuard AI Analyzer")

# --- 3. Model Paths & Verification ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_BEE_SEG_PATH = os.path.join(SCRIPT_DIR, "Bee_Seg.pt")
MODEL_DB_PATH = os.path.join(SCRIPT_DIR, "B_D_V.pt")
MODEL_LW_PATH = os.path.join(SCRIPT_DIR, "L_F_W.pt")
MODEL_CLASS_PATH = os.path.join(SCRIPT_DIR, "B_Q_PB.pt")

logging.info("Verifying model paths...")
for label, path in [("Segmentation", MODEL_BEE_SEG_PATH), 
                    ("Detection", MODEL_DB_PATH), 
                    ("Larvae", MODEL_LW_PATH), 
                    ("Classification", MODEL_CLASS_PATH)]:
    if os.path.exists(path):
        logging.info(f"✅ {label} model found in local directory: {path}")
    else:
        logging.error(f"❌ ERROR: {label} model missing at: {path}")

# --- 4. Constants & Thresholds ---
TARGET_WIDTH = 1280
TILE_SIZE = 640
OVERLAP = 0.20

SEG_CONF = 0.40
MIN_BEE_AREA = 800
DET_CONF = 0.35
MITE_CONF = 0.35
CLASS_CONF = 0.30
NMS_THRESH = 0.45

COLORS = {
    "pollenbee": (0, 255, 0), "queen": (255, 0, 255),
    "mite": (0, 0, 255), "drone_cell": (255, 0, 0), "beetle": (0, 0, 200),
    "larvae": (0, 165, 255), "web": (255, 255, 255)
}

# --- 5. Load Models to CPU ---
logging.info("Loading Models to CPU...")
try:
    model_seg = YOLO(MODEL_BEE_SEG_PATH, task="segment")
    model_db = YOLO(MODEL_DB_PATH, task="detect")
    model_lw = YOLO(MODEL_LW_PATH, task="detect")
    model_class = YOLO(MODEL_CLASS_PATH, task="detect")
    logging.info("Models successfully loaded to memory.")
except Exception as e:
    logging.error(f"Error loading models: {e}")

# --- 6. Helper Functions ---
def get_true_corners(hull):
    x, y, w, h = cv2.boundingRect(hull)
    bbox_corners = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
    hull_pts = hull.reshape(-1, 2)
    rect = np.zeros((4, 2), dtype="float32")
    for i in range(4):
        corner = bbox_corners[i]
        distances = np.linalg.norm(hull_pts - corner, axis=1)
        closest_index = np.argmin(distances)
        rect[i] = hull_pts[closest_index]
    return rect

def auto_crop_and_straighten(img):
    logging.info("Starting auto-crop and straightening process.")
    img_h, img_w = img.shape[:2]
    total_area = img_h * img_w
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    edges = cv2.Canny(blurred, 30, 120)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    fused_mask = cv2.dilate(edges, kernel, iterations=4)
    fused_mask = cv2.erode(fused_mask, kernel, iterations=2)
    contours, _ = cv2.findContours(fused_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return img
        
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    largest_contour = contours[0]
    
    if cv2.contourArea(largest_contour) < total_area * 0.15:
        return img
        
    hull = cv2.convexHull(largest_contour)
    rect_ordered = get_true_corners(hull)
    (tl, tr, br, bl) = rect_ordered
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect_ordered, dst)
    warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
    
    if maxHeight > maxWidth:
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
        
    margin = int(min(maxWidth, maxHeight) * 0.015)
    final_h, final_w = warped.shape[:2]
    return warped[margin:final_h - margin, margin:final_w - margin]

def fast_preprocess(img):
    logging.info("Starting fast preprocessing (CLAHE & Blurring).")
    h, w = img.shape[:2]
    new_h = int(h * (TARGET_WIDTH / w))
    img = cv2.resize(img, (TARGET_WIDTH, new_h), interpolation=cv2.INTER_LINEAR)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return img

def draw_commercial_box(img, x, y, w, h, label, color):
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(img, (x, y - text_h - 6), (x + text_w + 4, y), color, -1)
    cv2.putText(img, label, (x + 2, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

def draw_commercial_circle(img, cx, cy, radius, label, color):
    cv2.circle(img, (cx, cy), radius, color, 2)
    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(img, (cx - radius, cy - radius - text_h - 6), (cx - radius + text_w + 4, cy - radius), color, -1)
    cv2.putText(img, label, (cx - radius + 2, cy - radius - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

# --- 7. API Endpoint ---
@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    logging.info(f"Received API request. Filename: {file.filename}")
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img_raw = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img_raw is None:
        logging.error("Failed to decode uploaded image. Returning 400.")
        return JSONResponse(content={"error": "Invalid image file uploaded."}, status_code=400)

    logging.info(f"Image decoded successfully. Original Dimensions: {img_raw.shape}")

    img_straightened = auto_crop_and_straighten(img_raw)
    img_ready = fast_preprocess(img_straightened)
    img_h, img_w = img_ready.shape[:2]
    overlay = img_ready.copy()

    stats = {"bees": 0, "pollenbees": 0, "queens": 0, "mites": 0, "drone_cells": 0, "beetles": 0, "larvae": 0, "web": 0}

    master_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    
    all_boxes = []
    all_scores = []
    all_class_ids = []
    all_source_models = []

    # Macro processing
    logging.info("Running DB model on full image...")
    res_db_macro = model_db(img_ready, conf=DET_CONF, verbose=False)[0]
    for box in res_db_macro.boxes:
        cls_id = int(box.cls[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0].numpy())
        w, h = x2 - x1, y2 - y1
        all_boxes.append([x1, y1, w, h])
        all_scores.append(float(box.conf[0]))
        all_class_ids.append(cls_id)
        all_source_models.append("db")

    logging.info("Running LW model on full image...")
    res_lw_macro = model_lw(img_ready, conf=DET_CONF, verbose=False)[0]
    for box in res_lw_macro.boxes:
        cls_id = int(box.cls[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0].numpy())
        w, h = x2 - x1, y2 - y1
        all_boxes.append([x1, y1, w, h])
        all_scores.append(float(box.conf[0]))
        all_class_ids.append(cls_id)
        all_source_models.append("lw")

    # Tiled Processing
    step_size = int(TILE_SIZE * (1.0 - OVERLAP))
    logging.info(f"Starting Tiled Processing. Tile Size: {TILE_SIZE}, Step: {step_size}")

    tiles_processed = 0
    for y in range(0, img_h, step_size):
        for x in range(0, img_w, step_size):
            tiles_processed += 1
            y1, y2 = y, min(y + TILE_SIZE, img_h)
            x1, x2 = x, min(x + TILE_SIZE, img_w)
            tile = img_ready[y1:y2, x1:x2]

            res_seg = model_seg(tile, conf=SEG_CONF, verbose=False)[0]
            local_mask = np.zeros(tile.shape[:2], dtype=np.uint8)

            if res_seg.masks is not None:
                for poly in res_seg.masks.xy:
                    if len(poly) >= 3:
                        poly_pts = np.array(poly, dtype=np.int32)
                        if cv2.contourArea(poly_pts) > (MIN_BEE_AREA / 4):
                            cv2.fillPoly(local_mask, [poly_pts], 1)
                            master_mask[y1:y2, x1:x2] = cv2.bitwise_or(master_mask[y1:y2, x1:x2], local_mask)

            blackout_tile = cv2.bitwise_and(tile, tile, mask=local_mask)

            res_class = model_class(blackout_tile, conf=CLASS_CONF, verbose=False)[0]
            for box in res_class.boxes:
                cls_id = int(box.cls[0])
                bx1, by1, bx2, by2 = map(int, box.xyxy[0].numpy())
                w, h = bx2 - bx1, by2 - by1
                all_boxes.append([x1 + bx1, y1 + by1, w, h])
                all_scores.append(float(box.conf[0]))
                all_class_ids.append(cls_id)
                all_source_models.append("class")

            res_mite = model_db(blackout_tile, conf=MITE_CONF, verbose=False)[0]
            for box in res_mite.boxes:
                cls_id = int(box.cls[0])
                if cls_id == 2:
                    mx1, my1, mx2, my2 = map(int, box.xyxy[0].numpy())
                    w, h = mx2 - mx1, my2 - my1
                    all_boxes.append([x1 + mx1, y1 + my1, w, h])
                    all_scores.append(float(box.conf[0]))
                    all_class_ids.append(cls_id)
                    all_source_models.append("db")

    logging.info(f"Tiled Processing Complete. Extracted {tiles_processed} tiles.")

    # Mask counting
    logging.info("Calculating generic bee count via segmentation master mask...")
    master_mask = cv2.morphologyEx(master_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(master_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        if cv2.contourArea(c) > MIN_BEE_AREA:
            stats["bees"] += 1

    # NMS & Drawing
    logging.info(f"Applying Non-Maximum Suppression on {len(all_boxes)} bounding boxes...")
    if len(all_boxes) > 0:
        indices = cv2.dnn.NMSBoxes(all_boxes, all_scores, CLASS_CONF, NMS_THRESH)
        if len(indices) > 0: 
            for i in indices.flatten():
                bx, by, w, h = all_boxes[i]
                cls_id = all_class_ids[i]
                src = all_source_models[i]
                
                if src == "db":
                    if cls_id == 0:
                        draw_commercial_box(overlay, bx, by, w, h, "Drone", COLORS["drone_cell"])
                        stats["drone_cells"] += 1
                    elif cls_id == 1:
                        draw_commercial_circle(overlay, bx + w // 2, by + h // 2, 20, "Beetle", COLORS["beetle"])
                        stats["beetles"] += 1
                    elif cls_id == 2:
                        draw_commercial_circle(overlay, bx + w // 2, by + h // 2, 15, "Mite", COLORS["mite"])
                        stats["mites"] += 1
                elif src == "lw":
                    if cls_id == 0:
                        draw_commercial_box(overlay, bx, by, w, h, "Larvae", COLORS["larvae"])
                        stats["larvae"] += 1
                    elif cls_id == 2:
                        draw_commercial_box(overlay, bx, by, w, h, "Web", COLORS["web"])
                        stats["web"] += 1
                elif src == "class":
                    if cls_id == 2:
                        draw_commercial_box(overlay, bx, by, w, h, "Pollen", COLORS["pollenbee"])
                        stats["pollenbees"] += 1
                    elif cls_id == 3:
                        draw_commercial_box(overlay, bx, by, w, h, "Queen", COLORS["queen"])
                        stats["queens"] += 1
    else:
        logging.info("No boxes detected; NMS skipped.")

    logging.info("Fusing overlays and writing telemetry headers...")
    final_img = cv2.addWeighted(overlay, 0.85, img_ready, 0.15, 0)
    header_height = 100
    final_img = cv2.copyMakeBorder(final_img, header_height, 0, 0, 0, cv2.BORDER_CONSTANT, value=(20, 20, 20))

    cv2.putText(final_img, f"APPROXIMATE BEES DETECTED: {stats['bees']}", (30, 45), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 255), 2, cv2.LINE_AA)

    details = [
        (f"Mites: {stats['mites']}", COLORS["mite"]),
        (f"Queens: {stats['queens']}", COLORS["queen"]),
        (f"Pollen: {stats['pollenbees']}", COLORS["pollenbee"]),
        (f"Drones: {stats['drone_cells']}", COLORS["drone_cell"]),
        (f"Beetles: {stats['beetles']}", COLORS["beetle"]),
        (f"Larvae: {stats['larvae']}", COLORS["larvae"]),
        (f"Web: {stats['web']}", COLORS["web"])
    ]

    x_offset = 30
    y_pos = 85
    for text, color in details:
        cv2.circle(final_img, (x_offset + 10, y_pos - 5), 6, color, -1)
        cv2.putText(final_img, text, (x_offset + 25, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA)
        (text_width, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        x_offset += text_width + 45

    logging.info("Encoding output to Base64...")
    _, buffer = cv2.imencode(".jpg", final_img)
    encoded_image = base64.b64encode(buffer).decode("utf_8")

    logging.info(f"API Request processed successfully. Final Stats: {stats}")

    return JSONResponse(content={
        "status": "success",
        "statistics": stats,
        "processed_image_base64": encoded_image
    })

# --- 8. Uvicorn Runner ---
if __name__ == "__main__":
    # You can change the port here if 8000 is occupied.
    uvicorn.run(app, host="127.0.0.1", port=8000)