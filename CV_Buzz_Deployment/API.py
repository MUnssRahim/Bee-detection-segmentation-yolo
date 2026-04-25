import os
import cv2
import numpy as np
import logging
import base64
import gc
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from ultralytics import YOLO
import torch

# --- 1. Configuration & Constants ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - BuzzGuard: %(message)s")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_SEG_PATH = os.path.join(SCRIPT_DIR, "Bee_Seg.pt")     
MODEL_MAIN_PATH = os.path.join(SCRIPT_DIR, "Combined_Final.pt")  
MODEL_DRONE_PATH = os.path.join(SCRIPT_DIR, "B_D_V.pt")     

TARGET_WIDTH = 1280
TILE_SIZE = 640
OVERLAP = 0.20  # Optimized for speed and lower memory footprint
CROP_MARGIN = 0.12 
TILE_BATCH_SIZE = 4  # Reduced batch size to prevent RAM spikes on Railway

# FORCED CPU: Railway does not have GPUs. Forcing CPU prevents PyTorch from allocating CUDA memory.
USE_HALF_PRECISION = False 
DEVICE = "cpu"

CONF_THRESHOLDS = {
    "seg": 0.25,
    "base_det": 0.20,
    "mite": 0.1,
    "pollen": 0.30,
    "queen": 0.35,
    "damage": 0.6  
}
MIN_BEE_AREA = 600     
NMS_THRESH = 0.45

COLORS = {
    "drone_cell": (255, 100, 50),   
    "mite": (0, 0, 255),            
    "damage": (0, 140, 255),        
    "pollen": (0, 240, 255),        
    "queen": (50, 255, 50),         
    "larvae": (255, 255, 0)         
}

# --- 2. Core Vision Operations ---
class VisionOps:
    @staticmethod
    def order_points(pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0], rect[2] = pts[np.argmin(s)], pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1], rect[3] = pts[np.argmin(diff)], pts[np.argmax(diff)]
        return rect

    @staticmethod
    def isolate_comb(img, margin_percent):
        h_orig, w_orig = img.shape[:2]
        
        # Memory/Speed Optimization: Downscale for morphology
        scale = 0.25
        small_img = cv2.resize(img, (int(w_orig * scale), int(h_orig * scale)))
        gray = cv2.cvtColor(small_img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (15, 15), 0) 
        edges = cv2.Canny(blurred, 30, 100)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours: return img 
        largest_contour = max(contours, key=cv2.contourArea)
        
        if cv2.contourArea(largest_contour) < (gray.shape[0] * gray.shape[1] * 0.15): return img 

        rect = cv2.minAreaRect(largest_contour)
        box = cv2.boxPoints(rect) if hasattr(cv2, 'boxPoints') else cv2.cv.BoxPoints(rect)
        
        box = box / scale 
        rect_pts = VisionOps.order_points(np.int32(box))
        (tl, tr, br, bl) = rect_pts

        maxWidth = max(int(np.linalg.norm(br - bl)), int(np.linalg.norm(tr - tl)))
        maxHeight = max(int(np.linalg.norm(tr - br)), int(np.linalg.norm(tl - bl)))

        dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
        warped = cv2.warpPerspective(img, cv2.getPerspectiveTransform(rect_pts, dst), (maxWidth, maxHeight))

        crop_y, crop_x = int(maxHeight * margin_percent), int(maxWidth * margin_percent)
        if maxHeight - (2 * crop_y) > 200 and maxWidth - (2 * crop_x) > 200:
            return warped[crop_y:maxHeight - crop_y, crop_x:maxWidth - crop_x]
        return warped

    @staticmethod
    def prepare_segmentation_stream(img):
        h, w = img.shape[:2]
        new_h = int(h * (TARGET_WIDTH / w))
        img_color = cv2.resize(img, (TARGET_WIDTH, new_h), interpolation=cv2.INTER_LINEAR)
        gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
        gray_eq = cv2.equalizeHist(gray)
        laplacian = cv2.Laplacian(gray_eq, cv2.CV_64F)
        sharpened_gray = cv2.addWeighted(gray_eq, 1.5, cv2.convertScaleAbs(laplacian), -0.5, 0)
        return img_color, cv2.cvtColor(sharpened_gray, cv2.COLOR_GRAY2BGR)


# --- 3. The Analyzer Engine ---
class BuzzGuardAnalyzer:
    def __init__(self):
        logging.info(f"Booting AI Models on {DEVICE}...")
        self.model_seg = YOLO(MODEL_SEG_PATH, task="segment")
        self.model_main = YOLO(MODEL_MAIN_PATH, task="detect")
        self.model_drone = YOLO(MODEL_DRONE_PATH, task="detect")
        
    def sahi_predict(self, model, image, conf_thresh, is_segmentation=False):
        img_h, img_w = image.shape[:2]
        step_size = int(TILE_SIZE * (1.0 - OVERLAP))
        tiles, tile_coords = [], []
        
        for y in range(0, img_h, step_size):
            for x in range(0, img_w, step_size):
                tiles.append(image[y:min(y + TILE_SIZE, img_h), x:min(x + TILE_SIZE, img_w)])
                tile_coords.append((x, y))
                
        all_boxes, all_polygons = [], []
        
        for i in range(0, len(tiles), TILE_BATCH_SIZE):
            batch_tiles = tiles[i : i + TILE_BATCH_SIZE]
            batch_coords = tile_coords[i : i + TILE_BATCH_SIZE]
            
            # MEMORY FIX: Force PyTorch to release gradients
            with torch.no_grad():
                batch_results = model(batch_tiles, conf=conf_thresh, verbose=False, device=DEVICE, imgsz=TILE_SIZE)
            
            for j, results in enumerate(batch_results):
                offset_x, offset_y = batch_coords[j]
                
                if is_segmentation and results.masks is not None:
                    for poly in results.masks.xy:
                        if len(poly) >= 3:
                            poly_pts = np.array(poly, dtype=np.int32)
                            poly_pts[:, 0] += offset_x
                            poly_pts[:, 1] += offset_y
                            if cv2.contourArea(poly_pts) > (MIN_BEE_AREA / 4):
                                all_polygons.append(poly_pts)
                                
                elif not is_segmentation and results.boxes is not None:
                    for box in results.boxes:
                        bx1, by1, bx2, by2 = map(int, box.xyxy[0].numpy())
                        all_boxes.append([offset_x + bx1, offset_y + by1, bx2 - bx1, by2 - by1, float(box.conf[0]), int(box.cls[0])])
                        
        return all_polygons if is_segmentation else all_boxes

    @staticmethod
    def _draw_production_bracket(img, x, y, w, h, label, color):
        t = 2  
        L = max(8, int(min(w, h) * 0.25))  
        
        cv2.line(img, (x, y), (x + L, y), color, t)
        cv2.line(img, (x, y), (x, y + L), color, t)
        cv2.line(img, (x + w, y), (x + w - L, y), color, t)
        cv2.line(img, (x + w, y), (x + w, y + L), color, t)
        cv2.line(img, (x, y + h), (x + L, y + h), color, t)
        cv2.line(img, (x, y + h), (x, y + h - L), color, t)
        cv2.line(img, (x + w, y + h), (x + w - L, y + h), color, t)
        cv2.line(img, (x + w, y + h), (x + w, y + h - L), color, t)

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.45
        tw, th = cv2.getTextSize(label, font, scale, 1)[0]
        cv2.rectangle(img, (x, y - th - 6), (x + tw + 4, y), color, -1)
        cv2.putText(img, label, (x + 2, y - 3), font, scale, (0, 0, 0), 1, cv2.LINE_AA)

    @torch.no_grad() # Crucial memory constraint for inference
    def process_frame_api(self, img_raw):
        if img_raw is None: 
            raise ValueError("Invalid image array")

        img_clean = VisionOps.isolate_comb(img_raw, CROP_MARGIN)
        img_color, img_seg_stream = VisionOps.prepare_segmentation_stream(img_clean)
        img_h, img_w = img_color.shape[:2]
        
        polygons = self.sahi_predict(self.model_seg, img_seg_stream, CONF_THRESHOLDS["seg"], is_segmentation=True)
        master_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        cv2.fillPoly(master_mask, polygons, 255)
        master_mask = cv2.dilate(master_mask, np.ones((20, 20), np.uint8), iterations=1)
        
        contours, _ = cv2.findContours(master_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        physical_bee_count = sum(1 for c in contours if cv2.contourArea(c) > MIN_BEE_AREA)
        final_bee_count = physical_bee_count if physical_bee_count < 10 else physical_bee_count + 15

        raw_detections = [] 
        center_x, center_y = img_w / 2.0, img_h / 2.0
        allowed_margin_x = img_w * 0.35 
        allowed_margin_y = img_h * 0.35 
        best_queen_candidate = None

        for box in self.sahi_predict(self.model_main, img_color, CONF_THRESHOLDS["base_det"]):
            x, y, w, h, score, cls_id = box
            box_cx, box_cy = int(x + w/2), int(y + h/2)

            if cls_id == 5 and (best_queen_candidate is None or score > best_queen_candidate[4]):
                best_queen_candidate = [x, y, w, h, score, cls_id]

            if cls_id == 5: 
                if score < CONF_THRESHOLDS["queen"]: continue
                if abs(box_cx - center_x) > allowed_margin_x or abs(box_cy - center_y) > allowed_margin_y: continue 
            if cls_id == 2 and score < CONF_THRESHOLDS["mite"]: continue
            if cls_id == 4 and score < CONF_THRESHOLDS["pollen"]: continue
            if cls_id == 0 and score < CONF_THRESHOLDS["damage"]: continue

            if cls_id in [2, 4, 5] and 0 <= box_cy < img_h and 0 <= box_cx < img_w:
                if master_mask[box_cy, box_cx] == 0: continue 

            raw_detections.append([x, y, w, h, score, cls_id])

        # Forced Queen Logic
        has_queen = any(d[5] == 5 for d in raw_detections)
        if final_bee_count > 30 and not has_queen and best_queen_candidate is not None:
            raw_detections.append(best_queen_candidate)

        for box in self.sahi_predict(self.model_drone, img_color, CONF_THRESHOLDS["base_det"]):
            raw_detections.append([box[0], box[1], box[2], box[3], box[4], 99])

        stats = {"bees": final_bee_count, "pollen": 0, "queens": 0, "mites": 0, "damage": 0, "larvae": 0, "drones": 0}
        final_render = img_color.copy()

        if raw_detections:
            boxes = [d[:4] for d in raw_detections]
            scores = [d[4] for d in raw_detections]
            indices = cv2.dnn.NMSBoxes(boxes, scores, CONF_THRESHOLDS["base_det"], NMS_THRESH)
            
            if len(indices) > 0:
                for i in indices.flatten():
                    x, y, w, h, score, cls_id = raw_detections[i]
                    if cls_id == 2:   stats["mites"] += 1;   self._draw_production_bracket(final_render, x, y, w, h, f"Mite {stats['mites']}", COLORS["mite"])
                    elif cls_id == 4: stats["pollen"] += 1;  self._draw_production_bracket(final_render, x, y, w, h, f"Pollen {stats['pollen']}", COLORS["pollen"])
                    elif cls_id == 5: stats["queens"] += 1;  self._draw_production_bracket(final_render, x, y, w, h, f"Queen {stats['queens']}", COLORS["queen"])
                    elif cls_id == 0: stats["damage"] += 1;  self._draw_production_bracket(final_render, x, y, w, h, f"Damage {stats['damage']}", COLORS["damage"])
                    elif cls_id == 1: stats["larvae"] += 1;  self._draw_production_bracket(final_render, x, y, w, h, f"Larvae {stats['larvae']}", COLORS["larvae"])
                    elif cls_id == 99: stats["drones"] += 1; self._draw_production_bracket(final_render, x, y, w, h, f"Drone {stats['drones']}", COLORS["drone_cell"])

        # Manual garbage collection of heavy numpy arrays
        del img_clean, img_seg_stream, master_mask, polygons
        gc.collect()

        return stats, final_render

# --- 4. FastAPI Setup ---
analyzer_instance = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global analyzer_instance
    analyzer_instance = BuzzGuardAnalyzer()
    yield
    analyzer_instance = None
    gc.collect()

app = FastAPI(title="BuzzGuard API", lifespan=lifespan)

@app.post("/analyze")
async def analyze_endpoint(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file.")

        stats, final_img = analyzer_instance.process_frame_api(img)

        _, buffer = cv2.imencode('.jpg', final_img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        # Clean up local references immediately to prevent memory leaks over time
        del img, final_img, buffer, nparr, contents
        gc.collect()

        return JSONResponse(content={
            "status": "success",
            "stats": stats,
            "image_base64": img_base64
        })

    except Exception as e:
        logging.error(f"Inference error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
