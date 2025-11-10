"""
REST API for PCB Defect Detection

Usage:
    python api.py                                # Auto-detect best model
    python api.py --model runs/.../best.pt       # Specific model
    python api.py --port 8000                    # Custom port

Endpoints:
    POST /detect          - Detect defects in image
    GET  /health          - Health check
    GET  /model-info      - Model information
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
from ultralytics import YOLO
import uvicorn
import numpy as np
from PIL import Image
import io
import argparse
import glob
import os

# Response models
class Detection(BaseModel):
    class_name: str
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2]

class DetectionResponse(BaseModel):
    detections: List[Detection]
    count: int
    inference_time_ms: float
    image_size: List[int]  # [width, height]

class ModelInfo(BaseModel):
    model_path: str
    model_type: str
    num_classes: int
    class_names: List[str]

# Global model variable
model = None
model_path = None

def find_best_model():
    """Find the best trained model"""
    patterns = [
        'runs/train/production_yolov8s*/weights/best.pt',
        'runs/train/baseline_yolov8n*/weights/best.pt',
    ]
    
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            return max(matches, key=os.path.getmtime)
    
    raise FileNotFoundError(
        "No trained model found. Train first:\n"
        "  python -m cli train-baseline"
    )

def load_model(model_path: str):
    """Load YOLO model"""
    global model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    print(f"Model loaded successfully!")
    print(f"Classes: {list(model.names.values())}")

# Create FastAPI app
app = FastAPI(
    title="PCB Defect Detection API",
    description="Deep learning API for detecting defects in printed circuit boards",
    version="1.0.0"
)

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    global model_path
    if model_path is None:
        model_path = find_best_model()
    load_model(model_path)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "PCB Defect Detection API",
        "endpoints": {
            "POST /detect": "Detect defects in uploaded image",
            "GET /health": "Check API health",
            "GET /model-info": "Get model information"
        }
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_path": model_path
    }

@app.get("/model-info", response_model=ModelInfo)
async def model_info():
    """Get model information"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return ModelInfo(
        model_path=model_path,
        model_type=os.path.basename(model_path).split('.')[0],
        num_classes=len(model.names),
        class_names=list(model.names.values())
    )

@app.post("/detect", response_model=DetectionResponse)
async def detect_defects(
    file: UploadFile = File(...),
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45
):
    """
    Detect defects in uploaded PCB image
    
    Args:
        file: Image file (JPG, PNG)
        conf_threshold: Confidence threshold (0.0-1.0)
        iou_threshold: IoU threshold for NMS (0.0-1.0)
    
    Returns:
        DetectionResponse with detected defects
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Validate thresholds
    if not 0.0 <= conf_threshold <= 1.0:
        raise HTTPException(status_code=400, detail="conf_threshold must be between 0.0 and 1.0")
    if not 0.0 <= iou_threshold <= 1.0:
        raise HTTPException(status_code=400, detail="iou_threshold must be between 0.0 and 1.0")
    
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Run inference
        results = model.predict(
            image,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False
        )
        
        # Parse results
        detections = []
        result = results[0]  # Single image
        
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            
            for box, conf, cls_id in zip(boxes, confidences, class_ids):
                detections.append(Detection(
                    class_name=model.names[cls_id],
                    confidence=float(conf),
                    bbox=box.tolist()
                ))
        
        # Get inference time
        inference_time = result.speed['inference'] if hasattr(result, 'speed') else 0.0
        
        return DetectionResponse(
            detections=detections,
            count=len(detections),
            inference_time_ms=inference_time,
            image_size=[image.width, image.height]
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PCB Defect Detection API")
    parser.add_argument("--model", type=str, default=None, help="Path to model file")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    args = parser.parse_args()
    
    # Set model path
    if args.model:
        model_path = args.model
    
    print(f"\n{'='*60}")
    print("PCB DEFECT DETECTION API")
    print(f"{'='*60}")
    print(f"Starting server at http://{args.host}:{args.port}")
    print(f"\nAPI Documentation: http://{args.host}:{args.port}/docs")
    print(f"{'='*60}\n")
    
    uvicorn.run(app, host=args.host, port=args.port)