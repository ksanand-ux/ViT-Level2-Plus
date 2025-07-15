from ultralytics import YOLO
import torch
from pathlib import Path
from PIL import Image

# Config
model_path = "yolov8n.pt"  # Or your trained model like "models/yolo_best.pt"
test_image = "yolo-data/val/images/image-22-_jpg.rf.76cfa72efc798063cac543803b86f94c.jpg"  # Place one sample test image here

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {device}")

# Load model
try:
    model = YOLO(model_path)
    print("[INFO] YOLOv8 model loaded successfully.")
except Exception as e:
    print(f"[ERROR] Failed to load YOLOv8 model: {e}")
    exit(1)

# Run inference on one image
try:
    results = model(test_image, device=device)
    print(f"[SMOKE TEST] Prediction Results:\n{results[0].boxes}")
except Exception as e:
    print(f"[ERROR] Inference failed: {e}")
    exit(1)

print("[✅] YOLOv8 smoke test completed successfully.")
