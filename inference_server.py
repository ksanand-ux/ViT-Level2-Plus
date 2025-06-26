import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image, ImageDraw, ImageFont
import torch
from transformers import ViTFeatureExtractor, ViTForImageClassification, ViTConfig
from ultralytics import YOLO
from prometheus_fastapi_instrumentator import Instrumentator
import boto3
from botocore.exceptions import NoCredentialsError
import io
import os
import time
import cv2
import datetime

app = FastAPI()
# Add Prometheus instrumentation
Instrumentator().instrument(app).expose(app)
s3_client = boto3.client('s3')
BUCKET_NAME = "e-see-vit-model"
instrumentator = Instrumentator().instrument(app).expose(app)

# === Load ViT Model ===
vit_model_path = "outputs/vit_model.pt"
vit_config = ViTConfig.from_pretrained("google/vit-base-patch16-224", num_labels=3)
vit_model = ViTForImageClassification(vit_config)
vit_model.load_state_dict(torch.load(vit_model_path, map_location=torch.device("cpu")))
vit_model.eval()

vit_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
vit_class_mapping = {0: "airplane", 1: "cat", 2: "dog"}

# === Load YOLOv8 Model ===
yolo_model_path = "outputs/yolov8_best.pt"
yolo_model = YOLO(yolo_model_path)

# === Root Endpoint ===
@app.get("/")
def root():
    return {"message": "Unified Inference API: ViT (classification) + YOLOv8 (detection)"}

# === ViT Prediction Endpoint ===
@app.post("/predict_vit")
async def predict_vit(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        inputs = vit_extractor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = vit_model(**inputs)

        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
        confidence = torch.softmax(logits, dim=1)[0][predicted_class_idx].item()
        predicted_class = vit_class_mapping[predicted_class_idx]

        # ✅ Save image with label overlay
        output_image = image.copy()
        draw = ImageDraw.Draw(output_image)
        try:
            font = ImageFont.truetype("arial.ttf", size=20)
        except:
            font = ImageFont.load_default()

        draw.text((10, 10), f"{predicted_class} ({confidence:.2f})", fill="red", font=font)

        print("[DEBUG] Creating outputs/vit directory if not exists...")
        os.makedirs("outputs/vit", exist_ok=True)
        timestamp = int(time.time())
        output_path = f"outputs/vit/vit_output_{timestamp}.jpg"
        print(f"[DEBUG] Image will be saved to: {output_path}")
        output_image.save(output_path)
        print(f"[INFO] Saved ViT output image as {output_path}")

        return {
            "model": "ViT",
            "predicted_class": predicted_class,
            "confidence": round(confidence, 4)
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# === YOLOv8 Prediction Endpoint ===
@app.post("/predict_yolo")
async def predict_yolo(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Save input temporarily
        os.makedirs("temp_inputs", exist_ok=True)
        temp_input_path = f"temp_inputs/{file.filename}"
        image.save(temp_input_path)

        # Run detection
        results = yolo_model(temp_input_path)
        result = results[0]
        plotted = result.plot()  # RGB ndarray with boxes

        # Save output image
        os.makedirs("outputs/yolo", exist_ok=True)
        timestamp = int(time.time())
        output_path = f"outputs/yolo/yolo_output_{timestamp}.jpg"
        cv2.imwrite(output_path, plotted[:, :, ::-1])  # RGB to BGR
        
        # === OPTIONAL ENHANCEMENT: Upload to S3 ===
        import datetime

        ts_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        s3_key = f"inference-results/yolo/yolo_output_{ts_str}.jpg"

        try:
            s3_client.upload_file(output_path, BUCKET_NAME, s3_key)
            print(f"[S3 UPLOAD] Uploaded {output_path} to s3://{BUCKET_NAME}/{s3_key}")
        except Exception as e:
            print(f"[S3 ERROR] Failed to upload to S3: {e}")

        # Parse detection boxes
        detections = []
        for box in result.boxes.data.tolist():
            x1, y1, x2, y2, conf, class_id = box
            detections.append({
                "class_id": int(class_id),
                "confidence": round(conf, 4),
                "box": [round(x1), round(y1), round(x2), round(y2)]
            })

        return {
            "model": "YOLOv8",
            "detections": detections,
            "output_image": output_path
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# === Start Server ===
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
