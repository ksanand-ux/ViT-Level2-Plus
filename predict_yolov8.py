import argparse
import os
from ultralytics import YOLO
from PIL import Image

def run_inference(model_path, image_path, save_dir):
    print("[INFO] Loading YOLOv8 model...")
    model = YOLO(model_path)

    print(f"[INFO] Running inference on {image_path}...")
    results = model.predict(source=image_path, save=True, project=save_dir, name="inference")

    print(f"[DONE] Inference complete. Results saved to {os.path.join(save_dir, 'inference')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="outputs/yolov8_best.pt", help="Path to trained YOLO model")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--save_dir", type=str, default="outputs", help="Directory to save results")
    args = parser.parse_args()

    run_inference(args.model, args.image, args.save_dir)
