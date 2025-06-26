import argparse
import os
import datetime
from ultralytics import YOLO
from PIL import Image

def run_inference(model_path, image_path, save_dir):
    print("[INFO] Loading YOLOv8 model...")
    model = YOLO(model_path)

    print(f"[INFO] Running inference on {image_path}...")
    results = model(image_path)  # returns list of results

    # ⏳ Generate timestamped path
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join(save_dir, "yolo_preds")
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"yolo_output_{timestamp}.jpg")

    # 💾 Save annotated image
    result_image = results[0].plot()  # draw boxes on image
    result_image = Image.fromarray(result_image)
    result_image.save(save_path)

    print(f"[🖼️ SAVED] YOLO prediction image saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="outputs/yolov8_best.pt", help="Path to trained YOLO model")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--save_dir", type=str, default="outputs", help="Directory to save results")
    args = parser.parse_args()

    run_inference(args.model, args.image, args.save_dir)
