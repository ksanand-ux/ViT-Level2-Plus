import argparse
import os
import shutil
from ultralytics import YOLO
from utils.s3_helper import upload_to_s3

def main():
    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to data.yaml")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--output_dir", type=str, default="runs", help="Output directory")
    parser.add_argument("--upload_to_s3", action="store_true", help="Upload model to S3")
    parser.add_argument("--s3_bucket", type=str, default="", help="S3 bucket name")

    args = parser.parse_args()

    # Load model
    print("[INFO] Loading YOLOv8 model...")
    model = YOLO("yolov8n.pt")  # You can replace with yolov8s.pt or yolov8m.pt etc.

    # Train
    print("[INFO] Starting training...")
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        project=args.output_dir,
        name="yolov8",
        exist_ok=True,
        verbose=True
    )

    # Save best model to a simpler path
    best_model_path = os.path.join(args.output_dir, "yolov8", "weights", "best.pt")
    saved_path = os.path.join(args.output_dir, "yolov8_best.pt")
    shutil.copy(best_model_path, saved_path)
    print(f"[MODEL SAVED] Model saved to {saved_path}")

    # Optional S3 upload
    if args.upload_to_s3:
        if args.s3_bucket == "":
            print("[S3 ERROR] --s3_bucket is required when using --upload_to_s3")
        else:
            version = "v1"
            upload_to_s3(saved_path, args.s3_bucket, f"yolo/{version}/{os.path.basename(saved_path)}")


if __name__ == "__main__":
    main()
