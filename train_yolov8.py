import argparse
import os
import shutil
from ultralytics import YOLO
from utils.s3_helper import upload_to_s3
import mlflow
from mlflow import MlflowClient
import json

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

    # Set MLflow tracking URI
    mlflow.set_tracking_uri("http://localhost:5000")
    # Optional: group all YOLOv8 runs under one experiment
    mlflow.set_experiment("YOLOv8")

    # Convert local best.pt file path to MLflow URI
    model_uri = f"file://{os.path.abspath(saved_path)}"

    # Path to metrics.json (YOLOv8 saves it under runs/<name>/metrics.json)
    metrics_path = os.path.join(args.output_dir, "yolov8", "metrics.json")
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"[ERROR] Cannot find metrics.json at {metrics_path}")

    # Load accuracy
    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    accuracy = metrics.get("metrics", {}).get("accuracy", None)
    if accuracy is None:
        raise ValueError("[ERROR] Accuracy not found in metrics.json")

    print(f"[INFO] Validation accuracy from metrics.json: {accuracy}")

    # Check threshold
    if accuracy < 0.80:
        print(f"[WARNING] Accuracy {accuracy} is below threshold 0.80 — skipping MLflow registration.")
        return  # Skip remaining steps

    # Register the model to MLflow
    model_name = "YOLOv8_Model"
    result = mlflow.register_model(model_uri=model_uri, name=model_name)

    # Promote model to Staging
    client = MlflowClient()
    client.transition_model_version_stage(
        name=model_name,
        version=result.version,
        stage="Staging"
    )
    print(f"[MLFLOW] Model registered and promoted to Staging: v{result.version}")


if __name__ == "__main__":
    main()
# Trigger YOLOv8
