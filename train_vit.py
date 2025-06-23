import argparse
import os
import json
import time
import torch
import mlflow
import mlflow.pytorch
import boto3
import numpy as np
from pathlib import Path
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification, ViTFeatureExtractor
from torch import nn, optim
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str, required=True)
    parser.add_argument("--val_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="google/vit-base-patch16-224")
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--log_to_mlflow", action="store_true")
    parser.add_argument("--upload_to_s3", action="store_true")
    parser.add_argument("--s3_bucket", type=str, default="")
    parser.add_argument("--dummy", action="store_true")
    return parser.parse_args()

def safe_mkdir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def upload_to_s3(file_path, bucket_name, s3_key):
    try:
        s3 = boto3.client('s3')
        s3.upload_file(str(file_path), bucket_name, s3_key)
        print(f"[S3] Uploaded: {file_path} → s3://{bucket_name}/{s3_key}")
    except Exception as e:
        print(f"[S3 ERROR] Failed to upload {file_path}: {e}")

def main():
    args = parse_args()
    if args.dummy: 
     print(" Dummy training triggered via CI/CD. Skipping real training.")
     return

    safe_mkdir(args.output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Transforms
    extractor = ViTFeatureExtractor.from_pretrained(args.model_name)
    common_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=extractor.image_mean, std=extractor.image_std)
    ])

    # Load datasets
    if not os.path.exists(args.train_dir) or not os.path.exists(args.val_dir):
        raise FileNotFoundError("Train/Val directories do not exist. Check paths.")

    train_dataset = ImageFolder(args.train_dir, transform=common_transform)
    val_dataset = ImageFolder(args.val_dir, transform=common_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = ViTForImageClassification.from_pretrained(
        args.model_name,
        num_labels=args.num_classes,
        ignore_mismatched_sizes=True
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    if args.log_to_mlflow:
        mlflow.set_experiment("ViT_Classifier_Advanced")
        mlflow.start_run()
        mlflow.log_params({
            "model_name": args.model_name,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate
        })

    # Training
    for epoch in range(args.epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images).logits
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total

        print(f"[EPOCH {epoch+1}] Loss: {avg_loss:.4f} | Accuracy: {accuracy:.4f}")

        if args.log_to_mlflow:
            mlflow.log_metric("train_loss", avg_loss, step=epoch)
            mlflow.log_metric("train_accuracy", accuracy, step=epoch)

    # Validation
    model.eval()
    preds, targets = [], []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images).logits
            _, predicted = torch.max(outputs, 1)
            preds.extend(predicted.cpu().numpy())
            targets.extend(labels.numpy())

    report = classification_report(targets, preds, output_dict=True)
    conf_mat = confusion_matrix(targets, preds)

    acc = report["accuracy"]
    print(f"[VAL] Accuracy: {acc:.4f}")
    print(f"[VAL] Confusion Matrix:\n{conf_mat}")
    from sklearn.metrics import ConfusionMatrixDisplay
    import matplotlib.pyplot as plt

    class_names = val_dataset.classes  # if you're using ImageFolder

    disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    print("[INFO] Confusion matrix saved as confusion_matrix.png")


    # Save metrics
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(report, f, indent=4)

    np.save(os.path.join(args.output_dir, "confusion_matrix.npy"), conf_mat)

    # Save model
    model_path = os.path.join(args.output_dir, "vit_model.pt")
    torch.save(model.state_dict(), model_path)

    if args.log_to_mlflow:
        mlflow.log_metric("val_accuracy", acc)
        mlflow.log_artifact(metrics_path)
        mlflow.pytorch.log_model(model, artifact_path="model")
        mlflow.end_run()

    # Optional S3 upload
    if args.upload_to_s3:
        if args.s3_bucket == "":
            print("[S3 ERROR] --s3_bucket is required when using --upload_to_s3")
        else:
            version = "v1"
            upload_to_s3(model_path, args.s3_bucket, f"vit/{version}/{model_path.name}")
            upload_to_s3(metrics_path, args.s3_bucket, f"vit/{version}/{metrics_path.name}")

    print(f"[DONE] Model + metrics saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
