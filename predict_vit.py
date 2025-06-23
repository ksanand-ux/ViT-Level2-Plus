# predict_vit.py
import argparse
import json
import os
import torch
from PIL import Image
from torchvision import transforms
from transformers import ViTForImageClassification, ViTFeatureExtractor, ViTConfig

import boto3
from botocore.exceptions import NoCredentialsError

def download_from_s3(s3_bucket, s3_key, local_path):
    s3 = boto3.client('s3')
    try:
        s3.download_file(s3_bucket, s3_key, local_path)
        print(f"[S3] Downloaded: s3://{s3_bucket}/{s3_key} → {local_path}")
    except NoCredentialsError:
        print("❌ AWS credentials not found.")
        exit(1)

def load_image(image_path, feature_extractor):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
    ])
    return transform(image).unsqueeze(0)

def predict(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    model_path = args.model_path
    if args.use_s3:
        local_model_path = "./temp_vit_model.pt"
        download_from_s3(args.s3_bucket, args.model_path, local_model_path)
        model_path = local_model_path

    # Load model
    config = ViTConfig.from_pretrained("google/vit-base-patch16-224", num_labels=args.num_classes)
    model = ViTForImageClassification(config)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Load image
    extractor = ViTFeatureExtractor.from_pretrained(args.model_name)
    image_tensor = load_image(args.image_path, extractor).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        confidence, predicted_class = torch.max(probs, dim=1)

    label = str(predicted_class.item())

    if args.class_mapping and os.path.exists(args.class_mapping):
        with open(args.class_mapping, "r") as f:
            mapping = json.load(f)
        label = mapping.get(str(predicted_class.item()), label)

    print(f"\n🧠 Predicted Class: {label}")
    print(f"📊 Confidence: {confidence.item():.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True, help="Path to input image")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model .pt file")
    parser.add_argument("--model_name", type=str, default="google/vit-base-patch16-224")
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--class_mapping", type=str, default="class_mapping.json")
    parser.add_argument("--use_s3", action="store_true")
    parser.add_argument("--s3_bucket", type=str, default=None)
    args = parser.parse_args()

    predict(args)
