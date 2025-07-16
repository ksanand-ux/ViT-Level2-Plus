import argparse
import os
import requests
import json
from dotenv import load_dotenv
import csv
from glob import glob

# Load env vars
load_dotenv()
SERVER_URL = os.getenv("SERVER_URL", "http://localhost:8000")

def choose_endpoint(filename, task=None):
    if task:
        return "/predict_yolo" if task == "yolo" else "/predict_vit"
    filename = filename.lower()
    if any(word in filename for word in ["detect", "yolo"]):
        return "/predict_yolo"
    return "/predict_vit"

def send_request(image_path, endpoint, save=False):
    url = SERVER_URL.rstrip("/") + endpoint
    with open(image_path, "rb") as f:
        files = {"file": f}
        response = requests.post(url, files=files)

    if response.status_code != 200:
        print(f"[❌] Failed: {image_path} → {response.status_code}")
        return None

    result = response.json()
    print(f"[✅] {image_path} → {result}")

    if save:
        os.makedirs("outputs", exist_ok=True)
        out_path = os.path.join("outputs", f"pred_{os.path.basename(image_path)}.json")
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)

    return result

def load_image_list(input_path):
    if os.path.isdir(input_path):
        return sorted(glob(os.path.join(input_path, "*.jpg")) + glob(os.path.join(input_path, "*.png")))
    elif input_path.endswith(".csv"):
        with open(input_path, newline='') as f:
            return [row[0] for row in csv.reader(f)]
    else:
        raise ValueError(f"Unsupported input path: {input_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch prediction for ViT/YOLO")
    parser.add_argument("--input_path", type=str, required=True, help="Folder or CSV of image paths")
    parser.add_argument("--task", choices=["vit", "yolo"], help="Force task selection")
    parser.add_argument("--save", action="store_true", help="Save predictions to outputs/")
    args = parser.parse_args()

    image_list = load_image_list(args.input_path)

    for image_path in image_list:
        if not os.path.exists(image_path):
            print(f"[⚠️] Skipping missing: {image_path}")
            continue
        endpoint = choose_endpoint(image_path, args.task)
        send_request(image_path, endpoint, save=args.save)
