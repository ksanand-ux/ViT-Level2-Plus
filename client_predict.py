import argparse
import requests
import os
import json
from dotenv import load_dotenv

# Load environment variable (SERVER_URL)
load_dotenv()
SERVER_URL = os.getenv("SERVER_URL", "http://localhost:8000")

# Task auto-detection based on filename or CLI override
def choose_endpoint(filename, task=None):
    if task:
        return "/predict_yolo" if task == "yolo" else "/predict_vit"
    filename = filename.lower()
    if any(word in filename for word in ["detect", "yolo"]):
        return "/predict_yolo"
    return "/predict_vit"

# Send request to FastAPI server
def send_request(image_path, endpoint, save=False):
    url = SERVER_URL.rstrip("/") + endpoint
    with open(image_path, "rb") as f:
        files = {"file": f}
        response = requests.post(url, files=files)

    if response.status_code != 200:
        print(f"❌ Error: {response.status_code} - {response.text}")
        return None

    result = response.json()
    print("\n=== PREDICTION RESULT ===")
    print(json.dumps(result, indent=2))

    if save:
        output_path = f"outputs/prediction_response_{os.path.basename(image_path)}.json"
        os.makedirs("outputs", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[INFO] Saved response JSON to {output_path}")

    return result

# CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified client script for ViT and YOLO inference")
    parser.add_argument("image", type=str, help="Path to input image")
    parser.add_argument("--task", choices=["vit", "yolo"], help="Force task selection")
    parser.add_argument("--save", action="store_true", help="Save the response to a JSON file")

    args = parser.parse_args()

    endpoint = choose_endpoint(args.image, args.task)
    print(f"[INFO] Using endpoint: {endpoint}")

    send_request(args.image, endpoint, save=args.save)
