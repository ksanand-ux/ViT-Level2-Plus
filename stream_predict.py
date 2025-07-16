import os
import time
import json
import requests
import csv
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()
SERVER_URL = os.getenv("SERVER_URL", "http://localhost:8000")
WATCH_FOLDER = "stream_input"
PROCESSED_FOLDER = os.path.join(WATCH_FOLDER, "processed")
LOG_FILE = "outputs/stream_logs.csv"

os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs("outputs", exist_ok=True)

seen_files = set()

def choose_endpoint(filename):
    if any(word in filename.lower() for word in ["detect", "yolo"]):
        return "/predict_yolo"
    return "/predict_vit"

def send_request(image_path, endpoint):
    url = SERVER_URL.rstrip("/") + endpoint
    with open(image_path, "rb") as f:
        files = {"file": f}
        start_time = time.time()
        response = requests.post(url, files=files)
        latency = round(time.time() - start_time, 3)
        return response.json(), latency

def log_prediction(filename, model, predicted_class, confidence, latency):
    log_exists = os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not log_exists:
            writer.writerow(["timestamp", "filename", "model", "predicted_class", "confidence", "latency_sec"])
        writer.writerow([
            datetime.now().isoformat(timespec='seconds'),
            filename,
            model,
            predicted_class,
            round(confidence, 4),
            latency
        ])

print(f"[STREAM] Watching folder: {WATCH_FOLDER}...")

while True:
    for fname in os.listdir(WATCH_FOLDER):
        fpath = os.path.join(WATCH_FOLDER, fname)
        if not os.path.isfile(fpath) or not fname.lower().endswith((".jpg", ".png")):
            continue
        if fname in seen_files:
            continue

        try:
            endpoint = choose_endpoint(fname)
            response_json, latency = send_request(fpath, endpoint)
            model = response_json.get("model")
            predicted_class = response_json.get("predicted_class")
            confidence = response_json.get("confidence")

            log_prediction(fname, model, predicted_class, confidence, latency)
            print(f"[✅] {fname} → {predicted_class} ({confidence:.4f}) [{latency}s]")

            # Move to processed folder
            os.rename(fpath, os.path.join(PROCESSED_FOLDER, fname))
            seen_files.add(fname)
        except Exception as e:
            print(f"[ERROR] {fname} → {e}")

    time.sleep(5)
