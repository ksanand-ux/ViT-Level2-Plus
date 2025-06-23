# Vision Transformer + YOLOv8 MLOps Deployment (Level 2++)

This repository simulates a real-world MLOps pipeline using both image classification (Vision Transformer via HuggingFace) and object detection (YOLOv8 via Ultralytics).

It includes training, CI/CD, versioning, inference, observability, and monitoring in a production-style workflow.

---

## Features

- Modular training scripts: `train_vit.py`, `train_yolov8.py`
- CLI arguments for training configs (epochs, directories, S3 upload)
- FastAPI server with `/predict_vit` and `/predict_yolo` endpoints
- Unified client script for inference requests
- MLflow tracking with optional S3 model storage
- Docker + Docker Compose setup
- GitHub Actions CI/CD workflows for both models
- Prometheus and Grafana integration for monitoring

---

## Folder Structure

.
├── app/ # FastAPI inference server
├── client/ # CLI for prediction
├── data/, vit-data/, yolo-data/ # Datasets for classification and detection
├── models/ # Saved models (ignored by Git)
├── outputs/ # Logs and confusion matrices (ignored)
├── .github/workflows/ # CI/CD workflow YAML files
├── train_vit.py # Training script for ViT
├── train_yolov8.py # Training script for YOLOv8
├── predict_vit.py # Standalone ViT inference
├── predict_yolov8.py # Standalone YOLOv8 inference
├── docker-compose.yml
├── Dockerfile
└── README.md

---

## Quick Start

### Train a model

python train_vit.py --train_dir vit-data/train --val_dir vit-data/val --epochs 1
python train_yolov8.py --data yolo-data/data.yaml --epochs 1


### Start inference server

uvicorn inference_server:app --host 0.0.0.0 --port 8000


### Predict via CLI

python client_predict.py --image dog.jpg


### Docker Compose

docker-compose up --build


### API Documentation

http://localhost:8000/docs


---

## Tech Stack

- Vision Transformer (HuggingFace Transformers)
- YOLOv8 (Ultralytics)
- FastAPI
- MLflow
- AWS S3
- Docker and Docker Compose
- GitHub Actions
- Prometheus and Grafana

---

## Notes

This project is part of a Level 2++ MLOps portfolio simulation, created for demonstration, teaching, and learning purposes. It closely mirrors production-grade pipelines and covers the full machine learning lifecycle, including training, deployment, monitoring, and automation.

Last updated: 2025-06-23
# Trigger upload
# Trigger pipeline run
