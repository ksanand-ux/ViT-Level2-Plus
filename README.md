#  Vision Transformer + YOLOv8 MLOps Deployment (Level 2++)

This project simulates a **real-world MLOps pipeline** using both **image classification (ViT)** and **object detection (YOLOv8)** models. It includes:

- Modular training scripts
- S3 + MLflow logging
- FastAPI-based inference server
- Docker Compose deployment
- GitHub Actions CI/CD setup
- Prometheus + Grafana monitoring

---

#  Features

-  `train_vit.py` and `train_yolov8.py` with CLI support
-  MLflow tracking + optional S3 uploads
-  Inference via FastAPI `/predict_vit` and `/predict_yolo`
-  Docker Compose support for deployment
-  CI/CD pipeline using GitHub Actions
-  Monitoring with Prometheus + Grafana
-  Client CLI to send predictions to server

---

#  Folder Structure
   .
├── training/
│ ├── train_vit.py
│ └── train_yolov8.py
├── inference/
│ └── app.py
├── client/
│ └── client_predict.py
├── outputs/ # ignored except .gitkeep
├── metrics/ # confusion matrix, logs (ignored)
├── models/ # optional mount point
├── .github/workflows/ # CI/CD actions
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md

---

# Quick Start

# Train a model

python training/train_vit.py --epochs 1 --save_to_s3
python training/train_yolov8.py --epochs 1 --save_to_s3

# Start inference server
uvicorn inference.app:app --host 0.0.0.0 --port 8000

# Predict via CLI
python client/client_predict.py --image dog.jpg


# Docker Compose
docker-compose up --build

# API Docs
http://localhost:8000/docs

# Tech Stack
ViT (HuggingFace)

YOLOv8 (Ultralytics)

FastAPI

MLflow

S3 (AWS)

Docker + Compose

GitHub Actions

Prometheus + Grafana

# Notes
This is part of Level 2++ MLOps journey — built for portfolio, teaching, and learning purposes.
It simulates a production pipeline and includes real-world tools from training to monitoring.

