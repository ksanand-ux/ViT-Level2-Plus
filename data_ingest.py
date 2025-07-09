# data_ingest.py
import os
import argparse
import boto3
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()  # Optional: Load from .env if present

def upload_dir_to_s3(local_dir, bucket, s3_prefix):
    s3 = boto3.client("s3")
    local_path = Path(local_dir)

    for file_path in local_path.rglob("*"):
        if file_path.is_file():
            rel_path = file_path.relative_to(local_path)
            s3_key = f"{s3_prefix}/{rel_path.as_posix()}"
            try:
                s3.upload_file(str(file_path), bucket, s3_key)
                print(f"[UPLOAD] {file_path} → s3://{bucket}/{s3_key}")
            except Exception as e:
                print(f"[ERROR] {file_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--bucket", type=str, required=True)
    parser.add_argument("--version", type=str, required=True)
    parser.add_argument("--task", type=str, default="classification")

    args = parser.parse_args()

    for split in ["train", "val"]:
        local_split_dir = os.path.join(args.data_dir, split)
        s3_prefix = f"data/{args.task}/{args.version}/{split}"
        upload_dir_to_s3(local_split_dir, args.bucket, s3_prefix)
