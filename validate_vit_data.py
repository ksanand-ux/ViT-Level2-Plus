import boto3
import json
import os
from pathlib import Path
from PIL import Image
from io import BytesIO
from datetime import datetime
from typing import List
import great_expectations as ge

# === CONFIGURATION ===
BUCKET_NAME = "e-see-vit-model"
S3_SPLITS = ["data/classification/v1/train", "data/classification/v1/val"]
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
LOG_PATH = "logs/validation_log.json"
RAW_FOLDER = "raw"
STAGING_FOLDER = "staging"
VALIDATED_FOLDER = "validated"

# === HELPERS ===
s3 = boto3.client("s3")
context = ge.data_context.DataContext("/path/to/great_expectations")

def list_s3_objects(prefix):
    """List all S3 objects under the specified prefix (folder)."""
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET_NAME, Prefix=prefix):
        for obj in page.get("Contents", []):
            yield obj["Key"]

def is_valid_image_s3(bucket, key):
    """Check if the image in S3 is valid (can be opened by PIL)."""
    try:
        response = s3.get_object(Bucket=bucket, Key=key)
        content = response['Body'].read()
        Image.open(BytesIO(content)).verify()
        return True
    except Exception:
        return False

def get_class_folders(root: Path) -> List[Path]:
    """Return a list of class folders (subdirectories) in the root directory."""
    return [p for p in root.iterdir() if p.is_dir()]

def get_all_images(class_folder: Path) -> List[Path]:
    """Return a list of all files (images) inside a class folder."""
    return list(class_folder.glob("*"))

def validate_class_structure(root: Path):
    """Validate that the root directory contains class folders."""
    class_folders = get_class_folders(root)
    if not class_folders:
        raise ValueError(f"No class folders found in {root}")
    
    print(f"Class folders found: {[f.name for f in class_folders]}")
    return class_folders

def validate_images(class_folders: List[Path]):
    """Validate images in each class folder."""
    for folder in class_folders:
        image_files = get_all_images(folder)
        if not image_files:
            print(f"Warning: No images found in {folder.name}")
        else:
            print(f"{len(image_files)} image(s) found in {folder.name}")
        
        for image in image_files:
            if not image.is_file():
                continue  # Skip if it's not a file
            if image.suffix.lower() not in VALID_EXTENSIONS:
                print(f"Invalid file type in {folder.name}: {image.name}")

def validate_split(split_prefix):
    """Validate an S3 split (train or validation) for image files."""
    log_entries = []
    classes_seen = set()

    for key in list_s3_objects(split_prefix):
        if key.endswith("/"):
            continue  # skip folder placeholders

        relative_path = key.replace(split_prefix + "/", "")
        parts = relative_path.split("/")
        if len(parts) != 2:
            continue  # skip malformed paths

        class_name, filename = parts
        classes_seen.add(class_name)

        _, ext = os.path.splitext(filename.lower())
        valid_ext = ext in VALID_EXTENSIONS
        valid_img = is_valid_image_s3(BUCKET_NAME, key)

        entry = {
            "s3_key": key,
            "class": class_name,
            "filename": filename,
            "ext_ok": valid_ext,
            "image_ok": valid_img,
            "timestamp": datetime.utcnow().isoformat()
        }
        log_entries.append(entry)

    return log_entries, classes_seen

def create_s3_folder(bucket, folder_name):
    """Create an S3 folder (if not already present)."""
    s3.put_object(Bucket=bucket, Key=(folder_name + "/"))

def upload_to_s3(local_file, bucket, s3_key):
    """Upload the file to the S3 bucket."""
    s3.upload_file(local_file, bucket, s3_key)
    print(f"[S3 UPLOAD] Uploaded {local_file} to s3://{bucket}/{s3_key}")

def validate_and_upload(data_path, upload_folder):
    """Validate and upload images to different S3 stages."""
    class_folders = get_class_folders(data_path)
    validate_images(class_folders)

    # Upload to S3 (multi-stage: raw → staging → validated)
    for folder in class_folders:
        class_name = folder.name
        raw_folder = f"{RAW_FOLDER}/{class_name}/"
        staging_folder = f"{STAGING_FOLDER}/{class_name}/"
        validated_folder = f"{VALIDATED_FOLDER}/{class_name}/"

        # Create folders in S3
        create_s3_folder(BUCKET_NAME, raw_folder)
        create_s3_folder(BUCKET_NAME, staging_folder)
        create_s3_folder(BUCKET_NAME, validated_folder)

        for image in get_all_images(folder):
            s3_key = f"{raw_folder}{image.name}"

            # Upload raw image
            upload_to_s3(str(image), BUCKET_NAME, s3_key)

            # After successful validation, move to staging and then validated
            s3_key_staging = f"{staging_folder}{image.name}"
            s3_key_validated = f"{validated_folder}{image.name}"
            
            # Simulate validation check (this could be done using GE as well)
            if image.suffix.lower() in VALID_EXTENSIONS:
                # Move to staging
                upload_to_s3(str(image), BUCKET_NAME, s3_key_staging)
                # Move to validated
                upload_to_s3(str(image), BUCKET_NAME, s3_key_validated)

                # Optional: delete raw images after validation
                s3.delete_object(Bucket=BUCKET_NAME, Key=s3_key)

# === MAIN FUNCTION ===
def main():
    # You can change this to accept CLI argument or load from .env
    root_dir = Path("data/classification/v1/train")

    if not root_dir.exists():
        raise FileNotFoundError(f"Root folder not found: {root_dir}")
    
    print(f"Validating folder: {root_dir}")
    class_folders = validate_class_structure(root_dir)
    validate_images(class_folders)

    # Validate S3 splits (train/validation)
    os.makedirs("logs", exist_ok=True)
    all_logs = []
    all_classes = set()

    for split in S3_SPLITS:
        print(f"Validating split: {split}")
        logs, classes = validate_split(split)
        all_logs.extend(logs)
        all_classes.update(classes)

    # Save logs to a JSON file
    with open(LOG_PATH, "w") as f:
        json.dump(all_logs, f, indent=2)

    print(f"\nValidation complete. Log saved to {LOG_PATH}")
    print(f"Classes detected: {sorted(all_classes)}")
    print(f"Total images validated: {len(all_logs)}")

    # After validation, handle upload to S3 with multi-stage process
    validate_and_upload(root_dir, BUCKET_NAME)

# If script is run directly, execute main()
if __name__ == "__main__":
    main()
