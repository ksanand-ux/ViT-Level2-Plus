import boto3
import os

def upload_to_s3(local_file, bucket, s3_key):
    # Fetch credentials from environment
    aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")

    if not aws_access_key or not aws_secret_key:
        raise RuntimeError("❌ AWS credentials not found in environment variables")

    # Create S3 client with explicit credentials
    s3 = boto3.client(
        "s3",
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key
    )

    try:
        s3.upload_file(str(local_file), bucket, s3_key)
        print(f"[✅ S3 UPLOAD] Uploaded {local_file} to s3://{bucket}/{s3_key}")
    except Exception as e:
        print(f"[❌ S3 ERROR] Failed to upload {local_file}: {e}")
