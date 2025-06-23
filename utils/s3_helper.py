import boto3

def upload_to_s3(local_file, bucket, s3_key):
    s3 = boto3.client("s3")
    try:
        s3.upload_file(str(local_file), bucket, s3_key)
        print(f"[S3 UPLOAD] Uploaded {local_file} to s3://{bucket}/{s3_key}")
    except Exception as e:
        print(f"[S3 ERROR] Failed to upload {local_file}: {e}")
