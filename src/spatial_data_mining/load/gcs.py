from pathlib import Path

from google.cloud import storage


def upload_to_gcs(local_path: Path, bucket: str, prefix: str | None = None) -> str:
    local_path = Path(local_path)
    client = storage.Client()
    bucket_obj = client.bucket(bucket)
    obj_name = f"{prefix.strip('/')}/{local_path.name}" if prefix else local_path.name
    blob = bucket_obj.blob(obj_name)
    blob.upload_from_filename(str(local_path))
    return f"gs://{bucket}/{obj_name}"
