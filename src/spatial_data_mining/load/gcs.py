from pathlib import Path
import logging
import os
import random
import time

from google.cloud import storage

logger = logging.getLogger(__name__)


def _get_max_attempts() -> int:
    raw = os.getenv("SDM_GCS_UPLOAD_MAX_ATTEMPTS", "").strip()
    if not raw:
        return 3
    try:
        val = int(raw)
    except ValueError:
        return 3
    return max(2, val)


def upload_to_gcs(
    local_path: Path,
    bucket: str,
    prefix: str | None = None,
    object_name: str | None = None,
    max_attempts: int | None = None,
) -> str:
    local_path = Path(local_path)
    client = storage.Client()
    bucket_obj = client.bucket(bucket)
    base_name = object_name or local_path.name
    obj_name = f"{prefix.strip('/')}/{base_name}" if prefix else base_name
    blob = bucket_obj.blob(obj_name)

    attempts = max_attempts if max_attempts is not None else _get_max_attempts()
    attempts = max(2, attempts)
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            blob.upload_from_filename(str(local_path))
            return f"gs://{bucket}/{obj_name}"
        except Exception as exc:
            last_exc = exc
            if attempt >= attempts:
                break
            sleep_seconds = min(2 ** (attempt - 1), 10) + random.uniform(0, 0.5)
            logger.warning(
                "GCS upload failed (attempt %s/%s) for %s: %s",
                attempt,
                attempts,
                local_path,
                exc,
            )
            time.sleep(sleep_seconds)
    if last_exc is None:
        raise RuntimeError("GCS upload failed without a captured exception.")
    raise last_exc
