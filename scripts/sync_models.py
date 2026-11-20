"""
Push / pull the ONNX models to / from S3-compatible object storage, so containers do
not need a local model directory (the "download on startup" pattern).

    python scripts/sync_models.py push               # after scripts/export_onnx.py
    python scripts/sync_models.py pull --if-missing  # container entrypoint

Objects: s3://$S3_BUCKET/$MODELS_S3_PREFIX/<model>/<file>  (MODELS_S3_PREFIX e.g. "models/onnx")
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import settings  # noqa: E402

FILES = ("model_int8.onnx", "tokenizer.json", "config.json", "special_tokens_map.json", "tokenizer_config.json")


def _client():
    from src.storage.objects import S3Storage
    return S3Storage()


def _models():
    return [settings.embedding_model.split("/")[-1], settings.reranker_model.split("/")[-1]]


def push(prefix: str) -> None:
    s3 = _client()
    s3.ensure_bucket()
    for model in _models():
        for f in FILES:
            local = settings.onnx_model_dir / model / f
            if local.exists():
                key = f"{prefix}/{model}/{f}"
                print(f"upload {local} -> s3://{s3.bucket}/{key}")
                s3.client.upload_file(str(local), s3.bucket, key)


def pull(prefix: str, if_missing: bool) -> None:
    for model in _models():
        if if_missing and (settings.onnx_model_dir / model / "model_int8.onnx").exists():
            print(f"{model}: present, skip")
            continue
        s3 = _client()
        (settings.onnx_model_dir / model).mkdir(parents=True, exist_ok=True)
        for f in FILES:
            key = f"{prefix}/{model}/{f}"
            target = settings.onnx_model_dir / model / f
            try:
                s3.client.download_file(s3.bucket, key, str(target) + ".part")
                os.replace(str(target) + ".part", target)
                print(f"download s3://{s3.bucket}/{key}")
            except Exception as e:
                if f == "model_int8.onnx" or f == "tokenizer.json":
                    raise SystemExit(f"cannot download required {key}: {e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["push", "pull"])
    ap.add_argument("--prefix", default=os.environ.get("MODELS_S3_PREFIX", "models/onnx"))
    ap.add_argument("--if-missing", action="store_true")
    a = ap.parse_args()
    push(a.prefix) if a.action == "push" else pull(a.prefix, a.if_missing)
