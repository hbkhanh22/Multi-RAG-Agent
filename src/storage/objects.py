"""
Document storage: where the original files live.

  local  data/raw/<domain>/**  (+ loose files in data/raw => general). Dev default;
         the folder watcher picks up edits made directly on disk.
  s3     any S3-compatible bucket (AWS S3, MinIO, Cloudflare R2, GCS interop), objects
         keyed "<domain>/<filename>". The source of truth in a deployment: API replicas
         and the ingest worker share it, nothing important lives on a container disk.

Both expose the same small interface used by the ingestion service and the API.
"""

import logging
import shutil
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional

from config.settings import BASE_DIR, settings

logger = logging.getLogger(__name__)

DOMAINS = ("sport", "tech", "general")


@dataclass
class StoredFile:
    domain: str
    key: str            # stable id: project-relative path (local) or object key (s3)
    filename: str
    size: int
    mtime: Optional[float] = None
    etag: Optional[str] = None

    def version(self) -> tuple:
        return (self.size, self.etag) if self.etag else (self.size, self.mtime)


def _supported(name: str) -> bool:
    return Path(name).suffix.lower() in {e.lower() for e in settings.supported_file_types}


def safe_filename(name: str) -> str:
    """Strip directories and characters that are unsafe in paths / object keys."""
    name = Path(name.replace("\\", "/")).name.strip().lstrip(".")
    cleaned = "".join(c for c in name if c.isalnum() or c in " ._-()[]").strip()
    if not cleaned or not _supported(cleaned):
        raise ValueError(f"Unsupported or empty filename: {name!r}")
    return cleaned[:200]


class LocalStorage:
    kind = "local"

    def _rel_key(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(BASE_DIR).as_posix()
        except ValueError:
            return path.resolve().as_posix()

    def list_files(self) -> List[StoredFile]:
        out: List[StoredFile] = []
        for domain, folder in settings.domain_data_paths.items():
            folder = Path(folder)
            folder.mkdir(parents=True, exist_ok=True)
            for p in sorted(folder.glob("**/*")):
                if p.is_file() and _supported(p.name):
                    st = p.stat()
                    out.append(StoredFile(domain, self._rel_key(p), p.name, st.st_size, st.st_mtime))
        raw_root = Path(settings.data_dir) / "raw"   # loose files => general
        if raw_root.exists():
            for p in sorted(raw_root.glob("*")):
                if p.is_file() and _supported(p.name):
                    st = p.stat()
                    out.append(StoredFile("general", self._rel_key(p), p.name, st.st_size, st.st_mtime))
        return out

    @contextmanager
    def open_local(self, f: StoredFile) -> Iterator[Path]:
        path = Path(f.key)
        yield path if path.is_absolute() else BASE_DIR / path

    def put(self, domain: str, filename: str, fileobj) -> str:
        target = Path(settings.domain_data_paths[domain]) / safe_filename(filename)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".part")
        with open(tmp, "wb") as out:
            shutil.copyfileobj(fileobj, out)
        tmp.replace(target)   # atomic: the watcher never sees a half-written file
        return self._rel_key(target)

    def delete(self, key: str) -> None:
        path = Path(key)
        path = path if path.is_absolute() else BASE_DIR / path
        root = (Path(settings.data_dir) / "raw").resolve()
        if root not in path.resolve().parents:
            raise ValueError("refusing to delete outside data/raw")
        path.unlink(missing_ok=True)

    def healthy(self) -> bool:
        return (Path(settings.data_dir) / "raw").exists()


class S3Storage:
    kind = "s3"

    def __init__(self):
        import boto3
        from botocore.config import Config

        self.bucket = settings.s3_bucket
        self.client = boto3.client(
            "s3",
            endpoint_url=settings.s3_endpoint_url or None,
            aws_access_key_id=settings.s3_access_key or None,
            aws_secret_access_key=settings.s3_secret_key or None,
            region_name=settings.s3_region,
            config=Config(retries={"max_attempts": 5, "mode": "standard"}, s3={"addressing_style": "path"}),
        )

    def ensure_bucket(self) -> None:
        from botocore.exceptions import ClientError

        try:
            self.client.head_bucket(Bucket=self.bucket)
        except ClientError:
            self.client.create_bucket(Bucket=self.bucket)
            logger.info(f"Created bucket {self.bucket}")

    def list_files(self) -> List[StoredFile]:
        out: List[StoredFile] = []
        for page in self.client.get_paginator("list_objects_v2").paginate(Bucket=self.bucket):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                domain, _, rest = key.partition("/")
                if domain not in DOMAINS or not rest:
                    domain, rest = "general", key
                name = rest.rsplit("/", 1)[-1]
                if _supported(name):
                    out.append(StoredFile(domain, key, name, obj["Size"], etag=obj["ETag"].strip('"')))
        return out

    @contextmanager
    def open_local(self, f: StoredFile) -> Iterator[Path]:
        """Download to a temp file (loaders need a path); removed afterwards."""
        tmp_dir = Path(tempfile.mkdtemp(prefix="rag-ingest-"))
        try:
            path = tmp_dir / f.filename
            self.client.download_file(self.bucket, f.key, str(path))
            yield path
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def put(self, domain: str, filename: str, fileobj) -> str:
        key = f"{domain}/{safe_filename(filename)}"
        self.client.upload_fileobj(fileobj, self.bucket, key)
        return key

    def delete(self, key: str) -> None:
        self.client.delete_object(Bucket=self.bucket, Key=key)

    def healthy(self) -> bool:
        try:
            self.client.head_bucket(Bucket=self.bucket)
            return True
        except Exception as e:
            logger.error(f"S3 storage unavailable: {e}")
            return False


_storage = None


def get_storage():
    global _storage
    if _storage is None:
        kind = settings.document_storage.lower()
        if kind == "s3":
            _storage = S3Storage()
            _storage.ensure_bucket()
        elif kind == "local":
            _storage = LocalStorage()
        else:
            raise ValueError(f"Unknown DOCUMENT_STORAGE '{settings.document_storage}' (local|s3)")
    return _storage
