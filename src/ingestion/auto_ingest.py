"""
Auto Ingestion Module

Scans data/raw/{domain} on startup, chunks and embeds new/modified documents
into the Chroma stores, and removes chunks of files that changed or vanished.

Manifest keys are project-relative POSIX paths, so the manifest stays valid if
the project folder is moved. Stale chunks are deleted by `filename` before a
changed file is re-embedded, which prevents duplicate chunks.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterator, Tuple

from langchain_community.document_loaders import PyPDFLoader, TextLoader

from config.settings import settings, BASE_DIR
from src.ingestion.chunker import DocumentChunker
from src.ingestion.contextualizer import PageContextualizer, add_chunk_headers
from src.retrieval import vector_store_manager
from src.retrieval.vector_store import embedding_slug

logger = logging.getLogger(__name__)

MANIFEST_FILE = (
    Path(settings.data_dir) / "vectorstores" / f".ingested_manifest.{embedding_slug()}.json"
)


def _rel_key(path: Path) -> str:
    """Project-relative POSIX path used as manifest key (absolute if outside project)."""
    path = Path(path)
    try:
        return path.resolve().relative_to(BASE_DIR).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _load_manifest() -> Dict[str, Any]:
    """Load manifest; migrates legacy absolute-path keys to relative ones."""
    if not MANIFEST_FILE.exists():
        return {}
    try:
        with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception as e:
        logger.warning(f"Failed to read ingestion manifest, recreating: {e}")
        return {}
    return {_rel_key(Path(k)): v for k, v in raw.items()}


def _save_manifest(manifest: Dict[str, Any]) -> None:
    try:
        MANIFEST_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Failed to save ingestion manifest: {e}")


def _load_single_document(file_path: Path):
    """Load a single document based on its extension."""
    ext = file_path.suffix.lower()
    if ext == ".pdf":
        return PyPDFLoader(str(file_path)).load()
    if ext in (".txt", ".md"):
        return TextLoader(str(file_path), encoding="utf-8").load()
    if ext == ".docx":
        try:
            from langchain_community.document_loaders import Docx2txtLoader
            return Docx2txtLoader(str(file_path)).load()
        except ImportError:
            logger.error(f"docx2txt is not installed - cannot ingest {file_path.name} (pip install docx2txt)")
            return []
    logger.debug(f"Unsupported file format for auto-ingest: {file_path}")
    return []


def _iter_files() -> Iterator[Tuple[str, Path]]:
    """Yield (domain, file) for every supported file under data/raw."""
    supported = {e.lower() for e in settings.supported_file_types}
    folders = {
        "sport": Path(settings.sport_data_path),
        "tech": Path(settings.tech_data_path),
        "general": Path(settings.general_data_path),
    }
    for domain, folder in folders.items():
        folder.mkdir(parents=True, exist_ok=True)
        for file_path in sorted(folder.glob("**/*")):
            if file_path.is_file() and file_path.suffix.lower() in supported:
                yield domain, file_path

    # Loose files directly in data/raw are treated as "general"
    raw_root = Path(settings.data_dir) / "raw"
    if raw_root.exists():
        for file_path in sorted(raw_root.glob("*")):
            if file_path.is_file() and file_path.suffix.lower() in supported:
                yield "general", file_path


def auto_ingest_raw_documents() -> Dict[str, int]:
    """
    Synchronise data/raw with the vector stores.

    - new / modified files are (re-)embedded (old chunks removed first)
    - files that disappeared have their chunks removed
    """
    logger.info("Checking data/raw for documents to ingest...")

    manifest = _load_manifest()
    chunker = DocumentChunker(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
    stats = {"sport": 0, "tech": 0, "general": 0, "removed": 0, "total_chunks": 0}
    max_bytes = settings.max_file_size_mb * 1024 * 1024
    contextualizer = None  # created on first use (LLM client)

    present = set()
    for domain, file_path in _iter_files():
        key = _rel_key(file_path)
        present.add(key)
        stat = file_path.stat()

        entry = manifest.get(key)
        if entry and entry.get("mtime") == stat.st_mtime and entry.get("size") == stat.st_size:
            continue  # unchanged

        if stat.st_size > max_bytes:
            logger.warning(f"Skipping {file_path.name}: larger than {settings.max_file_size_mb} MB")
            continue

        logger.info(f"Ingesting new/modified document: {file_path.name} into '{domain}' store")
        try:
            docs = _load_single_document(file_path)
            if not docs:
                logger.warning(f"No content extracted from {file_path.name}")
                continue

            for d in docs:
                d.metadata["domain"] = domain
                d.metadata["filename"] = file_path.name
                d.metadata["source"] = key

            if settings.contextual_chunks:
                contextualizer = contextualizer or PageContextualizer()
                contextualizer.contextualize(docs, key, stat.st_size, stat.st_mtime)

            chunks = chunker.chunk(docs)
            if not chunks:
                continue
            if settings.contextual_chunks:
                add_chunk_headers(chunks)

            # Remove stale chunks of a previous version before adding the new ones
            if entry:
                vector_store_manager.delete_by_filename(entry.get("domain", domain), file_path.name)

            if vector_store_manager.add_documents(domain, chunks):
                manifest[key] = {
                    "domain": domain,
                    "filename": file_path.name,
                    "mtime": stat.st_mtime,
                    "size": stat.st_size,
                    "chunks_count": len(chunks),
                }
                stats[domain] += 1
                stats["total_chunks"] += len(chunks)
                logger.info(f"Embedded {file_path.name} ({len(chunks)} chunks) into '{domain}'")
        except Exception as e:
            logger.error(f"Error ingesting {file_path.name}: {e}", exc_info=True)

    # Files deleted from disk -> remove their chunks and manifest entries
    for key in [k for k in manifest if k not in present]:
        entry = manifest.pop(key)
        removed = vector_store_manager.delete_by_filename(entry.get("domain", "general"), entry.get("filename", ""))
        stats["removed"] += 1
        logger.info(f"Removed {removed} chunks of deleted file: {entry.get('filename')}")

    _save_manifest(manifest)

    ingested = stats["sport"] + stats["tech"] + stats["general"]
    if ingested or stats["removed"]:
        logger.info(
            f"Auto-ingestion completed: {ingested} documents ({stats['total_chunks']} chunks) added, "
            f"{stats['removed']} removed."
        )
    else:
        logger.info("Vector stores are up-to-date.")

    return stats
