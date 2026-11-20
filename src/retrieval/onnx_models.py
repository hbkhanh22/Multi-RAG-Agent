"""
ONNX Runtime versions of the embedding model and the cross-encoder reranker.

The int8 models are produced by `python scripts/export_onnx.py` (dynamic quantization
of the fp32 HuggingFace checkpoints). Only `onnxruntime` + `tokenizers`/`transformers`
are needed at runtime - no torch - which is what makes the API image small and lets
several workers share a 16 GB machine.

Both classes mirror what the sentence-transformers models do:
  bge-m3              CLS pooling + L2 normalisation
  bge-reranker-v2-m3  one logit per (query, passage) pair -> sigmoid (0..1)
"""

import logging
from pathlib import Path
from typing import List, Sequence

import numpy as np
from langchain_core.embeddings import Embeddings

from config.settings import settings

logger = logging.getLogger(__name__)

MODEL_FILE = "model_int8.onnx"


def onnx_dir(model_name: str) -> Path:
    return settings.onnx_model_dir / model_name.split("/")[-1]


def onnx_available(model_name: str) -> bool:
    d = onnx_dir(model_name)
    return (d / MODEL_FILE).exists() and (d / "tokenizer.json").exists()


def _session(path: Path):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if settings.onnx_threads:
        opts.intra_op_num_threads = settings.onnx_threads
    return ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])


def _tokenizer(path: Path, max_length: int):
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(path / "tokenizer.json"))
    tok.enable_truncation(max_length=max_length)
    tok.enable_padding(pad_id=tok.token_to_id("<pad>") or 1, pad_token="<pad>")
    return tok


def _feeds(session, encodings) -> dict:
    ids = np.array([e.ids for e in encodings], dtype=np.int64)
    mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
    names = {i.name for i in session.get_inputs()}
    feeds = {"input_ids": ids, "attention_mask": mask}
    if "token_type_ids" in names:
        feeds["token_type_ids"] = np.zeros_like(ids)
    return {k: v for k, v in feeds.items() if k in names}


class OnnxEmbeddings(Embeddings):
    """LangChain Embeddings backed by an int8 ONNX bge-m3 (CLS pooling, normalised)."""

    def __init__(self, model_name: str = None, max_length: int = 8192, batch_size: int = None):
        model_name = model_name or settings.embedding_model
        path = onnx_dir(model_name)
        logger.info(f"Loading ONNX embedding model from {path}")
        self.session = _session(path / MODEL_FILE)
        self.tokenizer = _tokenizer(path, max_length)
        self.batch_size = batch_size or settings.embedding_batch_size

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        out = []
        # sort by length so each batch pads to a similar size (big CPU win)
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        for start in range(0, len(order), self.batch_size):
            idx = order[start:start + self.batch_size]
            enc = self.tokenizer.encode_batch([texts[i] for i in idx])
            hidden = self.session.run(None, _feeds(self.session, enc))[0]  # (b, seq, dim)
            cls = hidden[:, 0, :]
            cls = cls / np.clip(np.linalg.norm(cls, axis=1, keepdims=True), 1e-12, None)
            out.extend(zip(idx, cls))
        out.sort(key=lambda t: t[0])
        return np.stack([v for _, v in out]) if out else np.zeros((0, 0), dtype=np.float32)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._encode(texts).tolist() if texts else []

    def embed_query(self, text: str) -> List[float]:
        return self._encode([text])[0].tolist()


class OnnxCrossEncoder:
    """Drop-in for sentence_transformers.CrossEncoder.predict (sigmoid scores)."""

    def __init__(self, model_name: str = None, max_length: int = None):
        model_name = model_name or settings.reranker_model
        path = onnx_dir(model_name)
        logger.info(f"Loading ONNX reranker from {path}")
        self.session = _session(path / MODEL_FILE)
        self.tokenizer = _tokenizer(path, max_length or settings.reranker_max_length)

    def predict(self, pairs, batch_size: int = 16, show_progress_bar: bool = False, **_):
        pairs = [tuple(p) for p in pairs]
        scores = np.zeros(len(pairs), dtype=np.float32)
        order = sorted(range(len(pairs)), key=lambda i: len(pairs[i][1]))
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            enc = self.tokenizer.encode_batch([pairs[i] for i in idx])
            logits = self.session.run(None, _feeds(self.session, enc))[0].reshape(len(idx), -1)[:, 0]
            scores[idx] = 1.0 / (1.0 + np.exp(-logits))
        return scores
