"""
Export the embedding model and the reranker to ONNX and quantize them to int8.

    python scripts/export_onnx.py                 # both models
    python scripts/export_onnx.py --only reranker
    python scripts/export_onnx.py --verify        # compare int8 vs fp32 torch outputs

Output: data/models/onnx/<model>/{model_int8.onnx, tokenizer.json, config.json}
(~0.57 GB each instead of 2.2 GB fp32). Set MODEL_BACKEND=onnx to use them; the
vector index gets its own "-onnx-int8" slug, so documents are re-embedded once.

Quantization: ONNX Runtime dynamic quantization (weights int8 per channel, activations
quantized on the fly). Needs `onnx` and `torch` at export time only.
"""

import argparse
import gc
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

from config.settings import settings  # noqa: E402
from src.retrieval.onnx_models import MODEL_FILE, onnx_dir  # noqa: E402


def export(kind: str) -> Path:
    import torch
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from transformers import AutoModel, AutoModelForSequenceClassification

    name = settings.embedding_model if kind == "embedding" else settings.reranker_model
    src = settings.embedding_model_path if kind == "embedding" else settings.reranker_model_path
    out_dir = onnx_dir(name)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    cls = AutoModel if kind == "embedding" else AutoModelForSequenceClassification
    hf_model = cls.from_pretrained(src, dtype=torch.float32, low_cpu_mem_usage=True,
                                   attn_implementation="eager").eval()
    output = "last_hidden_state" if kind == "embedding" else "logits"

    class _Export(torch.nn.Module):
        """Takes the usual 2-D padding mask but hands the model a ready additive 4-D mask:
        transformers 5 returns a 4-D mask as-is, skipping its mask builder, which cannot be
        traced by torch.onnx (IndexError in sdpa_mask)."""

        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_ids, attention_mask):
            ext = (1.0 - attention_mask[:, None, None, :].to(torch.float32)) * torch.finfo(torch.float32).min
            return getattr(self.m(input_ids=input_ids, attention_mask=ext), output)

    model = _Export(hf_model).eval()
    ids = torch.ones(2, 16, dtype=torch.long)
    mask = torch.ones(2, 16, dtype=torch.long)
    mask[1, 10:] = 0

    with tempfile.TemporaryDirectory(dir=settings.onnx_model_dir) as tmp:
        fp32 = Path(tmp) / "model_fp32.onnx"
        with torch.no_grad():
            torch.onnx.export(
                model, (ids, mask), str(fp32),
                input_names=["input_ids", "attention_mask"],
                output_names=[output],
                dynamic_axes={"input_ids": {0: "batch", 1: "seq"},
                              "attention_mask": {0: "batch", 1: "seq"},
                              output: {0: "batch"} if kind == "reranker" else {0: "batch", 1: "seq"}},
                opset_version=17,
                dynamo=False,
            )
        del model, hf_model
        gc.collect()
        print(f"[{kind}] fp32 ONNX exported in {time.time() - t0:.0f}s")

        t1 = time.time()
        quantize_dynamic(
            str(fp32), str(out_dir / MODEL_FILE),
            weight_type=QuantType.QInt8,
            per_channel=True,
            reduce_range=True,   # safe on AVX2 CPUs without VNNI saturation issues
        )
        print(f"[{kind}] int8 quantized in {time.time() - t1:.0f}s")

    for f in ("tokenizer.json", "config.json", "special_tokens_map.json", "tokenizer_config.json"):
        if (Path(src) / f).exists():
            shutil.copy2(Path(src) / f, out_dir / f)
    size = (out_dir / MODEL_FILE).stat().st_size / 1e9
    print(f"[{kind}] -> {out_dir / MODEL_FILE} ({size:.2f} GB)")
    return out_dir


TEXTS = [
    "Vòng tròn giữa sân có bán kính bao nhiêu mét?",
    "The centre mark is at the midpoint of the halfway line. A circle with a radius of 9.15 m (10 yds) is marked around it.",
    "A throw-in is awarded when the whole of the ball passes over the touchline.",
    "MCP uses JSON-RPC 2.0 as its base message format between clients and servers.",
    "Cầu thủ bị phạt thẻ đỏ khi phạm lỗi nghiêm trọng hoặc có hành vi bạo lực.",
    "Transformers rely on self-attention to model dependencies between tokens.",
]


def verify(kind: str) -> None:
    import numpy as np

    if kind == "embedding":
        from sentence_transformers import SentenceTransformer
        from src.retrieval.onnx_models import OnnxEmbeddings

        ref = SentenceTransformer(settings.embedding_model_path, device="cpu").encode(TEXTS, normalize_embeddings=True)
        gc.collect()
        q = np.array(OnnxEmbeddings().embed_documents(TEXTS))
        cos = (ref * q).sum(axis=1)
        print(f"[embedding] cosine(fp32, int8) per text: {np.round(cos, 4).tolist()}  min {cos.min():.4f}")
        sim_ref, sim_q = ref @ ref.T, q @ q.T
        print(f"[embedding] max |Δ similarity| between texts: {np.abs(sim_ref - sim_q).max():.4f}")
    else:
        from sentence_transformers import CrossEncoder
        from src.retrieval.onnx_models import OnnxCrossEncoder

        pairs = [(TEXTS[0], t) for t in TEXTS[1:]] + [(TEXTS[2], t) for t in TEXTS[1:]]
        ref = CrossEncoder(settings.reranker_model_path, max_length=512, device="cpu").predict(pairs)
        gc.collect()
        q = OnnxCrossEncoder().predict(pairs)
        print(f"[reranker] fp32: {np.round(ref, 3).tolist()}")
        print(f"[reranker] int8: {np.round(q, 3).tolist()}")
        same = [np.argsort(-ref[i:i + 5]).tolist() == np.argsort(-q[i:i + 5]).tolist() for i in (0, 5)]
        print(f"[reranker] max |Δ score| {np.abs(ref - q).max():.3f}; identical ranking: {same}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["embedding", "reranker"])
    ap.add_argument("--verify", action="store_true", help="compare existing int8 models with fp32")
    args = ap.parse_args()
    settings.onnx_model_dir.mkdir(parents=True, exist_ok=True)
    for kind in ([args.only] if args.only else ["embedding", "reranker"]):
        verify(kind) if args.verify else export(kind)
        gc.collect()
