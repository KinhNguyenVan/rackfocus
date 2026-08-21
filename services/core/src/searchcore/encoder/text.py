"""ONNX text tower của SigLIP. Encode query text -> vector cùng không gian với keyframe.

Ba bẫy của SigLIP, sai bất kỳ cái nào cũng cho vector lệch khỏi manifold của image
embedding mà KHÔNG có lỗi nào báo — rất dễ bị quy oan cho "tag tệ" hoặc "recall HNSW thấp".
Xem docs/search-design.md §5.

1. `padding="max_length", max_length=64` — BẮT BUỘC, khác CLIP. Dynamic padding ra vector
   sai vì SigLIP train với mọi câu pad đủ 64 token.
2. L2-normalize phải bake trong graph ONNX (giống `_ImageEncoder` bên vision). Ở đây vẫn
   normalize lại một lần nữa cho chắc — idempotent nên vô hại.
3. Canonicalization: SigLIP chuẩn hoá text trước khi tokenize (bỏ dấu câu, lowercase,
   gộp khoảng trắng). Bỏ qua bước này thì query lệch nhẹ và âm thầm.
"""
from __future__ import annotations

import logging
import os
import re
import string
import threading

import numpy as np

from .base import Encoder, resolve_bundle

log = logging.getLogger("searchcore.encoder.text")

MAX_LENGTH = 64          # SigLIP: cố định, không phải tuỳ chọn
_PUNCT = str.maketrans("", "", string.punctuation)


def canonicalize(text: str) -> str:
    """Bản port của `canonicalize_text` trong big_vision (repo train SigLIP).

    Idempotent nên gọi thêm một lần dù tokenizer HF đã làm cũng không sao.
    """
    text = text.replace("_", " ").translate(_PUNCT).lower()
    return re.sub(r"\s+", " ", text).strip()


class SigLipTextEncoder(Encoder):
    """`model.onnx` (input_ids -> text_embeds) + tokenizer, nạp 1 lần lúc start server."""

    def __init__(self, source: str, *, name: str = "", max_concurrent: int = 2,
                 max_length: int = MAX_LENGTH):
        import onnxruntime as ort

        path = resolve_bundle(source)
        onnx_path = os.path.join(path, "model.onnx")
        if not os.path.exists(onnx_path):
            raise RuntimeError(
                f"Thiếu model.onnx trong {path}. Bundle này phải là TEXT tower — "
                "export_siglip_onnx.py chỉ export vision (get_image_features), "
                "dùng export_siglip_text_onnx.py để tạo bundle text.")

        so = ort.SessionOptions()
        # Encoder đã ngốn 53.3 GFLOP/query; để ORT tự spawn thread trên máy 4 vCPU cùng
        # với 8 gRPC worker là 8x oversubscription -> p99 sụp. Giới hạn ở đây, và chặn
        # số encode đồng thời bằng semaphore bên dưới.
        so.intra_op_num_threads = int(os.environ.get("OMP_NUM_THREADS", "4"))
        so.inter_op_num_threads = 1
        self.session = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])

        outs = [o.name for o in self.session.get_outputs()]
        if "text_embeds" not in outs:
            raise RuntimeError(
                f"ONNX output là {outs}, cần 'text_embeds'. Bundle vision (image_embeds) "
                "KHÔNG encode được text.")

        self._tokenizer, self._kind, self._pad_id = _load_tokenizer(path)
        self.max_length = max_length
        self.name = name or os.path.basename(path.rstrip("/"))
        # Semaphore, không phải lock: cho phép max_concurrent encode song song.
        self._sem = threading.BoundedSemaphore(max(1, max_concurrent))

        probe = self.encode(["a"])
        self.dim = int(probe.shape[1])
        log.info("text encoder %s: dim=%d, tokenizer=%s, max_concurrent=%d",
                 self.name, self.dim, self._kind, max_concurrent)

    def _tokenize(self, texts: list[str]) -> np.ndarray:
        canon = [canonicalize(t) for t in texts]
        if self._kind == "tokenizers":
            self._tokenizer.enable_padding(length=self.max_length,
                                           pad_id=self._pad_id, pad_token="<pad>")
            self._tokenizer.enable_truncation(max_length=self.max_length)
            encs = self._tokenizer.encode_batch(canon)
            return np.asarray([e.ids for e in encs], dtype=np.int64)
        out = self._tokenizer(canon, padding="max_length", max_length=self.max_length,
                              truncation=True, return_tensors="np")
        return out["input_ids"].astype(np.int64)

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dim or 0), dtype=np.float32)
        input_ids = self._tokenize(texts)
        with self._sem:
            feats = self.session.run(["text_embeds"], {"input_ids": input_ids})[0]
        feats = np.asarray(feats, dtype=np.float32)
        # Bake trong graph rồi, nhưng normalize lại là idempotent và rẻ — bảo hiểm cho
        # trường hợp bundle được export thiếu bước đó.
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        return feats / np.maximum(norms, 1e-12)


def _load_tokenizer(path: str) -> tuple[object, str, int]:
    """(tokenizer, kind, pad_id). Ưu tiên `tokenizers` (Rust, nhẹ) nếu có tokenizer.json.

    Lý do ưu tiên: kéo `transformers` + `sentencepiece` vào core thêm ~1GB deps, ngược
    với chủ trương "core chỉ ONNX" của I1.
    """
    tj = os.path.join(path, "tokenizer.json")
    if os.path.exists(tj):
        try:
            from tokenizers import Tokenizer

            tok = Tokenizer.from_file(tj)
            pad = tok.token_to_id("<pad>")
            if pad is None:
                pad = tok.token_to_id("</s>")
            return tok, "tokenizers", int(pad or 0)
        except ImportError:
            pass

    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(path), "transformers", 0


def load(source: str, *, name: str = "", max_concurrent: int = 2) -> SigLipTextEncoder:
    return SigLipTextEncoder(source, name=name, max_concurrent=max_concurrent)
