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

Chạy CPU hay GPU: xem `choose_providers`. Mặc định `auto` — có GPU thì dùng, không thì
CPU. Bản thân file model.onnx không đổi (fp32 chạy được trên cả hai), nên bật GPU không
cần export lại bundle.
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

CPU_EP = "CPUExecutionProvider"
# EP được thử khi SC_ENCODER_PROVIDERS=auto. TensorRT CỐ Ý không có trong danh sách:
# nó build engine ở lần chạy đầu (vài phút) và cache theo đúng GPU đó, nên bật tự động
# sẽ biến "pod lên sau 200s" thành "pod treo im 5 phút" mà không ai hiểu vì sao.
GPU_EPS = ("CUDAExecutionProvider", "ROCMExecutionProvider")
_EP_ALIAS = {
    "cuda": "CUDAExecutionProvider",
    "rocm": "ROCMExecutionProvider",
    "coreml": "CoreMLExecutionProvider",
    "tensorrt": "TensorrtExecutionProvider",
}


def choose_providers(available: list[str], want: str = "auto") -> tuple[list[str], str]:
    """Chọn execution provider cho ONNX Runtime. Trả `(providers, warning)`.

    `want` (env `SC_ENCODER_PROVIDERS`):
      * `auto` (mặc định) — có GPU thì dùng, không có thì CPU, không cảnh báo gì.
      * `cpu` — ép CPU.
      * danh sách EP cách nhau bởi dấu phẩy (`cuda`, `CUDAExecutionProvider`, ...) — ép
        đúng cái đó.

    Hai quyết định cần giải thích:

    1. CPU LUÔN được thêm vào cuối. ORT chỉ thử EP kế tiếp cho node mà EP trước không
       nhận; không có CPU chốt hạ thì một op lạ làm chết cả session.
    2. `want` chỉ định GPU mà máy không có GPU thì **cảnh báo rồi chạy CPU**, không raise.
       Cảnh báo vì chạy CPU im lặng là tệ nhất — chậm hơn ~20x mà không dấu hiệu gì.
       Không raise vì lúc thi thà chậm còn hơn pod không lên.
    """
    avail = list(available)
    w = (want or "auto").strip().lower()

    if w in ("", "auto"):
        for gpu in GPU_EPS:
            if gpu in avail:
                return [gpu, CPU_EP], ""
        return [CPU_EP], ""

    if w == "cpu":
        return [CPU_EP], ""

    asked = [_EP_ALIAS.get(p.strip().lower(), p.strip())
             for p in want.split(",") if p.strip()]
    keep = [p for p in asked if p in avail]
    missing = [p for p in asked if p not in avail]

    warning = ""
    if missing:
        warning = (
            f"SC_ENCODER_PROVIDERS yêu cầu {missing} nhưng onnxruntime chỉ có {avail}. "
            f"Sẽ chạy {keep or [CPU_EP]}. Nguyên nhân thường gặp: cài gói `onnxruntime` "
            f"(chỉ CPU) thay vì `onnxruntime-gpu[cuda,cudnn]`, hoặc pod không có GPU.")

    if CPU_EP not in keep:
        keep.append(CPU_EP)
    return keep, warning


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
        # số encode đồng thời bằng semaphore bên dưới. (Chỉ có tác dụng với CPU EP.)
        so.intra_op_num_threads = int(os.environ.get("OMP_NUM_THREADS", "4"))
        so.inter_op_num_threads = 1

        self.session = _make_session(ort, onnx_path, so)
        self.providers = list(self.session.get_providers())
        self.on_gpu = any(p in GPU_EPS for p in self.providers)

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
        log.info("text encoder %s: dim=%d, tokenizer=%s, EP=%s, max_concurrent=%d",
                 self.name, self.dim, self._kind, self.providers[0], max_concurrent)
        if self.on_gpu and max_concurrent <= 2:
            # Mặc định 2 được chọn cho CPU (xem config.py): 8 worker x 4 BLAS thread trên
            # 4 vCPU là oversubscription. Trên GPU thì cái nghẽn là 1 stream CUDA chứ
            # không phải thread CPU, nên 2 chỉ làm GPU nằm chờ.
            log.info("đang chạy %s: SC_MAX_CONCURRENT_ENCODES=%d là con số tính cho CPU, "
                     "nâng lên ~8 để dùng hết GPU", self.providers[0], max_concurrent)

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


def _make_session(ort, onnx_path: str, so):
    """Tạo InferenceSession theo EP đã chọn, rơi về CPU nếu không dựng được.

    Cần try/except chứ không chỉ dựa vào `get_available_providers()`: gói
    `onnxruntime-gpu` LUÔN khai CUDAExecutionProvider là "available" kể cả trên máy
    không có GPU hay thiếu driver — lỗi chỉ lộ ra lúc tạo session.
    """
    want = os.environ.get("SC_ENCODER_PROVIDERS", "auto")
    providers, warning = choose_providers(ort.get_available_providers(), want)
    if warning:
        log.warning("%s", warning)

    if any(p in GPU_EPS for p in providers):
        # onnxruntime-gpu mới nạp CUDA/cuDNN từ các wheel nvidia-* (cài kèm qua
        # `onnxruntime-gpu[cuda,cudnn]`). Không gọi thì session vẫn tạo được nhưng CUDA
        # EP im lặng không dùng được -> chạy CPU mà log lại ghi là CUDA.
        if hasattr(ort, "preload_dlls"):
            try:
                ort.preload_dlls()
            except Exception as ex:                       # noqa: BLE001
                log.warning("ort.preload_dlls() lỗi (%s) — CUDA có thể không nạp được", ex)

        # TF32 mặc định BẬT trên Ampere trở lên: mantissa 10 bit thay vì 23. Qua 27 layer
        # thì vector lệch khỏi bản CPU đủ để đảo thứ tự các hit gần bằng điểm nhau — đúng
        # loại lỗi âm thầm mà docstring đầu file cảnh báo. Tắt để GPU và CPU cho cùng kết
        # quả; đặt SC_CUDA_TF32=1 nếu cần đổi độ chính xác lấy tốc độ (đo bằng
        # ops/runpod/gpu_parity.py trước khi bật).
        if os.environ.get("SC_CUDA_TF32", "0") != "1":
            providers = [(p, {"use_tf32": 0}) if p == "CUDAExecutionProvider" else p
                         for p in providers]

    try:
        return ort.InferenceSession(onnx_path, so, providers=providers)
    except Exception as ex:                               # noqa: BLE001
        names = [p[0] if isinstance(p, tuple) else p for p in providers]
        if names == [CPU_EP]:
            raise
        log.warning("không dựng được session với %s (%s) -> rơi về CPU", names, ex)
        return ort.InferenceSession(onnx_path, so, providers=[CPU_EP])


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
