"""Export TEXT tower của SigLIP sang ONNX (+ tokenizer) thành bundle cho search core.

Song song với `export_siglip_onnx.py` (vision tower, dùng cho ingest). Cần file riêng vì
file đó CỐ Ý chỉ export `get_image_features` — xem docstring của nó — nên bundle vision
KHÔNG encode được text query.

Ba thứ phải làm đúng, sai cái nào cũng cho vector lệch khỏi manifold của image embedding
mà không có lỗi nào báo (xem docs/search-design.md §5):

1. `padding="max_length", max_length=64`. SigLIP train với MỌI câu pad đủ 64 token, khác
   CLIP. Vì vậy graph nhận `input_ids` shape [batch, 64] cố định, không dynamic theo seq.
2. L2-normalize bake vào graph, khớp `_ImageEncoder` bên vision.
3. Canonicalize text trước tokenize (core làm, xem `encoder/text.py::canonicalize`).

Gate verify BẰNG SỐ, không phải "cosine phải cao": cosine thật của cặp ảnh-caption khớp ở
SigLIP chỉ ~0.05-0.15 (model có logit scale + bias), nên "cao" là bẫy hai chiều — thấy
0.11 rồi đi debug một graph đúng, hoặc thấy 0.11 trên graph SAI rồi ship. Dùng:
  - PyTorch vs ONNX >= 0.999 (cùng chuẩn với gate vision), VÀ
  - retrieval: caption đúng phải là top-1 trong tập caption thử.

Dùng:
    python -m ingest.export_siglip_text_onnx --model google/siglip-so400m-patch14-384 \
        --output-dir siglip-text-onnx/ \
        [--s3-uri s3://bucket/models/siglip-so400m-patch14-384-text-onnx]
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

MAX_LENGTH = 64  # SigLIP: cố định, không phải tuỳ chọn


class _TextEncoder(torch.nn.Module):
    """`get_text_features` + L2-normalize, để graph ONNX xuất thẳng vector cuối."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids):
        feats = self.model.get_text_features(input_ids=input_ids)
        # transformers >= 5 trả BaseModelOutputWithPooling thay vì tensor thẳng
        # (cùng vấn đề đã gặp ở embed.py / _ImageEncoder).
        feats = getattr(feats, "pooler_output", feats)
        return feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def load_torch_siglip(name: str):
    from transformers import AutoTokenizer, SiglipModel

    source = name
    if name.startswith("s3://"):
        from .storage import download_dir

        source = download_dir(name)
    model = SiglipModel.from_pretrained(source, attn_implementation="eager").eval()
    tokenizer = AutoTokenizer.from_pretrained(source)
    return model, tokenizer


def export_onnx(model, onnx_path: str, opset: int = 17) -> None:
    """Batch động, seq CỐ ĐỊNH 64 (SigLIP không chấp nhận độ dài khác)."""
    wrapper = _TextEncoder(model).eval()
    dummy = torch.zeros(1, MAX_LENGTH, dtype=torch.int64)
    torch.onnx.export(
        wrapper, dummy, onnx_path,
        input_names=["input_ids"], output_names=["text_embeds"],
        dynamic_axes={"input_ids": {0: "batch"}, "text_embeds": {0: "batch"}},
        opset_version=opset,
        dynamo=False,  # exporter dynamo cần thêm onnxscript
    )


def canonicalize(text: str) -> str:
    """PHẢI khớp `searchcore.encoder.text.canonicalize` — verify bằng text đã chuẩn hoá
    khác cách core chuẩn hoá thì gate không kiểm đúng cái sẽ chạy thật.

    Bản port của `canonicalize_text` trong big_vision (repo train SigLIP).
    """
    import re
    import string

    text = text.replace("_", " ").translate(
        str.maketrans("", "", string.punctuation)).lower()
    return re.sub(r"\s+", " ", text).strip()


def _input_ids(tokenizer, texts: list[str], *, framework: str = "pt"):
    return tokenizer([canonicalize(t) for t in texts], padding="max_length",
                     max_length=MAX_LENGTH, truncation=True,
                     return_tensors=framework)["input_ids"]


def verify(model, tokenizer, onnx_path: str, captions: list[str]) -> bool:
    """PyTorch vs ONNX >= 0.999. Trả True nếu qua gate."""
    import onnxruntime as ort

    input_ids = _input_ids(tokenizer, captions)

    with torch.no_grad():
        torch_out = _TextEncoder(model).eval()(input_ids).numpy()

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    onnx_out = session.run(["text_embeds"], {"input_ids": input_ids.numpy().astype(np.int64)})[0]

    cos = (torch_out * onnx_out).sum(1) / (
        np.linalg.norm(torch_out, axis=1) * np.linalg.norm(onnx_out, axis=1))
    print(f"cosine PyTorch vs ONNX: min={cos.min():.6f} "
          f"(mỗi caption: {np.round(cos, 6).tolist()})")
    print(f"max abs diff: {np.abs(torch_out - onnx_out).max():.3e}")

    norms = np.linalg.norm(onnx_out, axis=1)
    print(f"‖text_embeds‖: min={norms.min():.6f} max={norms.max():.6f} "
          "(phải ~1.0 — L2 đã bake vào graph)")

    ok = bool(cos.min() >= 0.999) and bool(np.abs(norms - 1).max() < 1e-3)
    if not ok:
        print("KHÔNG QUA GATE: cosine < 0.999 hoặc vector chưa normalize.")
    return ok


def verify_retrieval(model, tokenizer, onnx_path: str, image_path: str,
                     captions: list[str], correct_idx: int = 0) -> bool:
    """Caption đúng phải là top-1 khi so với ảnh thật.

    Đây là nửa còn lại của gate: so PyTorch-vs-ONNX chỉ chứng minh export TRUNG THỰC
    với PyTorch, không chứng minh đã export ĐÚNG NHÁNH. Xuất sai nhánh (pooling sai,
    thiếu xử lý pad) vẫn có thể đạt 0.999.
    """
    import onnxruntime as ort
    from PIL import Image
    from transformers import SiglipImageProcessor

    bundle_dir = os.path.dirname(onnx_path)
    proc_src = (bundle_dir if os.path.exists(
        os.path.join(bundle_dir, "preprocessor_config.json"))
        else "google/siglip-so400m-patch14-384")
    proc = SiglipImageProcessor.from_pretrained(proc_src)
    px = proc(images=[Image.open(image_path).convert("RGB")], return_tensors="pt")["pixel_values"]
    with torch.no_grad():
        img = model.get_image_features(pixel_values=px)
        img = getattr(img, "pooler_output", img)
        img = (img / img.norm(dim=-1, keepdim=True)).numpy()

    ids = _input_ids(tokenizer, captions, framework="np").astype(np.int64)
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    txt = session.run(["text_embeds"], {"input_ids": ids})[0]

    sims = (txt @ img[0])
    order = np.argsort(-sims)
    print("\nretrieval (cosine thật của SigLIP ~0.05-0.15, ĐỪNG kỳ vọng số lớn):")
    for r, i in enumerate(order):
        mark = " <- ĐÚNG" if i == correct_idx else ""
        print(f"  {r + 1}. {sims[i]:+.4f}  {captions[i]!r}{mark}")
    ok = int(order[0]) == correct_idx
    print("top-1 đúng" if ok else "TOP-1 SAI — có thể đã export sai nhánh")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description="Export SigLIP TEXT tower sang ONNX + verify")
    ap.add_argument("--model", default="google/siglip-so400m-patch14-384")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--s3-uri", default=None)
    ap.add_argument("--sample-image", default=None,
                     help="Ảnh thật để chạy gate retrieval (bỏ trống = bỏ qua nửa gate này)")
    ap.add_argument("--captions", nargs="*", default=[
        "a photo of a dog", "a photo of a car", "a bowl of noodles",
        "two people shaking hands"],
                     help="Caption thử; caption ĐẦU TIÊN phải là caption đúng của ảnh")
    args = ap.parse_args()

    model, tokenizer = load_torch_siglip(args.model)
    os.makedirs(args.output_dir, exist_ok=True)
    onnx_path = os.path.join(args.output_dir, "model.onnx")

    export_onnx(model, onnx_path, opset=args.opset)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Đã export text bundle -> {args.output_dir}")

    ok = verify(model, tokenizer, onnx_path, args.captions)
    if args.sample_image:
        ok = verify_retrieval(model, tokenizer, onnx_path, args.sample_image,
                              args.captions) and ok
    else:
        print("\nBỎ QUA gate retrieval (không có --sample-image). "
              "So PyTorch-vs-ONNX một mình KHÔNG phát hiện được export sai nhánh.")

    if not ok:
        raise SystemExit("Không qua gate verify — đừng đẩy bundle này lên S3.")

    if args.s3_uri:
        from .storage import upload_dir

        keys = upload_dir(args.output_dir, args.s3_uri)
        print(f"Đã đẩy {len(keys)} file lên {args.s3_uri}")


if __name__ == "__main__":
    main()
