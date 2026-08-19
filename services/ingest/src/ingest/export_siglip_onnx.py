"""Export vision tower của SigLIP sang ONNX (+ image processor đi kèm) thành 1
bundle nạp được thẳng bởi `stages/embed.py::load_siglip` — verify cosine similarity
so với PyTorch trước khi tin dùng.

Đứng độc lập, không phụ thuộc `stages.embed.load_siglip` (script đó giờ chỉ đọc
ONNX) — ở đây cần nạp checkpoint gốc bằng PyTorch/`SiglipModel` để export.

Chỉ export `get_image_features` (không export nhánh text) vì `embed_keyframes`
(`stages/embed.py`) chỉ dùng nhánh ảnh. Vector embedding này là nguồn sự thật fp32
để rebuild index (xem docstring `stages/embed.py`) — sai số dù nhỏ từ ONNX Runtime
cũng lệch vector lưu trữ, nên bắt buộc verify cosine similarity trước khi thay
PyTorch bằng ONNX ở pipeline thật.

Dùng:
    python -m ingest.export_siglip_onnx --model google/siglip-so400m-patch14-384 \
        --output-dir siglip-onnx/ [--sample-dir <thư mục ảnh thật>] \
        [--s3-uri s3://bucket/models/siglip-so400m-patch14-384-onnx]
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import torch


class _ImageEncoder(torch.nn.Module):
    """Bọc `get_image_features` + L2-normalize, khớp bước normalize trong
    `embed_keyframes` (stages/embed.py) để graph ONNX xuất thẳng vector cuối."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        feats = self.model.get_image_features(pixel_values=pixel_values)
        # transformers >= 5: trả BaseModelOutputWithPooling thay vì tensor thẳng
        # (xem note tương tự ở embed_keyframes, stages/embed.py).
        feats = getattr(feats, "pooler_output", feats)
        return feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def _resolve_source(name: str) -> str:
    """`s3://bucket/prefix` -> tải về cache dir cục bộ; ngược lại giữ nguyên
    (tên repo HF hoặc đường dẫn cục bộ)."""
    if name.startswith("s3://"):
        from .storage import download_dir

        return download_dir(name)
    return name


def load_torch_siglip(name: str):
    """Nạp SiglipModel + SiglipImageProcessor bằng PyTorch (chỉ để export, không
    dùng ở server) — `attn_implementation="eager"` để dễ trace ONNX ổn định hơn
    attention đã fuse (sdpa)."""
    from transformers import SiglipImageProcessor, SiglipModel

    source = _resolve_source(name)
    model = SiglipModel.from_pretrained(source, attn_implementation="eager").eval()
    image_processor = SiglipImageProcessor.from_pretrained(source)
    return model, image_processor


def export_onnx(model, image_size: int, onnx_path: str, opset: int = 17) -> None:
    """Export `_ImageEncoder(model)` sang ONNX, batch dimension động."""
    wrapper = _ImageEncoder(model).eval()
    dummy = torch.randn(1, 3, image_size, image_size)
    torch.onnx.export(
        wrapper, dummy, onnx_path,
        input_names=["pixel_values"], output_names=["image_embeds"],
        dynamic_axes={"pixel_values": {0: "batch"}, "image_embeds": {0: "batch"}},
        opset_version=opset,
        dynamo=False,  # exporter dynamo (mặc định ở torch mới) cần thêm onnxscript
    )


def _load_sample_pixel_values(sample_dir: str, image_processor, limit: int = 8):
    """Nạp vài ảnh thật (webp/jpg/png) qua image processor thật, để verify sát
    với dữ liệu production hơn tensor random. Trả `None` nếu không có ảnh nào."""
    from PIL import Image

    paths = sorted(
        glob.glob(os.path.join(sample_dir, "*.webp"))
        + glob.glob(os.path.join(sample_dir, "*.jpg"))
        + glob.glob(os.path.join(sample_dir, "*.png"))
    )[:limit]
    if not paths:
        return None
    images = [Image.open(p).convert("RGB") for p in paths]
    return image_processor(images=images, return_tensors="pt")["pixel_values"]


def verify_onnx(
    model, image_processor, onnx_path: str, image_size: int, sample_dir: str | None = None,
) -> None:
    """So sánh vector PyTorch vs ONNX Runtime (cosine similarity + max abs diff)."""
    import onnxruntime as ort

    pixel_values = _load_sample_pixel_values(sample_dir, image_processor) if sample_dir else None
    if pixel_values is None:
        pixel_values = torch.randn(4, 3, image_size, image_size)
        print(f"Không có ảnh mẫu, dùng {pixel_values.shape[0]} tensor random để smoke-test")

    with torch.no_grad():
        torch_out = _ImageEncoder(model).eval()(pixel_values).numpy()

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    onnx_out = session.run(None, {"pixel_values": pixel_values.numpy()})[0]

    cos_sim = (torch_out * onnx_out).sum(axis=1) / (
        np.linalg.norm(torch_out, axis=1) * np.linalg.norm(onnx_out, axis=1))
    max_abs_diff = float(np.abs(torch_out - onnx_out).max())

    print(f"Cosine similarity mỗi ảnh: {np.round(cos_sim, 6).tolist()}")
    print(f"Max abs diff: {max_abs_diff:.6e}")
    if cos_sim.min() < 0.999:
        print("CẢNH BÁO: cosine similarity < 0.999 — kiểm tra lại trước khi dùng "
              "ONNX thay PyTorch trong pipeline thật.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Export SigLIP image tower sang ONNX + verify")
    ap.add_argument("--model", default="google/siglip-so400m-patch14-384",
                     help="Tên HF repo / đường dẫn cục bộ / s3://bucket/prefix")
    ap.add_argument("--output-dir", required=True,
                     help="Thư mục ghi model.onnx + preprocessor_config.json")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--sample-dir", default=None,
                     help="Thư mục ảnh thật để verify (bỏ trống = dùng tensor random)")
    ap.add_argument("--s3-uri", default=None,
                     help="Nếu có: đẩy luôn --output-dir lên S3 sau khi verify xong")
    args = ap.parse_args()

    model, image_processor = load_torch_siglip(args.model)
    image_size = model.config.vision_config.image_size

    os.makedirs(args.output_dir, exist_ok=True)
    onnx_path = os.path.join(args.output_dir, "model.onnx")
    export_onnx(model, image_size, onnx_path, opset=args.opset)
    image_processor.save_pretrained(args.output_dir)
    print(f"Đã export ONNX bundle -> {args.output_dir}")

    verify_onnx(model, image_processor, onnx_path, image_size, args.sample_dir)

    if args.s3_uri:
        from .storage import upload_dir

        keys = upload_dir(args.output_dir, args.s3_uri)
        print(f"Đã đẩy {len(keys)} file lên {args.s3_uri}")


if __name__ == "__main__":
    main()
