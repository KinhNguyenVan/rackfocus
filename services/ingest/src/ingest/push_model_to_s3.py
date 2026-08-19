"""Đẩy 1 checkpoint HuggingFace (vd SigLIP gốc, trước khi export ONNX) lên S3 một
lần, để `export_siglip_onnx.py --model s3://...` export được mà không cần gọi
HuggingFace Hub mỗi lần.

Chỉ lưu model + image processor (không lưu tokenizer) — pipeline (`stages/embed.py`)
chỉ dùng nhánh ảnh, `export_siglip_onnx.py` cũng chỉ cần `SiglipImageProcessor`.

Dùng: python -m ingest.push_model_to_s3 --model google/siglip-so400m-patch14-384 \
    --s3-uri s3://<bucket>/models/siglip-so400m-patch14-384
"""

from __future__ import annotations

import argparse
import tempfile

from .storage import upload_dir


def push_model(model_name: str, s3_uri: str) -> list[str]:
    """Tải `model_name` từ HuggingFace Hub về thư mục tạm, rồi đẩy nguyên thư mục
    (config, weights, image processor) lên `s3_uri`."""
    from transformers import SiglipImageProcessor, SiglipModel

    model = SiglipModel.from_pretrained(model_name)
    image_processor = SiglipImageProcessor.from_pretrained(model_name)

    with tempfile.TemporaryDirectory() as tmp_dir:
        model.save_pretrained(tmp_dir)
        image_processor.save_pretrained(tmp_dir)
        keys = upload_dir(tmp_dir, s3_uri)

    print(f"Đã đẩy {len(keys)} file lên {s3_uri}")
    return keys


def main() -> None:
    ap = argparse.ArgumentParser(description="Đẩy checkpoint SigLIP lên S3")
    ap.add_argument("--model", default="google/siglip-so400m-patch14-384",
                     help="Tên repo HuggingFace của checkpoint")
    ap.add_argument("--s3-uri", required=True,
                     help="Đích trên S3, dạng s3://<bucket>/<prefix>")
    args = ap.parse_args()
    push_model(args.model, args.s3_uri)


if __name__ == "__main__":
    main()
