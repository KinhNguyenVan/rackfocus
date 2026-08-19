"""Upload media/parquet lên S3: keyframe, scene clip, embedding shard.

Env dùng chung convention với `services/be/src/app/clients/s3.py`:
AWS_ACCESS_KEY, AWS_SECRET_KEY, AWS_REGION, AWS_BUCKET_NAME.
"""

from __future__ import annotations

import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed

_client = None


def get_client():
    """boto3 S3 client, lazy singleton (đọc AWS_* từ env)."""
    global _client
    if _client is None:
        import boto3

        access_key = os.environ.get("AWS_ACCESS_KEY")
        secret_key = os.environ.get("AWS_SECRET_KEY")
        region = os.environ.get("AWS_REGION")
        if access_key and secret_key:
            _client = boto3.client(
                "s3",
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
                region_name=region,
            )
        else:
            _client = boto3.client("s3", region_name=region)
    return _client


def _bucket() -> str:
    bucket = os.environ.get("AWS_BUCKET_NAME")
    if not bucket:
        raise RuntimeError("Thiếu env AWS_BUCKET_NAME")
    return bucket


def upload_file(local_path: str, key: str) -> str:
    """Upload 1 file lên S3, trả về S3 key đã upload (dùng key này để ghi vào manifest)."""
    get_client().upload_file(local_path, _bucket(), key)
    return key


def upload_many(pairs: list[tuple[str, str]], max_workers: int = 8) -> list[str]:
    """Upload song song danh sách (local_path, key). Lỗi ở 1 file không chặn các file khác.

    Trả về key nếu upload thành công, `None` nếu lỗi — cùng thứ tự với `pairs`.
    """
    get_client()  # khởi tạo trước khi mở thread pool, tránh nhiều thread cùng tạo client

    results: list[str | None] = [None] * len(pairs)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(upload_file, local_path, key): i
            for i, (local_path, key) in enumerate(pairs)
        }
        for future in as_completed(futures):
            i = futures[future]
            try:
                results[i] = future.result()
            except Exception as ex:  # noqa: BLE001 — 1 file lỗi không chặn các file khác
                print(f"  ! Lỗi upload {pairs[i][0]}: {ex}")
    return results


def upload_video_outputs(video_out_dir: str, video_id: str) -> dict[str, str | None]:
    """Upload keyframes/, scenes/ và embed_*.parquet của 1 video lên S3.

    Key trên S3 giữ nguyên path tương đối trong `video_out_dir`, tiền tố bằng
    `video_id` (vd `keyframes/000012.webp` -> `<video_id>/keyframes/000012.webp`).
    Trả về mapping {local_relative_path: s3_key hoặc None nếu lỗi}.
    """
    pairs: list[tuple[str, str]] = []
    for root, _dirs, files in os.walk(video_out_dir):
        for name in files:
            if not (name.endswith((".webp", ".mp4", ".parquet"))):
                continue
            local_path = os.path.join(root, name)
            rel = os.path.relpath(local_path, video_out_dir).replace("\\", "/")
            key = f"{video_id}/{rel}"
            pairs.append((local_path, key))

    if not pairs:
        return {}

    uploaded = upload_many(pairs)
    return {
        os.path.relpath(local_path, video_out_dir).replace("\\", "/"): key
        for (local_path, _dest_key), key in zip(pairs, uploaded)
    }


# =====================================================================
# Checkpoint model (vd SigLIP): đẩy lên S3 1 lần, server tự tải cache cục bộ
# thay vì gọi HuggingFace Hub mỗi lần khởi động (xem `stages/embed.py::load_siglip`).
# =====================================================================
def parse_s3_uri(s3_uri: str) -> tuple[str, str]:
    """`s3://bucket/prefix/...` -> `(bucket, prefix)` (prefix không có `/` đầu/cuối)."""
    if not s3_uri.startswith("s3://"):
        raise ValueError(f"s3_uri phải bắt đầu bằng s3://, nhận: {s3_uri}")
    bucket, _, prefix = s3_uri[len("s3://"):].partition("/")
    if not bucket:
        raise ValueError(f"s3_uri thiếu bucket: {s3_uri}")
    return bucket, prefix.strip("/")


def upload_dir(local_dir: str, s3_uri: str) -> list[str]:
    """Đẩy toàn bộ file trong `local_dir` (checkpoint đã tải sẵn về máy, vd qua
    `model.save_pretrained(local_dir)`) lên S3 dưới `s3_uri` (`s3://bucket/prefix`).

    Chạy 1 lần offline; sau đó server dùng `download_dir` để nạp lại từ S3.
    """
    bucket, prefix = parse_s3_uri(s3_uri)
    client = get_client()
    keys = []
    for root, _dirs, files in os.walk(local_dir):
        for name in files:
            local_path = os.path.join(root, name)
            rel = os.path.relpath(local_path, local_dir).replace("\\", "/")
            key = f"{prefix}/{rel}" if prefix else rel
            client.upload_file(local_path, bucket, key)
            keys.append(key)
    return keys


def download_dir(s3_uri: str, cache_root: str | None = None) -> str:
    """Tải toàn bộ object dưới `s3_uri` (`s3://bucket/prefix`) về 1 cache dir cục bộ,
    trả về đường dẫn cache dir đó (dùng thẳng cho `from_pretrained(local_dir)`).

    Bỏ qua tải lại nếu cache dir đã có file marker `.s3_download_done` — nếu
    `cache_root` là volume bền (persistent) giữa các lần khởi động container, model
    (~vài GB) chỉ tải từ S3 đúng 1 lần thay vì lặp lại mỗi lần server/worker start.
    """
    bucket, prefix = parse_s3_uri(s3_uri)
    cache_root = cache_root or os.environ.get(
        "MODEL_CACHE_DIR", os.path.join(tempfile.gettempdir(), "rackfocus_models"))
    local_dir = os.path.join(cache_root, bucket, prefix.replace("/", os.sep))
    marker = os.path.join(local_dir, ".s3_download_done")
    if os.path.exists(marker):
        return local_dir

    client = get_client()
    list_prefix = f"{prefix}/" if prefix else ""
    keys = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=list_prefix):
        for obj in page.get("Contents", []):
            if not obj["Key"].endswith("/"):
                keys.append(obj["Key"])
    if not keys:
        raise RuntimeError(f"Không thấy file nào ở {s3_uri}")

    os.makedirs(local_dir, exist_ok=True)
    for key in keys:
        rel = key[len(list_prefix):]
        dest = os.path.join(local_dir, rel.replace("/", os.sep))
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        client.download_file(bucket, key, dest)

    open(marker, "w").close()
    return local_dir
