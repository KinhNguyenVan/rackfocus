"""Upload media/parquet lên S3: keyframe, scene clip, embedding shard.

Env dùng chung convention với `services/be/src/app/clients/s3.py`:
AWS_ACCESS_KEY, AWS_SECRET_KEY, AWS_REGION, AWS_BUCKET_NAME.
"""

from __future__ import annotations

import os
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
