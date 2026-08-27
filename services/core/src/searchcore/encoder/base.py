"""Interface encoder + tải bundle từ S3.

Encoder query là NGOẠI LỆ DUY NHẤT được chạy trong hot path (Handoff I1), và phải nạp sẵn
trong RAM (ONNX). Handoff I1 ghi "ONNX/CPU"; giờ chạy được cả GPU — xem
`text.choose_providers`, mặc định `auto` nên CPU vẫn là đường mặc định khi không có GPU.

Chi phí thực tế phải biết trước: SigLIP so400m text tower = 449.3M param, 53.3 GFLOP mỗi
query ở batch 1 / seq 64. Chi phí này CỐ ĐỊNH vì SigLIP buộc padding="max_length" — query
3 chữ trả giá y như query 60 token. Xem docs/search-design.md §1. Đo thật trên M2:

    CPU 1 thread   938ms   0.94 CPU-sec/query
    CPU 2 thread   517ms   1.04 CPU-sec/query
    CPU 4 thread   521ms   1.76 CPU-sec/query   <- thêm CPU, latency y nguyên

Batch KHÔNG giúp trên CPU (batch 1/4/8 đều ~0.95 CPU-sec/query) vì đã compute-bound.
Đó là lý do 53.3 GFLOP/query đẩy sang GPU là thay đổi duy nhất có ý nghĩa về throughput.
"""
from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod

import numpy as np

log = logging.getLogger("searchcore.encoder")


class Encoder(ABC):
    """Trả vector fp32 đã L2-normalize, cùng không gian với embedding trong snapshot."""

    dim: int = 0
    name: str = ""

    @abstractmethod
    def encode(self, texts: list[str]) -> np.ndarray:
        """(len(texts), dim) fp32, L2-normalized."""

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


def resolve_bundle(source: str, cache_root: str | None = None) -> str:
    """`s3://bucket/prefix` -> tải về cache dir; đường dẫn cục bộ -> giữ nguyên.

    Dùng chung `ingest.storage.download_dir` (marker `.s3_download_done`) nếu import
    được; nếu không thì tự tải bằng boto3 với cùng quy ước marker.

    CẢNH BÁO vận hành: marker này KHÔNG có invalidation. Cache dir phải key theo
    version/prefix và prefix phải immutable, nếu không host có marker sẽ serve bản cũ
    mãi mãi. Xem docs/search-design.md §7.
    """
    if not source.startswith("s3://"):
        return source

    cache_root = cache_root or os.environ.get("MODEL_CACHE_DIR", "/var/cache/searchcore")

    try:
        from ingest.storage import download_dir  # type: ignore

        return download_dir(source, cache_root)
    except ImportError:
        pass

    import boto3

    bucket, _, prefix = source[len("s3://"):].partition("/")
    prefix = prefix.strip("/")
    local = os.path.join(cache_root, bucket, prefix.replace("/", os.sep))
    marker = os.path.join(local, ".s3_download_done")
    if os.path.exists(marker):
        return local

    client = boto3.client(
        "s3",
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY") or None,
        aws_secret_access_key=os.environ.get("AWS_SECRET_KEY") or None,
        region_name=os.environ.get("AWS_REGION"),
    )
    list_prefix = f"{prefix}/" if prefix else ""
    keys = [
        obj["Key"]
        for page in client.get_paginator("list_objects_v2").paginate(
            Bucket=bucket, Prefix=list_prefix)
        for obj in page.get("Contents", [])
        if not obj["Key"].endswith("/")
    ]
    if not keys:
        raise RuntimeError(f"Không thấy object nào ở {source}")

    os.makedirs(local, exist_ok=True)
    for key in keys:
        dest = os.path.join(local, key[len(list_prefix):].replace("/", os.sep))
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        client.download_file(bucket, key, dest)
    open(marker, "w").close()
    log.info("tải %d file từ %s -> %s", len(keys), source, local)
    return local
