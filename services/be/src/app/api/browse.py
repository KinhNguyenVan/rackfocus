"""GET /api/neighbors -- N keyframe trước/sau 1 keyframe, cho nút "Xem 25 frames
trước/sau" bên FE (trước giờ là mock, chưa nối API).

Dùng `AWSStorageHelper.get_neighbor_frames` (clients/s3.py) -- đã viết sẵn, đúng logic,
chỉ chưa từng được gọi từ đâu trong app. Client đó tự đọc AWS_* từ env.
"""
from __future__ import annotations

import re

from fastapi import APIRouter, HTTPException, Query

from ..clients.s3 import AWSStorageHelper

_FRAME_RE = re.compile(r"(\d+)\.webp$")

router = APIRouter()

_helper: AWSStorageHelper | None = None


def _get_helper() -> AWSStorageHelper:
    global _helper
    if _helper is None:
        _helper = AWSStorageHelper()
    return _helper


def _to_key(helper: AWSStorageHelper, url_or_key: str) -> str:
    """`keyframe_url` đầy đủ (FE có sẵn từ kết quả search) hoặc key trần -> key."""
    prefix = f"https://{helper.bucket}.s3.{helper.region}.amazonaws.com/"
    return url_or_key[len(prefix):] if url_or_key.startswith(prefix) else url_or_key


@router.get("/neighbors")
def neighbors(
    key: str = Query(..., description="keyframe_url đầy đủ hoặc S3 key"),
    before: int = Query(25, ge=0, le=200),
    after: int = Query(25, ge=0, le=200),
) -> dict:
    helper = _get_helper()
    s3_key = _to_key(helper, key)
    try:
        neighbor_keys = helper.get_neighbor_frames(s3_key, before=before, after=after)
    except ValueError as ex:
        raise HTTPException(400, str(ex)) from ex

    frames = []
    for k in neighbor_keys:
        m = _FRAME_RE.search(k)
        if m:
            frames.append({"url": helper.get_s3_public_url(k), "frame": int(m.group(1))})
    return {"frames": frames}
