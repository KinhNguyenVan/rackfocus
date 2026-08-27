"""Browse keyframe lân cận và nối chúng về scene clip tương ứng.

Video gốc hiện không có trong bucket. S3 chỉ chứa keyframe và các ``scene_XXX.mp4``;
vì vậy API trả đủ timeline metadata để frontend phát scene clip rồi seek tương đối bằng
``keyframe_time - start_sec``. Khi có full video, seek tuyệt đối vẫn là
``keyframe_time`` và không cần đổi contract này.
"""
from __future__ import annotations

import json
import re
from bisect import bisect_right
from functools import lru_cache
from urllib.parse import unquote, urlparse

from botocore.exceptions import BotoCoreError, ClientError
from fastapi import APIRouter, HTTPException, Query

from ..clients.s3 import AWSStorageHelper

_KEYFRAME_RE = re.compile(
    r"^Keyframes_(?P<group>[A-Za-z0-9_-]+)/keyframes/"
    r"(?P<video>[A-Za-z0-9_-]+)/(?P<frame>\d+)\.webp$"
)

router = APIRouter()

_helper: AWSStorageHelper | None = None


def _get_helper() -> AWSStorageHelper:
    global _helper
    if _helper is None:
        _helper = AWSStorageHelper()
    return _helper


def _to_key(helper: AWSStorageHelper, url_or_key: str) -> str:
    """Chuẩn hoá URL/key và chỉ chấp nhận keyframe thuộc đúng bucket cấu hình."""
    parsed = urlparse(url_or_key)
    if parsed.scheme:
        allowed_hosts = {
            f"{helper.bucket}.s3.{helper.region}.amazonaws.com",
            f"{helper.bucket}.s3.amazonaws.com",
        }
        if parsed.scheme != "https" or parsed.hostname not in allowed_hosts:
            raise ValueError("URL keyframe không thuộc bucket S3 đã cấu hình")
        key = unquote(parsed.path.lstrip("/"))
    else:
        key = url_or_key.lstrip("/")
    if not _KEYFRAME_RE.fullmatch(key):
        raise ValueError("Keyframe phải có dạng Keyframes_<group>/keyframes/<video>/<frame>.webp")
    return key


@lru_cache(maxsize=512)
def _load_scenes(scenes_key: str) -> tuple[dict, ...]:
    helper = _get_helper()
    response = helper.s3_client.get_object(Bucket=helper.bucket, Key=scenes_key)
    payload = json.loads(response["Body"].read())
    if not isinstance(payload, list):
        raise ValueError(f"{scenes_key} không chứa danh sách scene")
    return tuple(payload)


def _scene_for_frame(scenes: tuple[dict, ...], frame: int) -> dict | None:
    if not scenes:
        return None
    starts = [int(scene["start_frame"]) for scene in scenes]
    index = bisect_right(starts, frame) - 1
    if index < 0:
        return None
    scene = scenes[index]
    return scene if frame <= int(scene["end_frame"]) else None


def _frame_time(scene: dict, frame: int) -> float:
    """Nội suy timestamp từ map frame/time đã validate trong scenes.json."""
    first = int(scene["start_frame"])
    last = int(scene["end_frame"])
    start = float(scene["start_time"])
    end = float(scene["end_time"])
    if last <= first:
        return start
    ratio = (frame - first) / (last - first)
    return start + ratio * (end - start)


@router.get("/neighbors")
def neighbors(
    key: str = Query(..., description="keyframe_url đầy đủ hoặc S3 key"),
    before: int = Query(25, ge=0, le=200),
    after: int = Query(25, ge=0, le=200),
    to_key: str | None = Query(
        None, description="keyframe_url/S3 key thứ 2, cùng video -- trả về tất cả "
        "frame giữa `key` và `to_key` cộng before/after. Rỗng = hành vi 1-mỏ neo cũ."),
) -> dict:
    helper = _get_helper()
    try:
        s3_key = _to_key(helper, key)
        s3_to_key = _to_key(helper, to_key) if to_key else None
        match = _KEYFRAME_RE.fullmatch(s3_key)
        assert match is not None
        group = match.group("group")
        video = match.group("video")
        current_frame = int(match.group("frame"))
        scenes_key = f"Keyscence_{group}/keyscence/{video}/scenes.json"

        neighbor_keys = helper.get_neighbor_frames(
            s3_key, before=before, after=after, to_key=s3_to_key)
        scenes = _load_scenes(scenes_key)
    except ValueError as ex:
        raise HTTPException(400, str(ex)) from ex
    except (BotoCoreError, ClientError, json.JSONDecodeError) as ex:
        raise HTTPException(502, f"Không đọc được media metadata từ S3: {type(ex).__name__}") from ex

    frames = []
    for neighbor_key in neighbor_keys:
        item = _KEYFRAME_RE.fullmatch(neighbor_key)
        if item is None:
            continue
        frame = int(item.group("frame"))
        scene = _scene_for_frame(scenes, frame)
        if scene is None:
            continue
        start_sec = float(scene["start_time"])
        end_sec = float(scene["end_time"])
        frames.append({
            "url": helper.get_s3_public_url(neighbor_key),
            "frame": frame,
            "keyframe_time": round(_frame_time(scene, frame), 6),
            "scene_idx": int(scene["scene_id"]),
            "start_sec": start_sec,
            "end_sec": end_sec,
            "clip_url": str(scene["scene_url"]),
            "is_current": frame == current_frame,
        })

    return {
        "video_name": video,
        "current_frame": current_frame,
        "frames": frames,
        "playback_source": "scene_clip",
    }
