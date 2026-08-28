"""Pydantic: GET /api/transcript/suggest — gợi ý scene theo keyword trong lời thoại.

Tách khỏi `api/transcript.py` để hợp đồng HTTP nằm một chỗ (FE đọc file này là đủ). Mỗi
item đủ dữ liệu để FE mở scene bằng `VideoPlayer` sẵn có: `clip_url` để phát, `start_sec`
để seek về đầu scene, `keyframe_url` làm thumbnail.
"""
from __future__ import annotations

from pydantic import BaseModel


class TranscriptSuggestItem(BaseModel):
    video_name: str
    scene_idx: int
    start_sec: float
    end_sec: float
    # URL công khai đã resolve từ key (S3) — FE dùng thẳng làm <video src> / <img src>.
    clip_url: str = ""
    keyframe_url: str = ""
    # Đoạn lời thoại quanh keyword, keyword bọc trong [ ] để FE highlight (snippet() FTS5).
    snippet: str = ""


class TranscriptSuggestResponse(BaseModel):
    query: str
    items: list[TranscriptSuggestItem]
