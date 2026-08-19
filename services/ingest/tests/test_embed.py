"""Test payload embedding: gán scene cho keyframe, dựng payload khớp `Payload` proto.

Chỉ test phần logic thuần (không cần model/GPU/Postgres/S3): `assign_scene_idx`,
`build_payload_rows`, `keyframe_point_id`.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ingest.stages.embed import (
    TIER_KEYFRAME,
    assign_scene_idx,
    build_payload_rows,
    keyframe_point_id,
)


def _keyframes(frames):
    return [
        {"frame": f, "timestamp": round(f / 25.0, 3), "keyframe_url": f"keyframes/{f:06d}.webp"}
        for f in frames
    ]


def _scenes():
    return [
        {"scene_id": 0, "start_frame": 0, "end_frame": 100,
         "start_time": 0.0, "end_time": 4.0, "script": "hello world",
         "scene_url": "scenes/scene_000.mp4"},
        {"scene_id": 1, "start_frame": 101, "end_frame": 200,
         "start_time": 4.04, "end_time": 8.0, "script": "",
         "scene_url": None},
    ]


# --------------------------- assign_scene_idx ---------------------------
def test_keyframe_maps_to_scene_containing_its_frame():
    keyframes = _keyframes([0, 50, 150])
    idx = assign_scene_idx(keyframes, _scenes())
    assert idx == [0, 0, 1]


def test_keyframe_beyond_last_end_frame_falls_into_last_scene():
    # Frame 205 vượt end_frame=200 của scene cuối do làm tròn lúc trích keyframe.
    keyframes = _keyframes([205])
    idx = assign_scene_idx(keyframes, _scenes())
    assert idx == [1]


def test_single_scene_all_keyframes_map_to_it():
    keyframes = _keyframes([0, 10, 30])
    scenes = [{"scene_id": 0, "start_frame": 0, "end_frame": 30,
               "start_time": 0.0, "end_time": 1.2, "script": "", "scene_url": None}]
    assert assign_scene_idx(keyframes, scenes) == [0, 0, 0]


# --------------------------- keyframe_point_id ---------------------------
def test_point_id_is_deterministic_and_unique_per_frame():
    assert keyframe_point_id(1, 0) == keyframe_point_id(1, 0)
    assert keyframe_point_id(1, 0) != keyframe_point_id(1, 1)
    assert keyframe_point_id(1, 0) != keyframe_point_id(2, 0)
    assert keyframe_point_id(1, 5) == (1 << 32) | 5


# --------------------------- build_payload_rows ---------------------------
def test_payload_fields_match_proto_payload_shape():
    keyframes = _keyframes([0, 150])
    rows = build_payload_rows(video_id=7, keyframes=keyframes, scenes=_scenes())

    assert len(rows) == 2

    first, second = rows
    assert first["point_id"] == keyframe_point_id(7, 0)
    assert first["video_id"] == 7
    assert first["scene_idx"] == 0
    assert first["has_speech"] is True
    assert first["keyframe_key"] == "7/keyframes/000000.webp"
    assert first["clip_key"] == "7/scenes/scene_000.mp4"
    assert first["tier"] == TIER_KEYFRAME
    assert first["objects"] == []
    assert first["has_ocr"] is False

    assert second["scene_idx"] == 1
    assert second["has_speech"] is False
    assert second["clip_key"] is None


def test_payload_start_end_sec_use_scene_time_range_not_keyframe_instant():
    # Keyframe ở frame 50 (timestamp 2.0s) thuộc scene 0 (start_time=0.0, end_time=4.0).
    # start_sec/end_sec phải là khoảng thời gian của SCENE (để Filter.min_start_sec/
    # max_end_sec/*_duration_sec lọc có ý nghĩa), không phải trùng keyframe_time.
    keyframes = _keyframes([50])
    rows = build_payload_rows(video_id=1, keyframes=keyframes, scenes=_scenes())
    assert rows[0]["keyframe_time"] == 2.0
    assert rows[0]["start_sec"] == 0.0
    assert rows[0]["end_sec"] == 4.0
