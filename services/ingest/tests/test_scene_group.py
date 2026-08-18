"""Test ranh giới scene trên video mẫu ngắn.

Chỉ test phần logic thuần (không cần model/GPU): gom scene từ mảng xác suất ranh
giới, và gán script ASR cho scene theo overlap thời gian.
"""

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ingest.stages.asr import _normalize_segments, _to_seconds, assign_script_to_scenes
from ingest.stages.scene_group import boundaries_to_scenes


def _shots_df(shots):
    """shots: list[(start_frame, end_frame)] -> DataFrame giống shots.csv (fps=10)."""
    fps = 10.0
    rows = [
        {"shot_id": i, "start_frame": s, "end_frame": e,
         "start_ts": round(s / fps, 3), "end_ts": round(e / fps, 3)}
        for i, (s, e) in enumerate(shots)
    ]
    return pd.DataFrame(rows)


# --------------------------- boundaries_to_scenes ---------------------------
def test_single_shot_single_scene():
    df = _shots_df([(0, 30)])
    scenes = boundaries_to_scenes(df, [], threshold=0.55)
    assert len(scenes) == 1
    assert scenes[0]["start_frame"] == 0 and scenes[0]["end_frame"] == 30


def test_no_boundary_merges_all():
    df = _shots_df([(0, 30), (31, 60), (61, 90)])
    scenes = boundaries_to_scenes(df, [0.1, 0.2], threshold=0.55)
    assert len(scenes) == 1
    assert scenes[0]["start_frame"] == 0
    assert scenes[0]["end_frame"] == 90  # kéo tới hết shot cuối


def test_boundary_splits_scenes():
    df = _shots_df([(0, 30), (31, 60), (61, 90)])
    # ranh giới giữa shot0 và shot1 (prob=0.9), không giữa shot1 và shot2.
    scenes = boundaries_to_scenes(df, [0.9, 0.2], threshold=0.55)
    assert len(scenes) == 2
    assert (scenes[0]["start_frame"], scenes[0]["end_frame"]) == (0, 30)
    assert (scenes[1]["start_frame"], scenes[1]["end_frame"]) == (31, 90)


def test_scenes_are_contiguous_and_cover_all():
    df = _shots_df([(0, 30), (31, 60), (61, 90), (91, 120)])
    scenes = boundaries_to_scenes(df, [0.9, 0.1, 0.8], threshold=0.55)
    assert scenes[0]["start_frame"] == 0
    assert scenes[-1]["end_frame"] == 120


# --------------------------- assign_script_to_scenes ---------------------------
def test_assign_script_by_time_overlap():
    scenes = [
        {"start_time": 0.0, "end_time": 3.0},
        {"start_time": 3.0, "end_time": 9.0},
    ]
    segments = [
        {"start": 0.5, "end": 1.5, "text": "xin chào"},   # midpoint 1.0 -> scene 0
        {"start": 4.0, "end": 5.0, "text": "hôm nay"},     # midpoint 4.5 -> scene 1
        {"start": 7.0, "end": 8.0, "text": "trời đẹp"},    # midpoint 7.5 -> scene 1
    ]
    assign_script_to_scenes(scenes, segments)
    assert scenes[0]["script"] == "xin chào"
    assert scenes[1]["script"] == "hôm nay trời đẹp"


def test_assign_script_empty_when_no_segment():
    scenes = [{"start_time": 0.0, "end_time": 2.0}]
    assign_script_to_scenes(scenes, [])
    assert scenes[0]["script"] == ""


# --------------------------- timestamp parsing ---------------------------
@pytest.mark.parametrize("value,expected", [
    (12.5, 12.5),
    ("00:00:05:250", 5.25),   # HH:MM:SS:ms
    ("00:01:30.500", 90.5),   # HH:MM:SS.ms
    ("7.0", 7.0),
])
def test_to_seconds(value, expected):
    assert _to_seconds(value) == pytest.approx(expected)


def test_normalize_segments_from_string():
    segs = _normalize_segments("chỉ có text")
    assert segs == [{"start": 0.0, "end": 0.0, "text": "chỉ có text"}]
