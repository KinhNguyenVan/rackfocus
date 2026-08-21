"""Contract browse: URL an toàn và mapping frame -> scene/timestamp."""
from types import SimpleNamespace

import pytest

from app.api import browse


def test_to_key_accepts_configured_bucket_only():
    helper = SimpleNamespace(bucket="aic-bucket-2026", region="ap-southeast-1")
    key = "Keyframes_L21_a/keyframes/L21_V001/000124.webp"

    assert browse._to_key(helper, key) == key
    assert browse._to_key(
        helper,
        f"https://aic-bucket-2026.s3.ap-southeast-1.amazonaws.com/{key}",
    ) == key

    with pytest.raises(ValueError, match="không thuộc bucket"):
        browse._to_key(helper, f"https://example.com/{key}")
    with pytest.raises(ValueError, match="Keyframe phải có dạng"):
        browse._to_key(helper, "snapshots/v1/manifest.json")


def test_frame_maps_to_scene_and_interpolated_timestamp():
    scenes = (
        {
            "scene_id": 0,
            "start_frame": 0,
            "end_frame": 30,
            "start_time": 0.0,
            "end_time": 1.0,
        },
        {
            "scene_id": 1,
            "start_frame": 31,
            "end_frame": 61,
            "start_time": 1.033,
            "end_time": 2.033,
        },
    )

    scene = browse._scene_for_frame(scenes, 46)
    assert scene is not None
    assert scene["scene_id"] == 1
    assert browse._frame_time(scene, 46) == pytest.approx(1.533)
    assert browse._scene_for_frame(scenes, 100) is None
