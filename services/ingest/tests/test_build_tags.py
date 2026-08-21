"""Pure tests for tag=domain_id assignment (build_tags.py)."""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import ingest.build_tags as bt
from ingest.build_tags import (
    TAGS_SENTINEL,
    _print_coverage,
    _update_manifest_checksums,
    _upload_to_s3,
    build_tag_vocab,
    compute_tags,
    load_frame_to_scene,
)
from ingest.domain.models import Domain


def test_tag_vocab_covers_every_domain_with_stable_ids() -> None:
    vocab = build_tag_vocab()
    assert len(vocab) == len(list(Domain))
    assert vocab["0"] == {"name": "politics_society", "description": "Chính trị - Xã hội"}
    # id ổn định = thứ tự định nghĩa Domain, không phải alphabet.
    assert vocab[str(len(vocab) - 1)]["name"] == "general_news"


def test_compute_tags_maps_known_scene_to_its_domain_tag_id() -> None:
    class FakeRepository:
        def active_domain_by_scene(self, video_ids):
            assert sorted(video_ids) == ["L21_V001", "L21_V002"]
            return {"L21_V001": {0: "sports", 1: "education"}}

    frame_to_scene = {("L21_V001", 12): 0, ("L21_V001", 45): 1, ("L21_V002", 3): 0}
    tags = compute_tags(
        ["L21_V001", "L21_V001", "L21_V002"], [12, 45, 3], frame_to_scene, FakeRepository()
    )
    sports_id = next(i for i, d in enumerate(Domain) if d.id == "sports")
    education_id = next(i for i, d in enumerate(Domain) if d.id == "education")
    assert tags.dtype == np.uint16
    assert tags.tolist() == [sports_id, education_id, TAGS_SENTINEL]


def test_compute_tags_sentinel_when_frame_missing_from_map_or_scene_unassigned() -> None:
    class FakeRepository:
        def active_domain_by_scene(self, video_ids):
            return {"L21_V001": {0: "sports"}}  # thiếu scene 1, thiếu cả video khác

    # frame=45 KHÔNG có trong frame_to_scene (video chưa được AIC_KeyframeSceneMap xử lý
    # tới) -- khác với case "có scene nhưng domain enrichment chưa phủ".
    frame_to_scene = {("L21_V001", 12): 0, ("L21_V999", 3): 0}
    tags = compute_tags(
        ["L21_V001", "L21_V001", "L21_V999"], [12, 45, 3], frame_to_scene, FakeRepository()
    )
    assert tags[1] == TAGS_SENTINEL  # frame không có trong map keyframe->scene
    assert tags[2] == TAGS_SENTINEL  # có scene nhưng video chưa có active analysis nào


def test_compute_tags_rejects_domain_id_outside_current_taxonomy() -> None:
    class FakeRepository:
        def active_domain_by_scene(self, video_ids):
            return {"L21_V001": {0: "domain_da_bi_xoa_khoi_enum"}}

    with pytest.raises(ValueError, match="không nằm trong Domain enum"):
        compute_tags(["L21_V001"], [12], {("L21_V001", 12): 0}, FakeRepository())


def _write_map_doc(base_dir: str, group: str, videos: dict) -> None:
    path = os.path.join(base_dir, f"Maps_{group}", "maps", "keyframe_scene_map.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"groups": [group], "videos": videos}, f)


def test_load_frame_to_scene_reads_keyframe_scene_map_and_skips_unmatched(tmp_path) -> None:
    _write_map_doc(str(tmp_path), "L21_a", {
        "L21_V001": {"keyframes": [
            {"frame": 12, "scene_id": 0},
            {"frame": 45, "scene_id": 1},
            {"frame": 99, "scene_id": None},  # notebook: video không scene nào khớp
        ]},
    })

    frame_to_scene = load_frame_to_scene(str(tmp_path))

    assert frame_to_scene == {("L21_V001", 12): 0, ("L21_V001", 45): 1}


def test_load_frame_to_scene_merges_multiple_groups(tmp_path) -> None:
    _write_map_doc(str(tmp_path), "L21_a", {"L21_V001": {"keyframes": [{"frame": 1, "scene_id": 0}]}})
    _write_map_doc(str(tmp_path), "L22_a", {"L22_V001": {"keyframes": [{"frame": 2, "scene_id": 0}]}})

    frame_to_scene = load_frame_to_scene(str(tmp_path))

    assert frame_to_scene == {("L21_V001", 1): 0, ("L22_V001", 2): 0}


def test_load_frame_to_scene_groups_filter_restricts_which_files_are_read(tmp_path) -> None:
    _write_map_doc(str(tmp_path), "L21_a", {"L21_V001": {"keyframes": [{"frame": 1, "scene_id": 0}]}})
    _write_map_doc(str(tmp_path), "L22_a", {"L22_V001": {"keyframes": [{"frame": 2, "scene_id": 0}]}})

    frame_to_scene = load_frame_to_scene(str(tmp_path), groups=["L21_a"])

    assert frame_to_scene == {("L21_V001", 1): 0}


def test_load_frame_to_scene_raises_when_nothing_found(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="Không thấy"):
        load_frame_to_scene(str(tmp_path))


def test_manifest_checksums_updated_for_written_files_and_preserves_existing(
    tmp_path,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps({"count": 1, "checksums": {"idmap.npy": "deadbeef"}}),
        encoding="utf-8",
    )
    (tmp_path / "tags.npy").write_bytes(b"\x00\x01")
    (tmp_path / "tag_vocab.json").write_text("{}", encoding="utf-8")

    _update_manifest_checksums(str(tmp_path), ["tags.npy", "tag_vocab.json"])

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["checksums"]["idmap.npy"] == "deadbeef"  # không bị đè
    assert set(manifest["checksums"]) == {"idmap.npy", "tags.npy", "tag_vocab.json"}
    assert len(manifest["checksums"]["tags.npy"]) == 64  # sha256 hex


def test_print_coverage_reports_assigned_fraction(capsys) -> None:
    sports_id = next(i for i, d in enumerate(Domain) if d.id == "sports")
    tags = np.array([sports_id, sports_id, TAGS_SENTINEL], dtype=np.uint16)
    _print_coverage(tags)
    out = capsys.readouterr().out
    assert "2/3" in out
    assert "sports" in out


def test_upload_to_s3_uses_bucket_from_uri_not_aws_bucket_name_env(
    monkeypatch, tmp_path
) -> None:
    """SNAPSHOT_S3 có thể trỏ bucket khác AWS_BUCKET_NAME -> phải lấy bucket từ chính URI
    (giống storage.upload_dir), không phải qua storage.upload_file (khoá cứng env)."""
    calls: list[tuple[str, str, str]] = []

    class FakeClient:
        def upload_file(self, local_path, bucket, key):
            calls.append((os.path.basename(local_path), bucket, key))

    monkeypatch.setattr(bt.storage, "get_client", lambda: FakeClient())
    (tmp_path / "tags.npy").write_bytes(b"x")
    (tmp_path / "tag_vocab.json").write_text("{}", encoding="utf-8")

    _upload_to_s3(
        str(tmp_path), "s3://other-bucket/snapshots/v1", ["tags.npy", "tag_vocab.json"]
    )

    assert calls == [
        ("tags.npy", "other-bucket", "snapshots/v1/tags.npy"),
        ("tag_vocab.json", "other-bucket", "snapshots/v1/tag_vocab.json"),
    ]
