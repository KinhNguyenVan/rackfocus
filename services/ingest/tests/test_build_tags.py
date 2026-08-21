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

    tags = compute_tags(
        ["L21_V001", "L21_V001", "L21_V002"], [0, 1, 0], FakeRepository()
    )
    sports_id = next(i for i, d in enumerate(Domain) if d.id == "sports")
    education_id = next(i for i, d in enumerate(Domain) if d.id == "education")
    assert tags.dtype == np.uint16
    assert tags.tolist() == [sports_id, education_id, TAGS_SENTINEL]


def test_compute_tags_sentinel_for_video_or_scene_not_in_active_analysis() -> None:
    class FakeRepository:
        def active_domain_by_scene(self, video_ids):
            return {"L21_V001": {0: "sports"}}  # thiếu scene 1, thiếu cả video khác

    tags = compute_tags(
        ["L21_V001", "L21_V001", "L21_V999"], [0, 1, 0], FakeRepository()
    )
    assert tags[1] == TAGS_SENTINEL  # scene chưa được domain enrichment phủ tới
    assert tags[2] == TAGS_SENTINEL  # video chưa có active analysis nào


def test_compute_tags_rejects_domain_id_outside_current_taxonomy() -> None:
    class FakeRepository:
        def active_domain_by_scene(self, video_ids):
            return {"L21_V001": {0: "domain_da_bi_xoa_khoi_enum"}}

    with pytest.raises(ValueError, match="không nằm trong Domain enum"):
        compute_tags(["L21_V001"], [0], FakeRepository())


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
