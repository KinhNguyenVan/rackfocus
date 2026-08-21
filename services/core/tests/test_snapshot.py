"""Test manifest validation: mọi kiểm tra phải FAIL-CLOSED.

Lý do các test này tồn tại: các chế độ hỏng của snapshot đều **âm thầm**. Trộn
`visual.faiss` của build #2 với `visual.f16` của build #1 cho mọi điểm là dot product với
vector của frame khác, mà mọi check kích thước/độ dài vẫn pass. Không có test thì không có
gì phát hiện được.

Xem docs/search-design.md §2.
"""

import json
import os

import numpy as np
import pytest
from conftest import DIM, ENCODER_NAME, NTAG, N, unit_vectors
from searchcore.snapshot import TAGS_SENTINEL, Snapshot, SnapshotError


def load(path, **kw):
    kw.setdefault("dim", DIM)
    kw.setdefault("encoder_name", ENCODER_NAME)
    return Snapshot(path, **kw)


# --------------------------- snapshot hợp lệ ---------------------------
def test_valid_snapshot_loads(snapshot_dir):
    snap = load(snapshot_dir)
    assert snap.count == N
    assert snap.dim == DIM
    assert len(snap.tag_vocab) == NTAG
    assert snap.unassigned_count == 0
    assert snap.tag_counts.sum() == N


def test_tags_partition_corpus_without_overlap(snap):
    """Mỗi frame đúng 1 tag -> bucket không giao nhau, candidate là hợp không trùng lặp.

    Toàn bộ thiết kế filter dựa vào tính chất này (docs/search-design.md §4).
    """
    rows = snap.rows_for_tags([0, 2])
    assert len(rows) == len(set(rows.tolist()))
    assert set(snap.tags[rows].tolist()) == {0, 2}
    assert len(rows) == snap.tag_counts[0] + snap.tag_counts[2]


def test_unknown_tag_raises_value_error_not_index_error(snap):
    """LLM ảo giác tag id -> lỗi của caller (400), không phải IndexError (500)."""
    with pytest.raises(ValueError, match="ngoài phạm vi"):
        snap.rows_for_tags([9999])


def test_alive_mask_reads_tombstone_lsb_first(snap):
    rows = np.arange(16, dtype=np.int64)
    assert snap.alive_mask(rows).all()

    tomb = np.zeros((N + 7) // 8, dtype=np.uint8)
    for r in (0, 3, 9):
        tomb[r >> 3] |= np.uint8(1 << (r & 7))
    snap.tombstone = tomb
    mask = snap.alive_mask(rows)
    assert not mask[0] and not mask[3] and not mask[9]
    assert mask[1] and mask[2] and mask[8] and mask[10]


# --------------------------- hợp đồng offline↔online ---------------------------
def test_rejects_dim_mismatch(snapshot_dir):
    with pytest.raises(SnapshotError, match="dim lệch"):
        load(snapshot_dir, dim=999)


def test_rejects_encoder_mismatch(snapshot_dir):
    with pytest.raises(SnapshotError, match="encoder lệch"):
        load(snapshot_dir, encoder_name="clip-vit-b32")


def test_dim_zero_means_skip_check(snapshot_dir):
    """VECTOR_DIM=0 = 'lấy theo manifest'. Dùng khi chưa chốt được dim thật."""
    assert load(snapshot_dir, dim=0).dim == DIM


# --------------------------- integrity ---------------------------
def test_rejects_truncated_refine_store(snapshot_dir):
    with open(os.path.join(snapshot_dir, "visual.f16"), "r+b") as f:
        f.truncate(N * DIM * 2 - 64)
    with pytest.raises(SnapshotError):
        load(snapshot_dir)


def test_rejects_swapped_refine_store_with_correct_size(snapshot_dir):
    """`visual.f16` KHÔNG có header -> size check pass với bất kỳ file cùng độ dài.

    Đây chính là ca mà checksum là phòng vệ duy nhất: hai build cùng N và dim cho file
    dài y hệt nhau.
    """
    other = unit_vectors(seed=99).astype(np.float16)
    np.ascontiguousarray(other).tofile(os.path.join(snapshot_dir, "visual.f16"))
    with pytest.raises(SnapshotError, match="checksum"):
        load(snapshot_dir)


def test_size_check_catches_mismatch_when_checksums_disabled(broken):
    """Tắt checksum thì size check vẫn phải bắt được count sai."""
    path = broken(count=N - 5)
    with pytest.raises(SnapshotError, match="sai kích thước"):
        load(path, verify_checksums=False)


def test_rejects_truncated_tombstone(snapshot_dir):
    with open(os.path.join(snapshot_dir, "tombstone.bin"), "r+b") as f:
        f.truncate(3)
    with pytest.raises(SnapshotError):
        load(snapshot_dir)


def test_rejects_missing_manifest(snapshot_dir):
    os.remove(os.path.join(snapshot_dir, "manifest.json"))
    with pytest.raises(SnapshotError, match="Thiếu manifest"):
        load(snapshot_dir)


def test_rejects_unnormalized_vectors(broken):
    """manifest ghi metric='cosine' nhưng index dùng INNER_PRODUCT — chỉ tương đương khi
    vector đã L2-normalize.

    Group nào embed bằng bundle ONNX chưa bake L2 có norm ~15 và **thống trị mọi query**
    bất kể nội dung. Phải kiểm toàn bộ, không lấy mẫu.
    """
    path = broken(vectors=unit_vectors(normalize=False))
    with pytest.raises(SnapshotError, match="không L2-normalized"):
        load(path)


def test_detects_few_bad_vectors_among_many_good(broken):
    """Lấy mẫu sẽ bỏ sót vài chục row hỏng trên 200 — nên phải quét hết."""
    v = unit_vectors()
    v[137] *= 12.0
    path = broken(vectors=v)
    with pytest.raises(SnapshotError, match="1/200 vector"):
        load(path)


# --------------------------- tags ---------------------------
def test_rejects_tag_outside_vocab(broken):
    path = broken(tags=np.full(N, 99, dtype=np.uint16))
    with pytest.raises(SnapshotError, match="không nằm trong"):
        load(path)


def test_rejects_wrong_tags_dtype(broken):
    path = broken(tags=np.zeros(N, dtype=np.int32))
    with pytest.raises(SnapshotError, match="phải là uint16"):
        load(path)


def test_sentinel_is_65535_not_255(broken):
    """255 là tag HỢP LỆ khi vocab 100-500. Sentinel uint16 phải là 65535."""
    assert TAGS_SENTINEL == 65535

    tags = np.full(N, TAGS_SENTINEL, dtype=np.uint16)
    tags[:20] = 3
    path = broken(tags=tags)
    snap = load(path)
    assert snap.unassigned_count == N - 20
    assert len(snap.rows_for_tags([3])) == 20


def test_missing_tags_file_still_loads(broken):
    """Chưa có dữ liệu tag -> đường search toàn bộ vẫn chạy được."""
    path = broken(with_tags=False)
    snap = load(path)
    assert snap.unassigned_count == snap.count
    assert len(snap.rows_for_tags([])) == 0


@pytest.mark.parametrize("vocab,expect", [
    ({str(i): f"mô tả {i}" for i in range(NTAG)}, "mô tả 0"),
    ({str(i): {"name": f"t{i}", "description": f"mô tả {i}"} for i in range(NTAG)}, "mô tả 0"),
])
def test_accepts_both_vocab_formats(broken, vocab, expect):
    """Định dạng chính là {"tag_id": "description"} — phẳng, để nhồi thẳng vào prompt LLM."""
    snap = load(broken(vocab=vocab))
    assert snap.tag_vocab[0]["description"] == expect


@pytest.mark.parametrize("vocab,match", [
    ({"abc": "x"}, "không phải số"),
    ({"0": 123}, "phải là string hoặc object"),
])
def test_rejects_malformed_vocab(broken, vocab, match):
    with pytest.raises(SnapshotError, match=match):
        load(broken(vocab=vocab))


def test_warns_when_tag_files_outside_checksums(broken, caplog):
    """tags.npy do một pass khác sinh SAU khi manifest seal -> có thể ngoài checksums.

    Không chặn (chưa có dữ liệu tag thật), nhưng phải cảnh báo: không có gì bảo đảm
    tags.npy được sinh cho đúng bản snapshot này, mà lệch row order là sai toàn bộ.
    """
    path = broken(checksum_files=["visual.faiss", "visual.f16", "idmap.npy",
                                  "payload.parquet", "tombstone.bin"])
    load(path)
    warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert any("tags.npy" in m and "checksums" in m for m in warnings), warnings


def test_manifest_records_provenance(snapshot_dir):
    """Row order = sorted(glob(...)) và không ghi ở đâu -> tags phải join qua point_id."""
    with open(os.path.join(snapshot_dir, "manifest.json"), encoding="utf-8") as f:
        man = json.load(f)
    assert man["groups"] == ["L26_b"]
