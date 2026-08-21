"""Fixture: snapshot thật (nhỏ) để test không cần corpus 500k.

Dựng đúng 7 file mà notebook build-index sinh ra (`aic-embed-siglip-2026.ipynb`), kể cả
checksum trong manifest — nếu không thì mọi kiểm tra fail-closed của `snapshot.py` sẽ bị
bỏ qua và test hoá ra chẳng kiểm gì.

Snapshot là hợp đồng offline↔online thứ hai của hệ thống, ngang `.proto`. `.proto` có
`buf breaking` canh trong CI; snapshot thì chỉ có mấy test này.
"""

import hashlib
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

DIM = 32
NTAG = 5
N = 200
ENCODER_NAME = "siglip-test"

SNAPSHOT_FILES = ("visual.faiss", "visual.f16", "idmap.npy", "payload.parquet",
                  "tombstone.bin", "tags.npy", "tag_vocab.json")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def unit_vectors(n=N, dim=DIM, seed=0, normalize=True):
    v = np.random.default_rng(seed).standard_normal((n, dim)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True) if normalize else v * 15.0


def write_snapshot(out_dir, *, vectors=None, tags=None, vocab=None, count=None,
                   dim=None, encoder_name=ENCODER_NAME, tombstone_bytes=None,
                   with_tags=True, checksum_files=None, version="1"):
    """Ghi snapshot hợp lệ. Mọi tham số để None = giá trị đúng; truyền vào để tạo bản hỏng."""
    import faiss
    import pyarrow as pa
    import pyarrow.parquet as pq

    os.makedirs(out_dir, exist_ok=True)
    v = unit_vectors() if vectors is None else vectors
    n, d = v.shape

    index = faiss.index_factory(d, "HNSW32,SQ8", faiss.METRIC_INNER_PRODUCT)
    index.train(v)
    index.add(v)
    faiss.write_index(index, os.path.join(out_dir, "visual.faiss"))

    np.ascontiguousarray(v.astype(np.float16)).tofile(os.path.join(out_dir, "visual.f16"))
    np.save(os.path.join(out_dir, "idmap.npy"), (np.arange(n, dtype=np.int64) + 1) * 7919)

    pq.write_table(pa.table({
        "video_name": [f"L26_V{i % 4:03d}" for i in range(n)],
        "frame": list(range(n)),
        "keyframe_time": [round(i * 0.2, 3) for i in range(n)],
        "scene_idx": [i // 10 for i in range(n)],
        "start_sec": [float((i // 10) * 2.0) for i in range(n)],
        "end_sec": [float((i // 10) * 2.0 + 2.0) for i in range(n)],
        "has_speech": [i % 3 == 0 for i in range(n)],
        "keyframe_key": [f"https://b.s3.r.amazonaws.com/K/{i:06d}.webp" for i in range(n)],
        "clip_key": [f"https://b.s3.r.amazonaws.com/S/{i // 10:03d}.mp4" for i in range(n)],
    }), os.path.join(out_dir, "payload.parquet"))

    nbytes = (n + 7) // 8 if tombstone_bytes is None else tombstone_bytes
    np.zeros(nbytes, dtype=np.uint8).tofile(os.path.join(out_dir, "tombstone.bin"))

    if with_tags:
        t = np.array([i % NTAG for i in range(n)], dtype=np.uint16) if tags is None else tags
        np.save(os.path.join(out_dir, "tags.npy"), t)
        vocab = vocab if vocab is not None else {
            str(i): f"mô tả tag {i}" for i in range(NTAG)}
        with open(os.path.join(out_dir, "tag_vocab.json"), "w", encoding="utf-8") as f:
            json.dump(vocab, f, ensure_ascii=False)

    files = checksum_files if checksum_files is not None else [
        f for f in SNAPSHOT_FILES if os.path.exists(os.path.join(out_dir, f))]
    manifest = {
        "version": version,
        "dim": d if dim is None else dim,
        "metric": "cosine",
        "encoder_name": encoder_name,
        "count": n if count is None else count,
        "groups": ["L26_b"],
        "checksums": {f: sha256(os.path.join(out_dir, f)) for f in files},
    }
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f)
    return out_dir


@pytest.fixture
def snapshot_dir(tmp_path):
    """Snapshot hợp lệ, checksum đầy đủ."""
    return write_snapshot(str(tmp_path / "snap"))


@pytest.fixture
def snap(snapshot_dir):
    from searchcore.snapshot import Snapshot

    return Snapshot(snapshot_dir, dim=DIM, encoder_name=ENCODER_NAME)


@pytest.fixture
def broken(tmp_path):
    """Factory dựng snapshot hỏng theo từng kiểu: broken(**kwargs) -> path."""
    counter = {"n": 0}

    def _make(**kwargs):
        counter["n"] += 1
        return write_snapshot(str(tmp_path / f"broken{counter['n']}"), **kwargs)

    return _make
