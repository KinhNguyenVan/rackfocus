"""Fixture: core gRPC thật (snapshot nhỏ) + LLM mock + TestClient.

Chạy core THẬT qua Unix socket thay vì mock stub: phần lớn lỗi ở BE nằm đúng chỗ ghép
nối hai bên (sai message proto, map sai mã lỗi gRPC sang HTTP), mock stub sẽ giấu hết.

Mock hai thứ, vì cả hai đều là I/O ngoài và nặng:
  - LLM (litellm) — không gọi mạng trong test.
  - Text encoder — không tải bundle so400m 1.8GB.
"""

import hashlib
import json
import os
import sys
import tempfile
import types
from concurrent import futures

import grpc
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "core", "src"))

DIM = 32
NTAG = 5
N = 200

VOCAB = {
    0: "cảnh bóng đá trên sân",
    1: "phỏng vấn trong studio",
    2: "đường phố đông người",
    3: "món ăn cận cảnh",
    4: "phong cảnh thiên nhiên",
}


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _write_snapshot(out_dir):
    import faiss
    import pyarrow as pa
    import pyarrow.parquet as pq

    os.makedirs(out_dir, exist_ok=True)
    v = np.random.default_rng(5).standard_normal((N, DIM)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)

    index = faiss.index_factory(DIM, "HNSW32,SQ8", faiss.METRIC_INNER_PRODUCT)
    index.train(v)
    index.add(v)
    faiss.write_index(index, os.path.join(out_dir, "visual.faiss"))
    np.ascontiguousarray(v.astype(np.float16)).tofile(os.path.join(out_dir, "visual.f16"))
    np.save(os.path.join(out_dir, "idmap.npy"), (np.arange(N, dtype=np.int64) + 1) * 13)
    np.save(os.path.join(out_dir, "tags.npy"),
            np.array([i % NTAG for i in range(N)], dtype=np.uint16))
    with open(os.path.join(out_dir, "tag_vocab.json"), "w", encoding="utf-8") as f:
        json.dump({str(k): val for k, val in VOCAB.items()}, f, ensure_ascii=False)

    pq.write_table(pa.table({
        "video_name": [f"L26_V{i % 4:03d}" for i in range(N)],
        "frame": list(range(N)),
        "keyframe_time": [round(i * 0.2, 3) for i in range(N)],
        "scene_idx": [i // 10 for i in range(N)],
        "start_sec": [float((i // 10) * 2.0) for i in range(N)],
        "end_sec": [float((i // 10) * 2.0 + 2.0) for i in range(N)],
        "has_speech": [i % 3 == 0 for i in range(N)],
        "keyframe_key": [
            f"https://aic-bucket-2026.s3.ap-southeast-1.amazonaws.com/K/{i:06d}.webp"
            for i in range(N)],
        "clip_key": [
            f"https://aic-bucket-2026.s3.ap-southeast-1.amazonaws.com/S/{i // 10:03d}.mp4"
            for i in range(N)],
    }), os.path.join(out_dir, "payload.parquet"))
    np.zeros((N + 7) // 8, dtype=np.uint8).tofile(os.path.join(out_dir, "tombstone.bin"))

    files = ["visual.faiss", "visual.f16", "idmap.npy", "payload.parquet",
             "tombstone.bin", "tags.npy", "tag_vocab.json"]
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"version": "9", "dim": DIM, "metric": "cosine",
                   "encoder_name": "fake-enc", "count": N, "groups": ["L26_b"],
                   "checksums": {x: _sha256(os.path.join(out_dir, x)) for x in files}}, f)
    return out_dir


class FakeEncoder:
    name, dim = "fake-enc", DIM

    def encode(self, texts):
        out = []
        for t in texts:
            v = np.random.default_rng(abs(hash(t)) % 2**32).standard_normal(DIM)
            out.append((v / np.linalg.norm(v)).astype(np.float32))
        return np.asarray(out)

    def encode_one(self, text):
        return self.encode([text])[0]


@pytest.fixture
def llm():
    """Mock litellm. `llm.reply` để đổi hành vi, `llm.calls` để đếm.

    `reply` không khai "confidence" thì mock tự điền 1.0, tức "LLM rất tự tin" — để test
    KHÔNG nói về confidence giữ nguyên ý định của nó. Muốn test cổng confidence thì khai
    thẳng `{"confidence": 0.4, ...}`.

    Lưu ý: đây CHỈ là mặc định của mock. Hợp đồng thật ngược lại — model không khai
    confidence thì coi như 0.0 (không tin), xem `enrich._confidence` và
    test_taxonomy.py::test_confidence_thieu_hoac_rac_thi_coi_nhu_khong_tin.
    """
    state = types.SimpleNamespace(calls=0, last_prompt=None, reply=None, raises=None)

    async def acompletion(**kwargs):
        state.calls += 1
        state.last_prompt = kwargs["messages"][1]["content"]
        if state.raises:
            raise state.raises
        payload = dict(state.reply) if state.reply is not None else {"tags": [], "enriched": ""}
        payload.setdefault("confidence", 1.0)
        msg = types.SimpleNamespace(content=json.dumps(payload))
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])

    sys.modules["litellm"] = types.SimpleNamespace(acompletion=acompletion)
    yield state
    sys.modules.pop("litellm", None)


@pytest.fixture
def client(tmp_path, llm, monkeypatch):
    from fastapi.testclient import TestClient
    from searchcore.config import Config
    from searchcore.holder import IndexHolder
    from searchcore.pb.searchcore.v1 import admin_pb2_grpc as apb_grpc
    from searchcore.pb.searchcore.v1 import search_pb2_grpc as pb_grpc
    from searchcore.server import AdminServiceServicer, SearchCoreServiceServicer
    from searchcore.snapshot import Snapshot

    snap = Snapshot(_write_snapshot(str(tmp_path / "snap")))
    holder = IndexHolder()
    holder.swap(snap)

    sock_path = os.path.join(tempfile.mkdtemp(), "sc.sock")
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    cfg = Config()
    pb_grpc.add_SearchCoreServiceServicer_to_server(
        SearchCoreServiceServicer(holder, FakeEncoder(), cfg), server)
    apb_grpc.add_AdminServiceServicer_to_server(
        AdminServiceServicer(holder, FakeEncoder(), cfg), server)
    server.add_insecure_port(f"unix://{sock_path}")
    server.start()

    monkeypatch.setenv("SEARCHCORE_TARGET", f"unix://{sock_path}")
    monkeypatch.setenv("DEFAULT_TOP_K", "10")
    monkeypatch.setenv("DIVERSITY_MAX_PER_VIDEO", "0")

    from app.config import get_settings
    from app.services import tagvocab

    get_settings.cache_clear()
    # tagvocab cache là module-level -> reset giữa các test, nếu không test sau sẽ dùng
    # vocab của socket đã đóng.
    tagvocab._vocab, tagvocab._version, tagvocab._checked_ver = {}, "", ""

    from app.main import app

    with TestClient(app) as c:
        yield c

    server.stop(0)
    get_settings.cache_clear()
