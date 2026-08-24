"""Test servicer gRPC qua Unix socket THẬT (không gọi hàm trực tiếp).

Đi qua socket là có chủ ý: nó bắt được lỗi tầng hợp đồng mà gọi hàm trực tiếp bỏ sót —
ví dụ dùng nhầm message ở sai file proto (`SearchHit` nằm trong common.proto chứ không
phải search.proto), hoặc map sai mã lỗi gRPC.
"""

import tempfile
from concurrent import futures

import grpc
import numpy as np
import pytest
from conftest import DIM, NTAG, N
from searchcore.config import Config
from searchcore.holder import IndexHolder
from searchcore.pb.searchcore.v1 import admin_pb2 as apb
from searchcore.pb.searchcore.v1 import admin_pb2_grpc as apb_grpc
from searchcore.pb.searchcore.v1 import common_pb2 as cpb
from searchcore.pb.searchcore.v1 import search_pb2 as pb
from searchcore.pb.searchcore.v1 import search_pb2_grpc as pb_grpc
from searchcore.server import AdminServiceServicer, SearchCoreServiceServicer


class FakeEncoder:
    """Encoder tất định, không cần tải 1.8GB so400m."""

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
def stubs(snap):
    holder = IndexHolder()
    holder.swap(snap)
    cfg = Config()
    sock = f"unix://{tempfile.mkdtemp()}/sc.sock"

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    pb_grpc.add_SearchCoreServiceServicer_to_server(
        SearchCoreServiceServicer(holder, FakeEncoder(), cfg), server)
    apb_grpc.add_AdminServiceServicer_to_server(
        AdminServiceServicer(holder, FakeEncoder(), cfg), server)
    server.add_insecure_port(sock)
    server.start()

    channel = grpc.insecure_channel(sock)
    yield pb_grpc.SearchCoreServiceStub(channel), apb_grpc.AdminServiceStub(channel), snap
    channel.close()
    server.stop(0)


@pytest.fixture
def stubs_small_gap(snap):
    """Như `stubs` nhưng TRAKE_MIN_GAP_SEC nhỏ để test được với dt=1.6s của
    fixture snap có sẵn (video cách nhau keyframe_time=0.2s/row), và
    candidates_per_event=1 để chỉ có đúng 1 cặp (kiểm tra pooling/cap đã có
    ở test_temporal.py rồi, RPC test này chỉ cần xác nhận đường ống gRPC)."""
    holder = IndexHolder()
    holder.swap(snap)
    cfg = Config(trake_min_gap_sec=0.1, trake_max_gap_sec=5.0,
                trake_candidates_per_event=1)
    sock = f"unix://{tempfile.mkdtemp()}/sc.sock"

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    pb_grpc.add_SearchCoreServiceServicer_to_server(
        SearchCoreServiceServicer(holder, FakeEncoder(), cfg), server)
    server.add_insecure_port(sock)
    server.start()

    channel = grpc.insecure_channel(sock)
    yield pb_grpc.SearchCoreServiceStub(channel), snap
    channel.close()
    server.stop(0)


def _q(snap, row=0):
    return cpb.Vector(values=np.asarray(snap.refine[row], dtype=np.float32).tolist())


# --------------------------- Admin ---------------------------
def test_health_reports_snapshot(stubs):
    _, admin, snap = stubs
    h = admin.Health(apb.HealthRequest())
    assert h.ready
    assert h.snapshot_ver == snap.version
    assert h.point_count == N
    assert h.encoder_name == "fake-enc"


def test_get_tag_vocab_returns_counts(stubs):
    _, admin, _ = stubs
    v = admin.GetTagVocab(apb.GetTagVocabRequest())
    assert len(v.tags) == NTAG
    assert v.tags[0].description.startswith("mô tả")
    # point_count để BE/LLM biết tag nào quá rộng (chọn nó thì không thu hẹp được gì).
    assert v.tags[0].point_count == N // NTAG
    assert v.unassigned_count == 0


def test_health_unavailable_before_snapshot_loaded():
    """Core mất vài phút tải + validate snapshot; trong khoảng đó phải trả UNAVAILABLE
    chứ không phải kết quả rỗng — FE cần phân biệt được hai thứ."""
    holder = IndexHolder()
    sock = f"unix://{tempfile.mkdtemp()}/sc.sock"
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    pb_grpc.add_SearchCoreServiceServicer_to_server(
        SearchCoreServiceServicer(holder, None, Config()), server)
    apb_grpc.add_AdminServiceServicer_to_server(
        AdminServiceServicer(holder, None, Config()), server)
    server.add_insecure_port(sock)
    server.start()
    try:
        ch = grpc.insecure_channel(sock)
        assert not apb_grpc.AdminServiceStub(ch).Health(apb.HealthRequest()).ready
        with pytest.raises(grpc.RpcError) as ex:
            pb_grpc.SearchCoreServiceStub(ch).Search(
                pb.SearchRequest(query=[pb.QueryPart(text="x")], top_k=5))
        assert ex.value.code() == grpc.StatusCode.UNAVAILABLE
        ch.close()
    finally:
        server.stop(0)


# --------------------------- Search ---------------------------
def test_search_returns_hits_with_payload(stubs):
    search, _, snap = stubs
    r = search.Search(pb.SearchRequest(
        query=[pb.QueryPart(vector=_q(snap))], top_k=5, with_payload=True))

    assert len(r.hits) == 5
    assert r.hits[0].index_row == 0
    assert r.hits[0].id == int(snap.idmap[0])
    assert r.meta.snapshot_ver == snap.version

    p = r.hits[0].payload
    # Ba field này TRƯỚC ĐÂY không có trong proto: thiếu video_name thì
    # Diversity.max_per_video group theo 0; thiếu frame thì không nộp bài được;
    # thiếu keyframe_time thì player seek theo start_sec của SCENE, lệch hàng chục giây.
    assert p.video_name == "L26_V000"
    assert p.frame == 0
    assert p.keyframe_time == pytest.approx(0.0)
    assert p.keyframe_key.startswith("https://")


def test_search_echoes_tags_used_and_strategy(stubs):
    """Không trả tags_used thì user không biết mình vừa search cả kho hay 1/5 kho."""
    search, _, snap = stubs
    r = search.Search(pb.SearchRequest(
        query=[pb.QueryPart(vector=_q(snap))], top_k=5,
        filter=cpb.Filter(tags=[1, 3])))

    assert list(r.meta.tags_used) == [1, 3]
    assert r.meta.timings.filter_matched == 2 * (N // NTAG)
    assert r.meta.timings.filter_strategy_used == 3  # EXACT_SUBSET
    assert all(snap.tags[h.index_row] in (1, 3) for h in r.hits)


def test_search_without_tags_reports_full_corpus(stubs):
    search, _, snap = stubs
    r = search.Search(pb.SearchRequest(
        query=[pb.QueryPart(vector=_q(snap))], top_k=5))
    assert list(r.meta.tags_used) == []
    assert r.meta.timings.filter_matched == N


def test_search_by_text_uses_encoder(stubs):
    search, _, _ = stubs
    r = search.Search(pb.SearchRequest(
        query=[pb.QueryPart(text="hai người đang nói chuyện")], top_k=3))
    assert len(r.hits) == 3
    assert r.meta.timings.encode_ms >= 0


def test_unknown_tag_is_invalid_argument_not_internal(stubs):
    search, _, snap = stubs
    with pytest.raises(grpc.RpcError) as ex:
        search.Search(pb.SearchRequest(
            query=[pb.QueryPart(vector=_q(snap))], top_k=5,
            filter=cpb.Filter(tags=[9999])))
    assert ex.value.code() == grpc.StatusCode.INVALID_ARGUMENT


def test_wrong_vector_dim_is_invalid_argument(stubs):
    search, _, _ = stubs
    with pytest.raises(grpc.RpcError) as ex:
        search.Search(pb.SearchRequest(
            query=[pb.QueryPart(vector=cpb.Vector(values=[0.0] * (DIM + 1)))], top_k=5))
    assert ex.value.code() == grpc.StatusCode.INVALID_ARGUMENT


def test_diversity_caps_per_video(stubs):
    from collections import Counter

    search, _, snap = stubs
    r = search.Search(pb.SearchRequest(
        query=[pb.QueryPart(vector=_q(snap))], top_k=12, with_payload=True,
        diversity=cpb.Diversity(max_per_video=2)))
    names = Counter(h.payload.video_name for h in r.hits)
    assert max(names.values()) <= 2


def test_encode_rpc_returns_normalized_vector(stubs):
    search, _, _ = stubs
    r = search.Encode(pb.EncodeRequest(text="xe chạy trên đường", normalize=True))
    v = np.asarray(r.vector.values, dtype=np.float32)
    assert v.size == DIM
    assert np.linalg.norm(v) == pytest.approx(1.0, abs=1e-3)


def test_stats_counts_only_successful_queries(stubs):
    search, admin, snap = stubs
    search.Search(pb.SearchRequest(query=[pb.QueryPart(vector=_q(snap))], top_k=3))
    with pytest.raises(grpc.RpcError):
        search.Search(pb.SearchRequest(
            query=[pb.QueryPart(vector=_q(snap))], top_k=3,
            filter=cpb.Filter(tags=[9999])))

    st = admin.Stats(apb.StatsRequest(reset=True))
    assert st.queries_total >= 1


# --------------------------- SearchTemporal ---------------------------
def test_search_temporal_returns_chain_with_two_hits(stubs_small_gap):
    search, snap = stubs_small_gap
    r = search.SearchTemporal(pb.SearchTemporalRequest(
        events=[
            pb.TemporalEvent(query=[pb.QueryPart(vector=_q(snap, 4))]),
            pb.TemporalEvent(query=[pb.QueryPart(vector=_q(snap, 12))]),
        ],
        with_payload=True))

    assert len(r.chains) == 1
    c = r.chains[0]
    assert len(c.hits) == 2
    assert c.hits[0].payload.video_name == "L26_V000"
    assert c.hits[0].index_row == 4
    assert c.hits[1].index_row == 12
    assert c.span_sec == pytest.approx(1.6, abs=1e-6)
    assert c.score > 0


def test_search_temporal_rejects_wrong_event_count(stubs):
    search, _, snap = stubs
    with pytest.raises(grpc.RpcError) as ex:
        search.SearchTemporal(pb.SearchTemporalRequest(
            events=[pb.TemporalEvent(query=[pb.QueryPart(vector=_q(snap, 0))])]))
    assert ex.value.code() == grpc.StatusCode.INVALID_ARGUMENT


def test_search_temporal_unavailable_before_snapshot_loaded():
    holder = IndexHolder()
    sock = f"unix://{tempfile.mkdtemp()}/sc.sock"
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    pb_grpc.add_SearchCoreServiceServicer_to_server(
        SearchCoreServiceServicer(holder, None, Config()), server)
    server.add_insecure_port(sock)
    server.start()
    try:
        ch = grpc.insecure_channel(sock)
        with pytest.raises(grpc.RpcError) as ex:
            pb_grpc.SearchCoreServiceStub(ch).SearchTemporal(
                pb.SearchTemporalRequest(events=[
                    pb.TemporalEvent(query=[pb.QueryPart(text="x")]),
                    pb.TemporalEvent(query=[pb.QueryPart(text="y")]),
                ]))
        assert ex.value.code() == grpc.StatusCode.UNAVAILABLE
        ch.close()
    finally:
        server.stop(0)
