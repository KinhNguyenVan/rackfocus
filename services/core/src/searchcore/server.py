"""Servicer gRPC: SearchCoreService (Search, Encode) + AdminService (Health, GetTagVocab...).

Map request proto -> hàm search, đo Timings, và **trả lại thông tin lọc**: `tags_used` +
`filter_matched` + `filter_strategy_used`. Không trả thì user không có cách nào biết mình
vừa search 500 giờ video hay 8 giờ — xem docs/search-design.md §4.
"""
from __future__ import annotations

import logging
import time

import grpc
import numpy as np

from . import search as S
from . import temporal as T
from .metrics import metrics
from .pb.searchcore.v1 import admin_pb2 as apb
from .pb.searchcore.v1 import admin_pb2_grpc as apb_grpc
from .pb.searchcore.v1 import common_pb2 as cpb
from .pb.searchcore.v1 import search_pb2 as pb
from .pb.searchcore.v1 import search_pb2_grpc as pb_grpc

log = logging.getLogger("searchcore.server")

TIER_KEYFRAME = 2

# Cột payload -> field của Payload proto. Cột thiếu thì bỏ qua (payload có thể tiến hoá).
_PAYLOAD_STR = ("video_name", "keyframe_key", "clip_key")
_PAYLOAD_NUM = ("frame", "scene_idx", "shot_id")
_PAYLOAD_F64 = ("keyframe_time", "start_sec", "end_sec")
_PAYLOAD_BOOL = ("has_ocr", "has_speech")


def _build_payloads(snap, rows: np.ndarray) -> list[cpb.Payload]:
    """Gather payload theo row, zero-copy columnar.

    KHÔNG dùng to_pylist() trên cả bảng: payload dạng Arrow là 268 B/row, dạng dict
    Python là ~1300 B/row (đo được 9.2x trên chính repo này ở cột vector). Chỉ `.take()`
    đúng số row của trang kết quả.
    """
    if rows.size == 0:
        return []
    tbl = snap.payload
    cols = set(tbl.column_names)
    idx = rows

    def col(name):
        return tbl.column(name).take(idx).to_pylist() if name in cols else None

    data = {n: col(n) for n in (*_PAYLOAD_STR, *_PAYLOAD_NUM, *_PAYLOAD_F64, *_PAYLOAD_BOOL)}
    out = []
    for i in range(rows.size):
        p = cpb.Payload(tier=TIER_KEYFRAME)
        for n in _PAYLOAD_STR:
            if data[n] is not None and data[n][i] is not None:
                setattr(p, n, data[n][i])
        for n in _PAYLOAD_NUM:
            if data[n] is not None and data[n][i] is not None:
                setattr(p, n, int(data[n][i]))
        for n in _PAYLOAD_F64:
            if data[n] is not None and data[n][i] is not None:
                setattr(p, n, float(data[n][i]))
        for n in _PAYLOAD_BOOL:
            if data[n] is not None and data[n][i] is not None:
                setattr(p, n, bool(data[n][i]))
        out.append(p)
    return out


def _query_vector(request, encoder, snap) -> np.ndarray:
    """Lấy vector từ QueryPart. Ưu tiên `vector` (BE đã encode) rồi mới `text`.

    BE nên encode SONG SONG với lời gọi LLM rồi gửi vector sang — tổng latency là
    max(LLM, encode) thay vì tổng (docs/search-design.md §3).
    """
    parts = list(request.query)
    if not parts:
        raise ValueError("query rỗng")
    if len(parts) > 1:
        raise ValueError("v1 chỉ nhận 1 QueryPart (chưa có multi-vector fusion)")

    part = parts[0]
    which = part.WhichOneof("source")
    if which == "vector":
        v = np.asarray(part.vector.values, dtype=np.float32)
        if v.size != snap.dim:
            raise ValueError(f"vector dim {v.size} != snapshot dim {snap.dim}")
        return v
    if which == "text":
        if encoder is None:
            raise ValueError("core chưa nạp text encoder — gửi `vector` hoặc cấu hình ENCODER_S3")
        return encoder.encode_one(part.text)
    raise ValueError(f"v1 chưa hỗ trợ QueryPart.{which} (chỉ text hoặc vector)")


class SearchCoreServiceServicer(pb_grpc.SearchCoreServiceServicer):
    def __init__(self, holder, encoder=None, cfg=None):
        self.holder = holder
        self.encoder = encoder
        self.cfg = cfg

    # ── helpers ──────────────────────────────────────────────────────
    def _snap_or_abort(self, context):
        snap = self.holder.snap if self.holder else None
        if snap is None:
            context.abort(grpc.StatusCode.UNAVAILABLE,
                          "snapshot chưa nạp xong (core đang khởi động)")
        return snap

    # ── RPC ──────────────────────────────────────────────────────────
    def Search(self, request, context):
        t_all = time.perf_counter()
        snap = self._snap_or_abort(context)

        try:
            t0 = time.perf_counter()
            qvec = _query_vector(request, self.encoder, snap)
            encode_ms = (time.perf_counter() - t0) * 1000
        except ValueError as ex:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(ex))

        params = request.params
        top_k = request.top_k or 10
        tags = list(request.filter.tags) if request.HasField("filter") else []

        try:
            res = S.search_with_fallback(
                snap, qvec, top_k=top_k, tags=tags,
                exact_max=self.cfg.exact_subset_max if self.cfg else 20_000,
                rerank_candidates=params.rerank_candidates or (
                    self.cfg.rerank_candidates if self.cfg else 800),
                requested_strategy=request.filter.strategy if request.HasField("filter") else 0,
                min_score=params.min_score or None,
            )
        except ValueError as ex:
            # Tag ngoài vocab: lỗi của caller, không phải lỗi server.
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(ex))

        rows, scores = res.rows, res.scores
        d = request.diversity
        if d.max_per_video or d.min_time_gap_sec or d.dedup_threshold:
            rows, scores = S.diversify(
                snap, rows, scores, top_k=top_k,
                max_per_video=d.max_per_video, min_time_gap_sec=d.min_time_gap_sec,
                dedup_threshold=d.dedup_threshold)

        payloads = _build_payloads(snap, rows) if request.with_payload else []
        # SearchHit nằm trong common.proto (không phải search.proto) — nó dùng chung cho
        # Search/SearchSimilar/SearchGrouped.
        hits = [
            cpb.SearchHit(
                id=int(snap.idmap[r]), index_row=int(r),
                score_exact=float(s), score_fused=float(s), rank=i,
                tier=TIER_KEYFRAME,
                **({"payload": payloads[i]} if payloads else {}),
            )
            for i, (r, s) in enumerate(zip(rows, scores))
        ]

        total_ms = (time.perf_counter() - t_all) * 1000
        timings = cpb.Timings(
            encode_ms=encode_ms,
            coarse_ms=res.timings_ms.get("coarse_ms", 0.0),
            rerank_ms=res.timings_ms.get("rerank_ms", 0.0),
            filter_ms=res.timings_ms.get("filter_ms", 0.0),
            total_ms=total_ms,
            candidates_scanned=int(min(res.candidate_count, 2**31 - 1)),
            filter_matched=int(res.candidate_count),
            filter_strategy_used=res.strategy,
        )
        metrics.observe_many({**res.timings_ms, "encode_ms": encode_ms, "total_ms": total_ms})
        metrics.incr("queries_total")
        metrics.incr_strategy(res.strategy)

        return pb.SearchResponse(
            hits=hits,
            meta=cpb.ResponseMeta(
                request_id=request.ctx.request_id, snapshot_ver=snap.version,
                timings=timings, warnings=res.warnings, tags_used=res.tags_used),
            total_estimated=int(res.candidate_count),
        )

    def SearchTemporal(self, request, context):
        t_all = time.perf_counter()
        snap = self._snap_or_abort(context)

        if len(request.events) != 2:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT,
                          "v1 SearchTemporal chỉ nhận đúng 2 event")

        try:
            qvec1 = _query_vector(request.events[0], self.encoder, snap)
            qvec2 = _query_vector(request.events[1], self.encoder, snap)
        except ValueError as ex:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(ex))

        params = request.params
        tags = list(request.filter.tags) if request.HasField("filter") else []
        requested_strategy = request.filter.strategy if request.HasField("filter") else 0
        cfg = self.cfg

        try:
            res = T.search_temporal(
                snap, qvec1, qvec2, tags=tags,
                candidates_per_event=cfg.trake_candidates_per_event if cfg else 500,
                max_pairs_per_video=cfg.trake_max_pairs_per_video if cfg else 5,
                min_gap_sec=cfg.trake_min_gap_sec if cfg else 5.0,
                max_gap_sec=cfg.trake_max_gap_sec if cfg else 120.0,
                lam=cfg.trake_lambda if cfg else 0.00557,
                sim_weight=cfg.trake_sim_weight if cfg else 0.8,
                time_weight=cfg.trake_time_weight if cfg else 0.2,
                top_k=min(request.top_k, cfg.trake_top_k_chains if cfg else 20)
                    if request.top_k else (cfg.trake_top_k_chains if cfg else 20),
                exact_max=cfg.exact_subset_max if cfg else 20_000,
                rerank_candidates=params.rerank_candidates or (cfg.rerank_candidates if cfg else 800),
                requested_strategy=requested_strategy,
            )
        except ValueError as ex:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(ex))

        chains = []
        for c in res.chains:
            rows = np.array([c.row1, c.row2])
            payloads = _build_payloads(snap, rows) if request.with_payload else [None, None]
            hits = [
                cpb.SearchHit(
                    id=int(snap.idmap[c.row1]), index_row=int(c.row1),
                    score_exact=float(c.sim1), score_fused=float(c.sim1), rank=0,
                    tier=TIER_KEYFRAME,
                    **({"payload": payloads[0]} if payloads[0] is not None else {})),
                cpb.SearchHit(
                    id=int(snap.idmap[c.row2]), index_row=int(c.row2),
                    score_exact=float(c.sim2), score_fused=float(c.sim2), rank=1,
                    tier=TIER_KEYFRAME,
                    **({"payload": payloads[1]} if payloads[1] is not None else {})),
            ]
            chains.append(pb.TemporalChain(
                hits=hits, score=c.score, span_sec=c.t2 - c.t1))

        total_ms = (time.perf_counter() - t_all) * 1000
        metrics.observe_many({**res.timings_ms, "total_ms": total_ms})
        metrics.incr("queries_total")
        metrics.incr("temporal_queries_total")
        return pb.SearchTemporalResponse(
            chains=chains,
            meta=cpb.ResponseMeta(
                request_id=request.ctx.request_id, snapshot_ver=snap.version,
                timings=cpb.Timings(total_ms=total_ms), warnings=res.warnings,
                tags_used=res.tags_used),
        )

    def Encode(self, request, context):
        """Lộ ra để BE encode SONG SONG với lời gọi LLM (docs/search-design.md §3)."""
        if self.encoder is None:
            context.abort(grpc.StatusCode.UNAVAILABLE, "core chưa nạp text encoder")
        if request.WhichOneof("input") != "text":
            context.abort(grpc.StatusCode.UNIMPLEMENTED,
                          "v1 chỉ encode text (vision tower nằm ở ingest)")
        t0 = time.perf_counter()
        vec = self.encoder.encode_one(request.text)
        ms = (time.perf_counter() - t0) * 1000
        metrics.observe("encode_ms", ms)
        metrics.incr("encode_total")
        return pb.EncodeResponse(
            vector=cpb.Vector(values=vec.tolist()),
            meta=cpb.ResponseMeta(request_id=request.ctx.request_id,
                                  timings=cpb.Timings(encode_ms=ms, total_ms=ms)),
        )


class AdminServiceServicer(apb_grpc.AdminServiceServicer):
    def __init__(self, holder, encoder=None, cfg=None, loader=None):
        self.holder = holder
        self.encoder = encoder
        self.cfg = cfg
        self.loader = loader   # callable(path) -> Snapshot, để LoadSnapshot

    def Health(self, request, context):
        snap = self.holder.snap if self.holder else None
        return apb.HealthResponse(
            ready=snap is not None,
            snapshot_ver=snap.version if snap else "",
            point_count=snap.count if snap else 0,
            encoder_name=getattr(self.encoder, "name", "") if self.encoder else "",
            stub_mode=bool(self.cfg.stub_mode) if self.cfg else False,
            loaded_tiers=[TIER_KEYFRAME] if snap else [],
        )

    def GetTagVocab(self, request, context):
        snap = self.holder.snap if self.holder else None
        if snap is None:
            context.abort(grpc.StatusCode.UNAVAILABLE, "snapshot chưa nạp xong")
        counts = snap.tag_counts
        tags = [
            apb.TagInfo(
                id=tid, name=info.get("name", ""), description=info.get("description", ""),
                point_count=int(counts[tid]) if tid < len(counts) else 0,
            )
            for tid, info in sorted(snap.tag_vocab.items())
        ]
        return apb.GetTagVocabResponse(
            tags=tags, snapshot_ver=snap.version,
            unassigned_count=int(snap.unassigned_count),
            meta=cpb.ResponseMeta(request_id=request.ctx.request_id,
                                  snapshot_ver=snap.version),
        )

    def Stats(self, request, context):
        snap = self.holder.snap if self.holder else None
        snapshot = metrics.snapshot()
        total = snapshot["stages"].get("total_ms", {})
        if request.reset:
            metrics.reset()
        return apb.StatsResponse(
            queries_total=snapshot["counters"].get("queries_total", 0),
            latency_p50_ms=total.get("p50", 0.0),
            latency_p95_ms=total.get("p95", 0.0),
            latency_p99_ms=total.get("p99", 0.0),
            index_bytes=(snap.count * snap.dim) if snap else 0,
            uptime_sec=snapshot["uptime_sec"],
            queries_by_task={k: v for k, v in snapshot["by_strategy"].items()},
        )

    def LoadSnapshot(self, request, context):
        if self.loader is None:
            context.abort(grpc.StatusCode.UNIMPLEMENTED, "core không cấu hình loader")
        t0 = time.perf_counter()
        try:
            # single-flight nằm trong holder: hai lần load chồng nhau giữ 3x snapshot -> OOM
            new = self.holder.load_and_swap(lambda: self.loader(request.path))
        except RuntimeError as ex:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(ex))
        except Exception as ex:   # snapshot lỗi không được làm chết server
            log.exception("load snapshot lỗi")
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, f"{type(ex).__name__}: {ex}")
        return apb.LoadSnapshotResponse(
            ok=True, message="đã swap", version=new.version,
            load_ms=int((time.perf_counter() - t0) * 1000))
