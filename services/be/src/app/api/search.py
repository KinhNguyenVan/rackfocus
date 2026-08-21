"""POST /api/search — LLM chọn tag SONG SONG với encode query.

Xem docs/search-design.md §3. Điểm cốt lõi: `asyncio.gather` chứ không await tuần tự.
LLM 80-300ms và encode 170-420ms chạy chồng nhau -> tổng là max, không phải tổng cộng.

Response trả lại `tags_used` + `candidate_count` + `strategy`: lọc tag là CỨNG, frame mang
tag không được chọn là không thể với tới. Không phơi ra thì user nhận một trang kết quả
trông rất tự tin từ 1/8 corpus không chứa đáp án mà không có cách nào biết.
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid

import grpc
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..clients import searchcore
from ..config import get_settings
from ..services import enrich as enrich_svc
from ..services import tagvocab

log = logging.getLogger("app.api.search")
router = APIRouter()

# Khớp FilterStrategy trong proto — trả tên ra ngoài cho dễ đọc.
_STRATEGY = {0: "unspecified", 1: "pre", 2: "post", 3: "exact_subset"}


class SearchRequest(BaseModel):
    text: str = Field(min_length=1)
    top_k: int | None = None
    # Tắt LLM để tìm toàn bộ corpus. Đây là đường lùi khi LLM chọn sai tag.
    use_llm: bool = True
    # Chỉ định tag thẳng, bỏ qua LLM (dùng để debug hoặc khi user tự chọn).
    tags: list[int] | None = None
    min_score: float | None = None


class Hit(BaseModel):
    point_id: int
    row: int
    score: float
    rank: int
    video_name: str = ""
    frame: int = 0
    keyframe_time: float = 0.0
    start_sec: float = 0.0
    end_sec: float = 0.0
    keyframe_url: str = ""
    clip_url: str = ""
    scene_idx: int = 0
    has_speech: bool = False


class SearchResponse(BaseModel):
    hits: list[Hit]
    # Phơi ra để user biết mình vừa tìm trong bao nhiêu phần của kho.
    tags_used: list[int]
    candidate_count: int
    corpus_count: int
    strategy: str
    warnings: list[str]
    snapshot_ver: str
    timings_ms: dict[str, float]
    enrichment: dict


@router.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest) -> SearchResponse:
    st = get_settings()
    rid = uuid.uuid4().hex[:12]
    t_all = time.perf_counter()

    top_k = min(req.top_k or st.default_top_k, st.max_top_k)

    vocab, snap_ver = await tagvocab.get(st)

    # ── Điểm mấu chốt: encode và LLM chạy SONG SONG ─────────────────
    # Encode QUERY GỐC của user, không phải bản LLM viết lại: giữ đúng cách diễn đạt,
    # và không phải chờ LLM xong mới bắt đầu encode.
    async def _encode():
        return await searchcore.encode(req.text, request_id=rid,
                                      timeout=st.encode_timeout_s)

    async def _enrich():
        if req.tags is not None or not req.use_llm:
            return enrich_svc.Enrichment(enriched_text=req.text)
        return await enrich_svc.enrich(req.text, vocab, st)

    t0 = time.perf_counter()
    try:
        vector, enrichment = await asyncio.gather(_encode(), _enrich())
    except grpc.aio.AioRpcError as ex:
        code = ex.code()
        if code == grpc.StatusCode.UNAVAILABLE:
            # Core còn đang nạp snapshot (2-3 phút) hoặc chưa có encoder.
            raise HTTPException(503, f"search core chưa sẵn sàng: {ex.details()}") from ex
        if code == grpc.StatusCode.DEADLINE_EXCEEDED:
            raise HTTPException(504, "encode query quá thời gian") from ex
        raise HTTPException(502, f"searchcore: {code.name}: {ex.details()}") from ex
    parallel_ms = (time.perf_counter() - t0) * 1000

    tags = req.tags if req.tags is not None else enrichment.tags

    # ── Search ───────────────────────────────────────────────────────
    try:
        resp = await searchcore.search(
            vector, top_k=top_k, tags=tags, request_id=rid,
            min_score=req.min_score,
            max_per_video=st.diversity_max_per_video,
            min_time_gap_sec=st.diversity_min_time_gap_sec,
            dedup_threshold=st.diversity_dedup_threshold,
            timeout=st.search_timeout_s)
    except grpc.aio.AioRpcError as ex:
        code = ex.code()
        if code == grpc.StatusCode.INVALID_ARGUMENT:
            raise HTTPException(400, f"searchcore từ chối: {ex.details()}") from ex
        if code == grpc.StatusCode.UNAVAILABLE:
            raise HTTPException(503, f"search core chưa sẵn sàng: {ex.details()}") from ex
        if code == grpc.StatusCode.DEADLINE_EXCEEDED:
            raise HTTPException(504, "search quá thời gian") from ex
        raise HTTPException(502, f"searchcore: {code.name}: {ex.details()}") from ex

    tm = resp.meta.timings
    hits = [
        Hit(point_id=h.id, row=h.index_row, score=h.score_exact, rank=h.rank,
            video_name=h.payload.video_name, frame=h.payload.frame,
            keyframe_time=h.payload.keyframe_time,
            start_sec=h.payload.start_sec, end_sec=h.payload.end_sec,
            keyframe_url=h.payload.keyframe_key, clip_url=h.payload.clip_key,
            scene_idx=h.payload.scene_idx, has_speech=h.payload.has_speech)
        for h in resp.hits
    ]

    warnings = list(resp.meta.warnings)
    if enrichment.error:
        warnings.append("llm_failed")

    total_ms = (time.perf_counter() - t_all) * 1000
    log.info("[%s] %r -> %d hit | tags=%s candidate=%d strategy=%s | "
             "llm+encode %.0fms, core %.0fms, tổng %.0fms",
             rid, req.text[:60], len(hits), list(resp.meta.tags_used),
             tm.filter_matched, _STRATEGY.get(tm.filter_strategy_used, "?"),
             parallel_ms, tm.total_ms, total_ms)

    return SearchResponse(
        hits=hits,
        tags_used=list(resp.meta.tags_used),
        candidate_count=tm.filter_matched,
        corpus_count=resp.total_estimated,
        strategy=_STRATEGY.get(tm.filter_strategy_used, "unspecified"),
        warnings=warnings,
        snapshot_ver=resp.meta.snapshot_ver or snap_ver,
        timings_ms={
            "llm_and_encode_parallel": round(parallel_ms, 2),
            "llm": round(enrichment.latency_ms, 2),
            "core_encode": round(tm.encode_ms, 2),
            "core_filter": round(tm.filter_ms, 2),
            "core_coarse": round(tm.coarse_ms, 2),
            "core_rerank": round(tm.rerank_ms, 2),
            "core_total": round(tm.total_ms, 2),
            "total": round(total_ms, 2),
        },
        enrichment={
            "model": enrichment.model,
            "tags": enrichment.tags,
            "enriched_text": enrichment.enriched_text,
            "error": enrichment.error,
            "used_llm": req.tags is None and req.use_llm,
        },
    )


@router.get("/tags")
async def tags() -> dict:
    """Vocab đang dùng — để FE hiển thị và để debug tại sao LLM chọn tag đó."""
    st = get_settings()
    vocab, ver = await tagvocab.get(st)
    return {"snapshot_ver": ver, "count": len(vocab),
            "tags": [{"id": k, "description": v} for k, v in sorted(vocab.items())]}
