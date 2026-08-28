"""POST /api/search/temporal — TRAKE: 2 sự kiện có thứ tự.

Xem docs/superpowers/specs/2026-08-24-temporal-search-design.md. Điểm quan trọng:
SearchTemporalRequest ở core chỉ có MỘT Filter cho cả request (không phải 1/event) --
nên khi use_llm=True, hai event được enrich RIÊNG (mỗi event mô tả một thứ khác nhau)
nhưng tag của chúng được HỢP lại thành một tập duy nhất trước khi gửi sang core.
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
from ..services import segment as segment_svc
from ..services import tagvocab
from .search import Hit

log = logging.getLogger("app.api.search_temporal")
router = APIRouter()


class TemporalSearchRequest(BaseModel):
    event1: str = Field(min_length=1)
    event2: str = Field(min_length=1)
    # Mặc định TẮT cho temporal (khác search thường mặc định BẬT): mỗi event enrich
    # riêng có thể ra tag khác nhau, và lọc tag là CỨNG -- một tag sai ở một event có
    # thể giết chuỗi. Xem docs/runbook.md về rủi ro TRAKE + tag partitioning.
    use_llm: bool = False
    exact: bool = False
    top_k: int | None = None


class TemporalChain(BaseModel):
    video_name: str
    score: float
    span_sec: float
    hits: list[Hit]


class TemporalSearchResponse(BaseModel):
    chains: list[TemporalChain]
    warnings: list[str]
    tags_used: list[int]
    snapshot_ver: str
    timings_ms: dict[str, float]


class PrepareRequest(BaseModel):
    query: str = Field(min_length=1)


class SegmentOut(BaseModel):
    order: int
    english_clip_query: str


class PrepareResponse(BaseModel):
    segments: list[SegmentOut]
    tags: list[int]
    # CẢ vocab, không chỉ tag đã chọn: UI phải hiện tag chưa chọn để user tick THÊM vào,
    # không chỉ bỏ bớt. Khoá JSON luôn là chuỗi -> tới FE thành {"3": "..."}.
    tag_names: dict[int, str]
    confidence: float
    tag_source: str
    warnings: list[str]
    snapshot_ver: str
    timings_ms: dict[str, float]


@router.post("/search/temporal/prepare", response_model=PrepareResponse)
async def prepare(req: PrepareRequest) -> PrepareResponse:
    """Bước 1 của temporal: LLM tách đoạn + LLM chọn tag, CHẠY SONG SONG.

    Không nằm trên hot path — sau bước này người dùng còn phải sửa câu và bấm chọn 2 sự
    kiện, nên ngân sách 100-200ms không áp ở đây. Bước tách đoạn nhiều khả năng là bên
    chậm hơn: `segment_prompt.txt` dài hơn hẳn prompt trong `enrich.py`.

    Không có try/except: cả `segment()` lẫn `enrich()` đều cam kết không raise.
    """
    st = get_settings()
    t_all = time.perf_counter()

    vocab, snap_ver = await tagvocab.get(st)

    segmentation, enrichment = await asyncio.gather(
        segment_svc.segment(req.query, st),
        enrich_svc.enrich(req.query, vocab, st),
    )

    warnings: list[str] = []
    if segmentation.error:
        warnings.append("llm_failed_segment")
    if enrichment.error:
        warnings.append("llm_failed_tags")

    return PrepareResponse(
        segments=[SegmentOut(order=s.order, english_clip_query=s.english_clip_query)
                  for s in segmentation.segments],
        tags=enrichment.tags,
        tag_names={tid: (info.name or info.description)
                   for tid, info in vocab.items()},
        confidence=enrichment.confidence,
        tag_source=enrichment.tag_source,
        warnings=warnings,
        snapshot_ver=snap_ver,
        timings_ms={
            "segment": round(segmentation.latency_ms, 2),
            "enrich": round(enrichment.latency_ms, 2),
            "total": round((time.perf_counter() - t_all) * 1000, 2),
        },
    )


@router.post("/search/temporal", response_model=TemporalSearchResponse)
async def search_temporal(req: TemporalSearchRequest) -> TemporalSearchResponse:
    st = get_settings()
    rid = uuid.uuid4().hex[:12]
    t_all = time.perf_counter()

    vocab, snap_ver = await tagvocab.get(st)

    async def _enrich(text: str):
        if not req.use_llm:
            return enrich_svc.Enrichment(enriched_text=text)
        return await enrich_svc.enrich(text, vocab, st)

    t0 = time.perf_counter()
    try:
        vec1, vec2, enr1, enr2 = await asyncio.gather(
            searchcore.encode(req.event1, request_id=rid, timeout=st.encode_timeout_s),
            searchcore.encode(req.event2, request_id=rid, timeout=st.encode_timeout_s),
            _enrich(req.event1),
            _enrich(req.event2),
        )
    except grpc.aio.AioRpcError as ex:
        code = ex.code()
        if code == grpc.StatusCode.UNAVAILABLE:
            raise HTTPException(503, f"search core chưa sẵn sàng: {ex.details()}") from ex
        if code == grpc.StatusCode.DEADLINE_EXCEEDED:
            raise HTTPException(504, "encode query quá thời gian") from ex
        raise HTTPException(502, f"searchcore: {code.name}: {ex.details()}") from ex
    parallel_ms = (time.perf_counter() - t0) * 1000

    tags = sorted(set(enr1.tags) | set(enr2.tags))

    try:
        resp = await searchcore.search_temporal(
            vec1, vec2, tags=tags, request_id=rid, exact=req.exact,
            top_k=req.top_k, timeout=st.search_timeout_s)
    except grpc.aio.AioRpcError as ex:
        code = ex.code()
        if code == grpc.StatusCode.INVALID_ARGUMENT:
            raise HTTPException(400, f"searchcore từ chối: {ex.details()}") from ex
        if code == grpc.StatusCode.UNAVAILABLE:
            raise HTTPException(503, f"search core chưa sẵn sàng: {ex.details()}") from ex
        if code == grpc.StatusCode.DEADLINE_EXCEEDED:
            raise HTTPException(504, "search quá thời gian") from ex
        raise HTTPException(502, f"searchcore: {code.name}: {ex.details()}") from ex

    def _hit(h) -> Hit:
        return Hit(point_id=h.id, row=h.index_row, score=h.score_exact, rank=h.rank,
                   video_name=h.payload.video_name, frame=h.payload.frame,
                   keyframe_time=h.payload.keyframe_time,
                   start_sec=h.payload.start_sec, end_sec=h.payload.end_sec,
                   keyframe_url=h.payload.keyframe_key, clip_url=h.payload.clip_key,
                   scene_idx=h.payload.scene_idx, has_speech=h.payload.has_speech)

    chains = [
        TemporalChain(
            video_name=c.hits[0].payload.video_name if c.hits else "",
            score=c.score, span_sec=c.span_sec, hits=[_hit(h) for h in c.hits])
        for c in resp.chains
    ]

    warnings = list(resp.meta.warnings)
    if enr1.error:
        warnings.append("llm_failed_event1")
    if enr2.error:
        warnings.append("llm_failed_event2")

    total_ms = (time.perf_counter() - t_all) * 1000
    log.info("[%s] %r + %r -> %d chain | tags=%s | encode+enrich %.0fms, core %.0fms, "
             "tổng %.0fms", rid, req.event1[:40], req.event2[:40], len(chains), tags,
             parallel_ms, resp.meta.timings.total_ms, total_ms)

    return TemporalSearchResponse(
        chains=chains, warnings=warnings, tags_used=list(resp.meta.tags_used),
        snapshot_ver=resp.meta.snapshot_ver or snap_ver,
        timings_ms={
            "encode_and_enrich_parallel": round(parallel_ms, 2),
            "core_total": round(resp.meta.timings.total_ms, 2),
            "total": round(total_ms, 2),
        },
    )
