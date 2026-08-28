"""POST /api/search/temporal — TRAKE: 2 sự kiện có thứ tự.

Xem docs/superpowers/specs/2026-08-28-temporal-llm-segmentation-design.md. Điểm quan trọng:
SearchTemporalRequest ở core chỉ có MỘT Filter cho cả request (không phải 1/event) --
nên chỉ có MỘT tập tag cho cả chuỗi.

Vì thế khi use_llm=True ta gọi enrich đúng MỘT lần trên hai event đã ghép, thay vì enrich
riêng từng event rồi hợp tag. Rẻ hơn một lời gọi LLM, mà còn đúng hơn: enrich riêng lẻ chỉ
thấy nửa câu chuyện nên dễ chọn lĩnh vực lệch, trong khi core dù sao cũng chỉ nhận một
Filter. An toàn vì ở temporal `enriched_text` bị VỨT -- vector query encode thẳng từ
`req.event1`/`req.event2`, enrich ở đây chỉ để lấy `.tags`.
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
    # Chỉ định tag thẳng, bỏ qua LLM — client gửi lại tag lấy từ /search/temporal/prepare
    # sau khi user tick/bỏ tick. Mirror `SearchRequest.tags` (schemas/search.py).
    #
    # [] và None KHÁC nhau và không được gộp: [] = "user đã bỏ tick hết, search toàn kho",
    # None = "không qua prepare, quyết định theo use_llm". Gộp lại là âm thầm bật lại lọc
    # tag mà user vừa cố ý tắt — mà lọc tag là CỨNG, không cứu được ở tầng nào khác.
    tags: list[int] | None = None
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
    # Tách event (segment) và lọc tag (enrich) là HAI công tắc độc lập trên UI. Cờ này chỉ
    # điều khiển enrich; đã vào tới đây nghĩa là user đã bật tách event. False -> bỏ hẳn
    # lời gọi enrich (không phải gọi rồi vứt kết quả), prepare còn đúng một lời gọi LLM.
    use_llm: bool = True


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
    """Bước 1 của temporal: LLM tách đoạn + (tuỳ chọn) LLM chọn tag, CHẠY SONG SONG.

    Không nằm trên hot path — sau bước này người dùng còn phải sửa câu và bấm chọn 2 sự
    kiện, nên ngân sách 100-200ms không áp ở đây. Bước tách đoạn nhiều khả năng là bên
    chậm hơn: `segment_prompt.txt` dài hơn hẳn prompt trong `enrich.py`.

    `use_llm=False` -> chỉ chạy segment. Xem PrepareRequest.use_llm.

    Không có try/except: cả `segment()` lẫn `enrich()` đều cam kết không raise.
    """
    st = get_settings()
    t_all = time.perf_counter()

    vocab, snap_ver = await tagvocab.get(st)

    async def _maybe_enrich() -> enrich_svc.Enrichment:
        if not req.use_llm:
            return enrich_svc.Enrichment(enriched_text=req.query, tag_source="disabled")
        return await enrich_svc.enrich(req.query, vocab, st)

    segmentation, enrichment = await asyncio.gather(
        segment_svc.segment(req.query, st),
        _maybe_enrich(),
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
        # Tắt lọc tag -> trả vocab rỗng luôn. Vocab tồn tại để UI hiện tag cho user tick;
        # không lọc thì không có gì để tick, gửi cả bảng xuống chỉ gây hiểu nhầm là có lọc.
        tag_names={} if not req.use_llm else {
            tid: (info.name or info.description) for tid, info in vocab.items()},
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

    # Tính một lần rồi dùng lại, giống api/search.py:87.
    used_llm = req.tags is None and req.use_llm

    async def _enrich() -> enrich_svc.Enrichment:
        if not used_llm:
            return enrich_svc.Enrichment()
        # Ghép hai event thành một câu rồi enrich MỘT lần. Xem docstring đầu file: core
        # chỉ nhận một Filter nên chỉ cần một tập tag, và LLM thấy cả chuỗi thì chọn lĩnh
        # vực sát hơn là đoán rời từng nửa.
        return await enrich_svc.enrich(f"{req.event1}. {req.event2}", vocab, st)

    t0 = time.perf_counter()
    try:
        vec1, vec2, enr = await asyncio.gather(
            searchcore.encode(req.event1, request_id=rid, timeout=st.encode_timeout_s),
            searchcore.encode(req.event2, request_id=rid, timeout=st.encode_timeout_s),
            _enrich(),
        )
    except grpc.aio.AioRpcError as ex:
        code = ex.code()
        if code == grpc.StatusCode.UNAVAILABLE:
            raise HTTPException(503, f"search core chưa sẵn sàng: {ex.details()}") from ex
        if code == grpc.StatusCode.DEADLINE_EXCEEDED:
            raise HTTPException(504, "encode query quá thời gian") from ex
        raise HTTPException(502, f"searchcore: {code.name}: {ex.details()}") from ex
    parallel_ms = (time.perf_counter() - t0) * 1000

    tags = req.tags if req.tags is not None else sorted(enr.tags)

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
    if enr.error:
        warnings.append("llm_failed_tags")

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
