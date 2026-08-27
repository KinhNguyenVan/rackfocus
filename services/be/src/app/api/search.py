"""POST /api/search — LLM chọn tag + viết lại query, rồi encode BẢN VIẾT LẠI.

docs/search-design.md §3 mô tả LLM chạy song song với encode (`asyncio.gather`, tổng =
max thay vì cộng). Thiết kế đó KHÔNG còn đúng: encode phải chờ LLM vì nó cần chính bản
`enriched` để embedding. Lý do đo được ghi ở comment trong `search()`.

Response trả lại `tags_used` + `candidate_count` + `strategy`: lọc tag là CỨNG, frame mang
tag không được chọn là không thể với tới. Không phơi ra thì user nhận một trang kết quả
trông rất tự tin từ 1/8 corpus không chứa đáp án mà không có cách nào biết. Cùng lý do,
`enrichment.encoded_text` phơi ra chữ THỰC SỰ được encode.
"""
from __future__ import annotations

import logging
import time
import uuid

import grpc
from fastapi import APIRouter, HTTPException

from ..clients import searchcore
from ..config import get_settings
from ..schemas.search import (
    STRATEGY_NAMES,
    EnrichmentInfo,
    Hit,
    SearchRequest,
    SearchResponse,
    TagItem,
    TagsResponse,
)
from ..services import cache
from ..services import enrich as enrich_svc
from ..services import tagvocab

log = logging.getLogger("app.api.search")
router = APIRouter()


@router.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest) -> SearchResponse:
    st = get_settings()
    rid = uuid.uuid4().hex[:12]
    t_all = time.perf_counter()

    top_k = min(req.top_k or st.default_top_k, st.max_top_k)

    vocab, snap_ver = await tagvocab.get(st)

    # ── Encode bản "enriched" tiếng Anh, KHÔNG phải query gốc ────────
    # Trước đây encode query gốc để chạy SONG SONG với LLM (tổng = max thay vì cộng).
    # Nhưng tokenizer SigLIP tốn 4-7,5x token cho tiếng Việt và giới hạn 64 token là
    # CỨNG (position_embedding [64, 1152]), nên query tiếng Việt dài bị cắt âm thầm và
    # mất đúng phần chi tiết phân biệt ở cuối câu.
    #
    # Đo trên câu 194 token ("nhóm hơn 5 người tập thể dục... một người đeo kính, ba
    # người đội nón đỏ"): encode query gốc -> video đích 0/300 frame; encode bản
    # enriched (28 token) -> frame đích ở rank 2, 203/300 frame thuộc video đích. Đây là
    # chênh lệch giữa "không tìm thấy" và "gần như đầu bảng", nên đáng đổi tính song song.
    #
    # Không còn chạy song song ở đường nào: nhánh có LLM phải chờ enriched mới biết
    # encode chữ gì, nhánh không LLM thì chẳng có gì để chờ.
    async def _encode(text: str):
        key = cache.embedding_key(text, snap_ver)
        hit = cache.embedding.get(key)
        if hit is not None:
            return hit
        vector = await searchcore.encode(text, request_id=rid,
                                         timeout=st.encode_timeout_s)
        cache.embedding.set(key, vector)
        return vector

    async def _enrich(text: str):
        # KHÔNG cache khi enrich lỗi: lỗi thường là tạm thời (timeout, rate limit) mà TTL
        # tới 1 giờ, cache lại là khoá cứng trạng thái "không lọc tag" cho cả phiên thi.
        key = cache.enrichment_key(text, snap_ver, st.llm_model, st.llm_max_tags,
                                   st.llm_tag_confidence_min)
        hit = cache.enrichment.get(key)
        if hit is not None:
            return hit, True
        result = await enrich_svc.enrich(text, vocab, st)
        if result.ok:
            cache.enrichment.set(key, result)
        return result, False

    # Tính một lần rồi dùng lại ở response: điều kiện này trước đây bị viết lặp ba chỗ.
    used_llm = req.tags is None and req.use_llm

    t0 = time.perf_counter()
    try:
        if not used_llm:
            # KHÔNG gọi LLM: client chỉ định tag sẵn, hoặc user tắt LLM.
            enrichment = enrich_svc.Enrichment(enriched_text=req.text)
            encoded_text = req.text
            from_cache = False
        else:
            enrichment, from_cache = await _enrich(req.text)
            # enrich lỗi -> enriched_text đã là query gốc, encode nó là đúng đường lùi.
            encoded_text = enrichment.enriched_text or req.text
        vector = await _encode(encoded_text)
    except grpc.aio.AioRpcError as ex:
        code = ex.code()
        if code == grpc.StatusCode.UNAVAILABLE:
            # Core còn đang nạp snapshot (2-3 phút) hoặc chưa có encoder.
            raise HTTPException(503, f"search core chưa sẵn sàng: {ex.details()}") from ex
        if code == grpc.StatusCode.DEADLINE_EXCEEDED:
            raise HTTPException(504, "encode query quá thời gian") from ex
        raise HTTPException(502, f"searchcore: {code.name}: {ex.details()}") from ex
    # Trước là thời gian của bước SONG SONG; giờ LLM và encode chạy tuần tự nên đây là
    # tổng của cả hai. Đổi tên field để số liệu không nói sai về cách hệ thống chạy.
    llm_encode_ms = (time.perf_counter() - t0) * 1000

    tags = req.tags if req.tags is not None else enrichment.tags

    # ── Search ───────────────────────────────────────────────────────
    try:
        resp = await searchcore.search(
            vector, top_k=top_k, tags=tags, request_id=rid,
            min_score=req.min_score, exact=req.exact,
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
             "llm+encode %.0fms, core %.0fms, tổng %.0fms | cache emb %.0f%% enr %.0f%%",
             rid, req.text[:60], len(hits), list(resp.meta.tags_used),
             tm.filter_matched, STRATEGY_NAMES.get(tm.filter_strategy_used, "?"),
             llm_encode_ms, tm.total_ms, total_ms,
             cache.embedding.stats()["hit_rate"] * 100,
             cache.enrichment.stats()["hit_rate"] * 100)

    return SearchResponse(
        hits=hits,
        tags_used=list(resp.meta.tags_used),
        candidate_count=tm.filter_matched,
        corpus_count=resp.total_estimated,
        strategy=STRATEGY_NAMES.get(tm.filter_strategy_used, "unspecified"),
        warnings=warnings,
        snapshot_ver=resp.meta.snapshot_ver or snap_ver,
        timings_ms={
            "llm_then_encode": round(llm_encode_ms, 2),
            # Cache hit thì request này không gọi LLM -> 0. Giữ latency lần gọi gốc ở
            # field riêng để vẫn thấy được cache đang tiết kiệm bao nhiêu.
            "llm": 0.0 if from_cache else round(enrichment.latency_ms, 2),
            "llm_when_cached": round(enrichment.latency_ms, 2) if from_cache else 0.0,
            "core_encode": round(tm.encode_ms, 2),
            "core_filter": round(tm.filter_ms, 2),
            "core_coarse": round(tm.coarse_ms, 2),
            "core_rerank": round(tm.rerank_ms, 2),
            "core_total": round(tm.total_ms, 2),
            "total": round(total_ms, 2),
        },
        enrichment=EnrichmentInfo(
            model=enrichment.model,
            tags=enrichment.tags,
            enriched_text=enrichment.enriched_text,
            error=enrichment.error,
            used_llm=used_llm,
            guard_added=enrichment.guard_added,
            confidence=enrichment.confidence,
            tag_source=enrichment.tag_source,
            encoded_text=encoded_text,
            cached=from_cache,
        ),
    )


@router.get("/tags", response_model=TagsResponse)
async def tags() -> TagsResponse:
    """Vocab đang dùng — để FE hiển thị và để debug tại sao LLM chọn tag đó."""
    st = get_settings()
    vocab, ver = await tagvocab.get(st)
    return TagsResponse(
        snapshot_ver=ver, count=len(vocab),
        tags=[TagItem(id=k, name=v.name, description=v.description,
                      point_count=v.point_count)
              for k, v in sorted(vocab.items())])
