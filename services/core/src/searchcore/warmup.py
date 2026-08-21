"""Chạy query giả sau khi load: warm HNSW graph, page cache của mmap, BLAS, ONNX arena.

Handoff I7: warmup là BẮT BUỘC, không phải tuỳ chọn — query đầu chậm 10-50x nếu bỏ qua,
và "trong thi đấu query đầu là query tính điểm".

Bản warmup cũ chỉ chạy 2-tier, tức bỏ sót đúng hai chỗ đắt nhất:
  - **ONNX text encoder**: 53.3 GFLOP/query, và lần run đầu còn phải cấp arena +
    pre-pack weight. Đây là thành phần chiếm >90% ngân sách latency.
  - **Nhánh EXACT_SUBSET**: gather ngẫu nhiên trên `visual.f16`. Không warm thì lần đầu
    là ~4800 major fault (~380ms) thay vì ~7ms.

Xem docs/search-design.md §7.
"""
from __future__ import annotations

import logging
import time

import numpy as np

from . import search as S

log = logging.getLogger("searchcore.warmup")

# Câu ngắn, không nghĩa gì — chỉ để chạy đủ graph. Độ dài không ảnh hưởng chi phí vì
# SigLIP pad cứng 64 token.
_TEXTS = ["a", "hai người đang nói chuyện", "xe chạy trên đường", "sân vận động"]


def warmup(snap, encoder=None, *, queries: int = 50, top_k: int = 10,
           exact_max: int = 20_000, rerank_candidates: int = 800) -> dict[str, float]:
    """Trả thời gian từng phần (ms) để log và để biết warmup có thực sự chạy."""
    out: dict[str, float] = {}
    rng = np.random.default_rng(0)

    # 1) Encoder — quan trọng nhất, và tốn nhất.
    if encoder is not None:
        t0 = time.perf_counter()
        first = None
        for i in range(max(1, min(queries, 8))):   # 8 lần là đủ để ORT ổn định
            t1 = time.perf_counter()
            encoder.encode([_TEXTS[i % len(_TEXTS)]])
            if first is None:
                first = (time.perf_counter() - t1) * 1000
        out["encoder_total_ms"] = (time.perf_counter() - t0) * 1000
        out["encoder_first_ms"] = first or 0.0
        log.info("warm encoder: lần đầu %.0fms, tổng %.0fms",
                 out["encoder_first_ms"], out["encoder_total_ms"])

    qs = rng.standard_normal((queries, snap.dim)).astype(np.float32)
    qs /= np.linalg.norm(qs, axis=1, keepdims=True)

    # 2) Đường 2-tier (không tag) — warm HNSW graph.
    t0 = time.perf_counter()
    for q in qs:
        S.search(snap, q, top_k=top_k, tags=None,
                 exact_max=exact_max, rerank_candidates=rerank_candidates)
    out["two_tier_ms"] = (time.perf_counter() - t0) * 1000

    # 3) Đường EXACT_SUBSET — warm page cache của visual.f16 trên các bucket tag.
    #    Không có tag thì vẫn phải chạm refine store: dùng row ngẫu nhiên.
    t0 = time.perf_counter()
    n_tags = len(snap.tag_counts) if getattr(snap, "tag_counts", None) is not None else 0
    if n_tags:
        for i in range(min(queries, n_tags)):
            S.search(snap, qs[i % len(qs)], top_k=top_k, tags=[i % n_tags],
                     exact_max=10**9)   # ép đi nhánh exact
    else:
        step = max(1, snap.count // 20_000)
        rows = np.arange(0, snap.count, step, dtype=np.int64)
        for q in qs[: min(8, len(qs))]:
            np.asarray(snap.refine[rows], dtype=np.float32) @ q
    out["exact_subset_ms"] = (time.perf_counter() - t0) * 1000

    log.info("warmup xong: 2-tier %.0fms, exact_subset %.0fms (%d query)",
             out["two_tier_ms"], out["exact_subset_ms"], queries)
    return out
