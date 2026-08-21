"""2-tier: coarse HNSW+SQ8 -> gather refine fp16 -> exact rerank BLAS.

Xem docs/search-design.md §4. Hai điểm dễ làm sai nhất, đã đo:

- **Chi phí là băng thông bộ nhớ, không phải FLOPs.** Gather 8.5k row fp16 = 19.6MB đọc
  ngẫu nhiên -> ép fp32 39MB -> matmul đọc lại 39MB. Đo được 7.4ms (chỉ 2.7 GFLOP/s),
  không phải "dưới 1ms" như tính theo FLOPs. Quét toàn bộ 250k = 256ms -> đường không-tag
  BUỘC phải dùng HNSW.
- **Tombstone phải lọc TRƯỚC khi rank.** Lọc sau thì top_k=10 với 3 điểm đã xoá trả về 7.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger("searchcore.search")

# Khớp FilterStrategy trong proto/searchcore/v1/common.proto
STRATEGY_UNSPECIFIED = 0
STRATEGY_PRE = 1
STRATEGY_POST = 2
STRATEGY_EXACT_SUBSET = 3


@dataclass
class SearchResult:
    rows: np.ndarray                      # row trong snapshot, đã sắp theo điểm giảm
    scores: np.ndarray
    strategy: int = STRATEGY_UNSPECIFIED
    candidate_count: int = 0
    tags_used: tuple[int, ...] = ()
    warnings: list[str] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)


def choose_strategy(candidate_count: int, universe: int, exact_max: int,
                    requested: int = STRATEGY_UNSPECIFIED) -> int:
    """Chọn theo cardinality THẬT (đếm từ CSR), không phải ước lượng.

    Ngưỡng duy nhất `exact_max`. Trước đây tài liệu ghi 20k ở bảng rồi lại nói "đúng như
    Handoff §3.5 (<=10_000)" — hai số khác nhau trình bày như một. Giờ chỉ còn một chỗ.
    """
    if requested != STRATEGY_UNSPECIFIED:
        return requested
    if candidate_count >= universe:
        # Filter khớp tất cả -> đi thẳng HNSW. Không short-circuit thì còn CHẬM HƠN cả
        # không filter, vì thêm selector match-everything vào mỗi bước duyệt đồ thị.
        return STRATEGY_PRE
    if candidate_count <= exact_max:
        return STRATEGY_EXACT_SUBSET
    return STRATEGY_PRE


def _rerank(snap, rows: np.ndarray, qvec: np.ndarray, top_k: int):
    """Gather fp16 -> fp32 -> 1 GEMV -> argpartition top_k.

    argpartition chứ không argsort: O(n) thay vì O(n log n), ~0.2ms vs 1-2ms ở 20k.
    """
    if rows.size == 0:
        return rows, np.empty(0, dtype=np.float32)
    cand = np.asarray(snap.refine[rows], dtype=np.float32)
    scores = cand @ qvec
    k = min(top_k, scores.size)
    part = np.argpartition(-scores, k - 1)[:k]
    part = part[np.argsort(-scores[part])]
    return rows[part], scores[part]


def _coarse(snap, qvec: np.ndarray, n_probe: int, allow_rows: np.ndarray | None):
    """HNSW coarse. `allow_rows` != None -> dùng IDSelector.

    Đây là chỗ mà tuyên bố "vấn đề IdSet biến mất" KHÔNG còn đúng: selector là bitmap/
    hash-set dựng lại mỗi request. Chỉ nhánh EXACT_SUBSET mới thật sự không cần nó.
    """
    import faiss

    q = qvec.reshape(1, -1).astype(np.float32)
    params = None
    if allow_rows is not None:
        sel = faiss.IDSelectorBatch(allow_rows.size, faiss.swig_ptr(
            np.ascontiguousarray(allow_rows, dtype=np.int64)))
        params = faiss.SearchParametersHNSW(sel=sel)
        params.efSearch = snap.index.hnsw.efSearch
    _, ind = snap.index.search(q, n_probe, params=params) if params is not None \
        else snap.index.search(q, n_probe)
    rows = ind[0]
    return rows[rows >= 0].astype(np.int64)


def search(snap, qvec: np.ndarray, *, top_k: int = 10, tags=None,
           exact_max: int = 20_000, rerank_candidates: int = 800,
           requested_strategy: int = STRATEGY_UNSPECIFIED,
           min_score: float | None = None) -> SearchResult:
    """Một query, một vector. `tags` rỗng/None = search toàn bộ corpus."""
    t = {}
    warnings: list[str] = []
    qvec = np.ascontiguousarray(qvec, dtype=np.float32).ravel()
    if qvec.size != snap.dim:
        raise ValueError(f"query dim {qvec.size} != snapshot dim {snap.dim}")

    tags = tuple(int(x) for x in (tags or ()))

    # ── candidate ────────────────────────────────────────────────────
    t0 = time.perf_counter()
    if tags:
        cand = snap.rows_for_tags(tags)          # ValueError nếu tag ngoài vocab
        if cand.size == 0:
            warnings.append("tag_empty")
        # Tombstone lọc TRƯỚC khi rank.
        cand = cand[snap.alive_mask(cand)]
    else:
        cand = None
    t["filter_ms"] = (time.perf_counter() - t0) * 1000

    universe = snap.count
    cand_count = universe if cand is None else int(cand.size)
    strategy = choose_strategy(cand_count, universe, exact_max, requested_strategy)

    # ── search ───────────────────────────────────────────────────────
    t0 = time.perf_counter()
    if strategy == STRATEGY_EXACT_SUBSET:
        rows, scores = _rerank(snap, cand, qvec, top_k)
        t["coarse_ms"] = 0.0
    else:
        coarse = _coarse(snap, qvec, max(rerank_candidates, top_k), cand)
        t["coarse_ms"] = (time.perf_counter() - t0) * 1000
        coarse = coarse[snap.alive_mask(coarse)]      # trước khi rank
        t0 = time.perf_counter()
        rows, scores = _rerank(snap, coarse, qvec, top_k)
    t["rerank_ms"] = (time.perf_counter() - t0) * 1000

    if min_score is not None and rows.size:
        # Điểm là fp16 (sai số đo được ~4.1e-6). Đừng dùng min_score như ngưỡng tuyệt
        # đối mà bỏ qua sai số này — xem docs/search-design.md §8.
        keep = scores >= min_score
        rows, scores = rows[keep], scores[keep]

    return SearchResult(rows=rows, scores=scores, strategy=strategy,
                        candidate_count=cand_count, tags_used=tags,
                        warnings=warnings, timings_ms=t)


def search_with_fallback(snap, qvec, *, top_k=10, tags=None, **kw) -> SearchResult:
    """Tagged search, tự rơi về untagged khi tag lọc quá tay.

    Lọc tag là CỨNG: frame mang tag không được chọn là không thể với tới ở mọi
    ef_search/top_k. Không có fallback thì LLM chọn sai tag cho một trang kết quả trông
    rất tự tin từ vùng corpus không chứa đáp án — user không có cách nào phân biệt.
    """
    res = search(snap, qvec, top_k=top_k, tags=tags, **kw)
    if tags and res.rows.size < top_k:
        full = search(snap, qvec, top_k=top_k, tags=None, **kw)
        if full.rows.size > res.rows.size:
            full.warnings = [*res.warnings, "tag_fallback"]
            full.tags_used = ()
            log.info("tag_fallback: tagged trả %d < top_k=%d -> search toàn bộ được %d",
                     res.rows.size, top_k, full.rows.size)
            return full
    return res


def diversify(snap, rows: np.ndarray, scores: np.ndarray, *, top_k: int,
              max_per_video: int = 0, min_time_gap_sec: float = 0.0,
              dedup_threshold: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Bỏ near-duplicate. Corpus CỐ Ý có 7 keyframe/shot nên không dedup thì 1 trang 10
    kết quả thực tế là 2 khoảnh khắc lặp lại — common.proto tự ghi điều này ở Diversity.

    Giữ thứ tự điểm giảm dần, chọn greedy.
    """
    if rows.size == 0 or not (max_per_video or min_time_gap_sec or dedup_threshold):
        return rows[:top_k], scores[:top_k]

    names = snap.payload.column("video_name").take(rows).to_pylist() \
        if "video_name" in snap.payload.column_names else [""] * rows.size
    times = snap.payload.column("keyframe_time").take(rows).to_pylist() \
        if "keyframe_time" in snap.payload.column_names else [0.0] * rows.size

    vecs = np.asarray(snap.refine[rows], dtype=np.float32) if dedup_threshold else None

    keep: list[int] = []
    per_video: dict[str, int] = {}
    for i in range(rows.size):
        if max_per_video and per_video.get(names[i], 0) >= max_per_video:
            continue
        if min_time_gap_sec and any(
                names[i] == names[j] and abs(times[i] - times[j]) < min_time_gap_sec
                for j in keep):
            continue
        if dedup_threshold and any(
                float(vecs[i] @ vecs[j]) >= dedup_threshold for j in keep):
            continue
        keep.append(i)
        per_video[names[i]] = per_video.get(names[i], 0) + 1
        if len(keep) >= top_k:
            break

    sel = np.asarray(keep, dtype=np.int64)
    return rows[sel], scores[sel]
