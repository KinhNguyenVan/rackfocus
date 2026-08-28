"""TRAKE: search 2 sự kiện rồi ghép cặp theo thứ tự thời gian + giới hạn khoảng cách.

Xem docs/superpowers/specs/2026-08-24-temporal-search-design.md. Tái dùng
search_with_fallback cho từng event (KHÔNG brute-force toàn corpus — đã đo là quá chậm
nếu làm hai lần) rồi ghép theo video_name — mỗi video giữ tối đa max_pairs_per_video cặp
tốt nhất (không chỉ MỘT), tất cả cặp của mọi video được gộp vào một pool rồi cắt còn
top_k theo điểm cao nhất. Không làm lại bước merge-overlapping-pairs của bản notebook
(giải quyết bài toán khác: phân tích khám phá nhiều cặp chồng lấn, không phải endpoint
xếp hạng kết quả) -- nhưng CÓ giữ lại ý tưởng PAIRS_PER_VIDEO của notebook đó.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from .search import search_with_fallback

# Event1/event2 search độc lập, cả hai đều là lời gọi FAISS/BLAS (code native, nhả GIL)
# -- chạy trên 2 thread riêng thay vì gọi lần lượt. An toàn: search() chỉ ĐỌC `snap`
# (giống hệt kiểu truy cập đồng thời đã dùng khi nhiều request gRPC cùng đọc chung
# snapshot qua IndexHolder). Pool ở mức module để trả phí tạo thread một lần, không phải
# mỗi request; giới hạn 2 -- đó là mức song song hữu ích tối đa cho hàm này, không phải
# pool dùng chung.
_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="trake-search")


@dataclass
class Chain:
    video_name: str
    row1: int
    row2: int
    score: float
    sim1: float
    sim2: float
    t1: float
    t2: float


@dataclass
class TemporalResult:
    chains: list[Chain] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)
    tags_used: tuple[int, ...] = ()


def search_temporal(snap, qvec1: np.ndarray, qvec2: np.ndarray, *, tags=None,
                    candidates_per_event: int, max_pairs_per_video: int,
                    min_gap_sec: float, max_gap_sec: float,
                    lam: float, sim_weight: float, time_weight: float, top_k: int,
                    exact_max: int = 20_000, rerank_candidates: int = 800,
                    requested_strategy: int = 0) -> TemporalResult:
    t: dict[str, float] = {}
    warnings: list[str] = []

    t0 = time.perf_counter()
    f1 = _executor.submit(search_with_fallback, snap, qvec1, top_k=candidates_per_event,
                          tags=tags, exact_max=exact_max, rerank_candidates=rerank_candidates,
                          requested_strategy=requested_strategy)
    f2 = _executor.submit(search_with_fallback, snap, qvec2, top_k=candidates_per_event,
                          tags=tags, exact_max=exact_max, rerank_candidates=rerank_candidates,
                          requested_strategy=requested_strategy)
    r1, r2 = f1.result(), f2.result()
    t["search_ms"] = (time.perf_counter() - t0) * 1000

    # Ghép warnings + tags_used của cả hai lần search_with_fallback -- trước đây bị bỏ
    # hoàn toàn, mất tín hiệu tag_empty/tag_fallback đúng lúc rủi ro tag-partition của
    # TRAKE (mỗi event enrich riêng) cần thấy nhất. Gộp TRƯỚC các warning early-return
    # bên dưới.
    warnings.extend(f"event1_{w}" for w in r1.warnings)
    warnings.extend(f"event2_{w}" for w in r2.warnings)
    tags_used = tuple(sorted(set(r1.tags_used) | set(r2.tags_used)))

    missing = [c for c in ("video_name", "keyframe_time") if c not in snap.payload.column_names]
    if missing:
        raise ValueError(f"TRAKE cần payload có {missing} -- snapshot thiếu cột thời gian")

    if r1.rows.size == 0:
        warnings.append("temporal_no_candidates_event1")
        return TemporalResult(warnings=warnings, timings_ms=t, tags_used=tags_used)
    if r2.rows.size == 0:
        warnings.append("temporal_no_candidates_event2")
        return TemporalResult(warnings=warnings, timings_ms=t, tags_used=tags_used)

    t0 = time.perf_counter()
    names_col = snap.payload.column("video_name")
    times_col = snap.payload.column("keyframe_time")

    def group(rows, scores):
        names = names_col.take(rows).to_pylist()
        times = times_col.take(rows).to_pylist()
        by_video: dict[str, list[tuple]] = {}
        for row, sim, name, tm in zip(rows.tolist(), scores.tolist(), names, times):
            by_video.setdefault(name, []).append((row, sim, tm))
        return by_video

    by_video1 = group(r1.rows, r1.scores)
    by_video2 = group(r2.rows, r2.scores)

    common = set(by_video1) & set(by_video2)
    if not common:
        warnings.append("temporal_no_common_video")
        return TemporalResult(warnings=warnings, timings_ms=t, tags_used=tags_used)

    # Ghép cặp bằng broadcast numpy thay vì vòng lặp Python lồng nhau: candidates_per_event
    # mặc định 500, nên một video có đủ ứng viên cả hai bên là 500x500 = 250k cặp cần so
    # dt -- vòng lặp Python thuần cỡ đó tốn hàng chục ms, đáng kể so với ngân sách
    # 100-200ms. Tính dt/decay/score cho CẢ ma trận n1 x n2 bằng C (numpy), rồi chỉ lọc +
    # tạo Chain cho phần thật sự giữ lại (<= max_pairs_per_video), giống cách _rerank()
    # trong search.py dùng argpartition thay vì sort toàn bộ.
    pool: list[Chain] = []
    for video in common:
        rows1, sims1, times1 = (np.asarray(x) for x in zip(*by_video1[video]))
        rows2, sims2, times2 = (np.asarray(x) for x in zip(*by_video2[video]))

        dt = times2[None, :] - times1[:, None]
        valid = (dt > 0) & (dt >= min_gap_sec) & (dt <= max_gap_sec)
        if not valid.any():
            continue

        decay = np.exp(-lam * (dt - min_gap_sec))
        score = sim_weight * (sims1[:, None] + sims2[None, :]) + time_weight * decay

        i_idx, j_idx = np.nonzero(valid)          # thứ tự (i tăng dần, j tăng dần trong i)
        pair_scores = score[i_idx, j_idx]

        k = min(max_pairs_per_video, pair_scores.size)
        top = np.argpartition(-pair_scores, k - 1)[:k]
        # kind="stable" để hoà điểm giữ đúng thứ tự sinh ra (i tăng dần, j tăng dần) --
        # khớp hành vi sort ổn định (list.sort) của bản vòng lặp Python cũ.
        top = top[np.argsort(-pair_scores[top], kind="stable")]

        pool.extend(
            Chain(video_name=video, row1=int(rows1[i]), row2=int(rows2[j]),
                 score=float(pair_scores[t]), sim1=float(sims1[i]), sim2=float(sims2[j]),
                 t1=float(times1[i]), t2=float(times2[j]))
            for t, i, j in zip(top, i_idx[top], j_idx[top])
        )

    if not pool:
        warnings.append("temporal_no_valid_gap")

    pool.sort(key=lambda c: -c.score)
    t["join_ms"] = (time.perf_counter() - t0) * 1000
    return TemporalResult(chains=pool[:top_k], warnings=warnings, timings_ms=t, tags_used=tags_used)
