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
from dataclasses import dataclass, field

import numpy as np

from .search import search_with_fallback


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


def search_temporal(snap, qvec1: np.ndarray, qvec2: np.ndarray, *, tags=None,
                    candidates_per_event: int, max_pairs_per_video: int,
                    min_gap_sec: float, max_gap_sec: float,
                    lam: float, sim_weight: float, time_weight: float, top_k: int,
                    exact_max: int = 20_000, rerank_candidates: int = 800,
                    requested_strategy: int = 0) -> TemporalResult:
    t: dict[str, float] = {}
    warnings: list[str] = []

    t0 = time.perf_counter()
    r1 = search_with_fallback(snap, qvec1, top_k=candidates_per_event, tags=tags,
                              exact_max=exact_max, rerank_candidates=rerank_candidates,
                              requested_strategy=requested_strategy)
    r2 = search_with_fallback(snap, qvec2, top_k=candidates_per_event, tags=tags,
                              exact_max=exact_max, rerank_candidates=rerank_candidates,
                              requested_strategy=requested_strategy)
    t["search_ms"] = (time.perf_counter() - t0) * 1000

    if r1.rows.size == 0:
        warnings.append("temporal_no_candidates_event1")
        return TemporalResult(warnings=warnings, timings_ms=t)
    if r2.rows.size == 0:
        warnings.append("temporal_no_candidates_event2")
        return TemporalResult(warnings=warnings, timings_ms=t)

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
        return TemporalResult(warnings=warnings, timings_ms=t)

    pool: list[Chain] = []
    for video in common:
        video_pairs: list[Chain] = []
        for row1, sim1, time1 in by_video1[video]:
            for row2, sim2, time2 in by_video2[video]:
                dt = time2 - time1
                if dt < min_gap_sec or dt > max_gap_sec:
                    continue
                decay = float(np.exp(-lam * (dt - min_gap_sec)))
                score = sim_weight * (sim1 + sim2) + time_weight * decay
                video_pairs.append(Chain(video_name=video, row1=row1, row2=row2,
                                         score=score, sim1=sim1, sim2=sim2,
                                         t1=time1, t2=time2))
        video_pairs.sort(key=lambda c: -c.score)
        pool.extend(video_pairs[:max_pairs_per_video])

    if not pool:
        warnings.append("temporal_no_valid_gap")

    pool.sort(key=lambda c: -c.score)
    t["join_ms"] = (time.perf_counter() - t0) * 1000
    return TemporalResult(chains=pool[:top_k], warnings=warnings, timings_ms=t)
