"""Test ghép chuỗi TRAKE: đúng thứ tự, tôn trọng gap, top-K cặp mỗi video, gộp +
cắt top_k trên toàn bộ pool.

Xem docs/superpowers/specs/2026-08-24-temporal-search-design.md. Dùng lại fixture `snap`
của conftest.py (200 row, 4 video luân phiên theo i%4, keyframe_time = i*0.2): row r nằm
ở video L26_V{r%4:03d}, thời điểm r*0.2s.
"""
import numpy as np
from conftest import DIM
from searchcore import temporal as T


def q(snap, row):
    return np.asarray(snap.refine[row], dtype=np.float32)


def test_pair_within_gap_bounds_is_scored(snap):
    # candidates_per_event=1 -> event1's only candidate is row 4 itself (self-match,
    # sim=1.0, unbeatable top-1), event2's only candidate is row 12 -- deterministic
    # regardless of the corpus's random vectors, since there's nothing else to pick.
    # row 4 (V000, t=0.8s), row 12 (V000, t=2.4s) -> dt=1.6s.
    res = T.search_temporal(
        snap, q(snap, 4), q(snap, 12), tags=None,
        candidates_per_event=1, max_pairs_per_video=10,
        min_gap_sec=0.1, max_gap_sec=5.0,
        lam=0.01, sim_weight=0.8, time_weight=0.2, top_k=10)

    assert len(res.chains) == 1
    c = res.chains[0]
    assert c.video_name == "L26_V000"
    assert c.row1 == 4
    assert c.row2 == 12
    assert c.t1 == 0.8 and c.t2 == 2.4
    assert c.score > 0
    assert res.warnings == []


def test_pair_below_min_gap_excluded(snap):
    # same rows as above (dt=1.6s), but min_gap_sec=5.0 excludes it
    res = T.search_temporal(
        snap, q(snap, 4), q(snap, 12), tags=None,
        candidates_per_event=1, max_pairs_per_video=10,
        min_gap_sec=5.0, max_gap_sec=120.0,
        lam=0.01, sim_weight=0.8, time_weight=0.2, top_k=10)
    assert res.chains == []
    assert "temporal_no_valid_gap" in res.warnings


def test_pair_above_max_gap_excluded(snap):
    res = T.search_temporal(
        snap, q(snap, 4), q(snap, 12), tags=None,
        candidates_per_event=1, max_pairs_per_video=10,
        min_gap_sec=0.1, max_gap_sec=1.0,
        lam=0.01, sim_weight=0.8, time_weight=0.2, top_k=10)
    assert res.chains == []
    assert "temporal_no_valid_gap" in res.warnings


def test_order_violation_excluded_via_min_gap(snap):
    # event1 query matches the LATER row (12, t=2.4), event2 matches the EARLIER
    # row (4, t=0.8) -> dt = 0.8 - 2.4 = -1.6, always < a positive min_gap_sec
    res = T.search_temporal(
        snap, q(snap, 12), q(snap, 4), tags=None,
        candidates_per_event=1, max_pairs_per_video=10,
        min_gap_sec=0.1, max_gap_sec=120.0,
        lam=0.01, sim_weight=0.8, time_weight=0.2, top_k=10)
    assert res.chains == []
    assert "temporal_no_valid_gap" in res.warnings
