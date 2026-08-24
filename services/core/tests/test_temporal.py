"""Test ghép chuỗi TRAKE: đúng thứ tự, tôn trọng gap, top-K cặp mỗi video, gộp +
cắt top_k trên toàn bộ pool.

Xem docs/superpowers/specs/2026-08-24-temporal-search-design.md. Dùng lại fixture `snap`
của conftest.py (200 row, 4 video luân phiên theo i%4, keyframe_time = i*0.2): row r nằm
ở video L26_V{r%4:03d}, thời điểm r*0.2s.
"""
import numpy as np
import pytest
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


@pytest.fixture
def truth(snap):
    return np.asarray(snap.refine, dtype=np.float32)


def test_caps_pairs_per_video(snap, truth):
    """V000 has 50 rows (0,4,8,...,196). candidates_per_event=200 (the whole
    corpus) makes EVERY row of V000 a candidate for both events, deterministically
    (membership doesn't depend on similarity ranking when top_k >= corpus size) --
    so V000 has far more valid (row1,row2) pairs than a small cap. Only the top
    `cap` by score should survive for that video."""
    q1, q2 = q(snap, 4), q(snap, 12)
    cap = 3
    lam, sim_w, time_w, min_gap, max_gap = 0.01, 0.8, 0.2, 0.1, 120.0

    res = T.search_temporal(
        snap, q1, q2, tags=None,
        candidates_per_event=200, max_pairs_per_video=cap,
        min_gap_sec=min_gap, max_gap_sec=max_gap,
        lam=lam, sim_weight=sim_w, time_weight=time_w, top_k=1000)

    v000_rows = np.arange(0, 200, 4)  # every row of L26_V000
    times = v000_rows * 0.2
    sim1 = truth[v000_rows] @ q1
    sim2 = truth[v000_rows] @ q2

    expected = []
    for i, r1 in enumerate(v000_rows):
        for j, r2 in enumerate(v000_rows):
            dt = times[j] - times[i]
            if dt < min_gap or dt > max_gap:
                continue
            decay = np.exp(-lam * (dt - min_gap))
            score = sim_w * (sim1[i] + sim2[j]) + time_w * decay
            expected.append((score, int(r1), int(r2)))
    expected.sort(key=lambda x: -x[0])
    want = [(r1, r2) for _, r1, r2 in expected[:cap]]

    v000_chains = [c for c in res.chains if c.video_name == "L26_V000"]
    assert len(v000_chains) == cap
    got = {(c.row1, c.row2) for c in v000_chains}
    assert got == set(want)


def test_pools_and_cuts_globally_across_videos(snap, truth):
    """candidates_per_event=200 makes ALL 4 videos have valid pairs (same
    reasoning as above, applied to every video). With a per-video cap of 2 and
    a small global top_k, the final result must be the true top-`top_k` pairs
    across the WHOLE pool (possibly multiple per video), not "top_k videos each
    contributing one" or any other per-video-first cut."""
    q1, q2 = q(snap, 4), q(snap, 12)
    cap = 2
    top_k = 5
    lam, sim_w, time_w, min_gap, max_gap = 0.01, 0.8, 0.2, 0.1, 120.0

    res = T.search_temporal(
        snap, q1, q2, tags=None,
        candidates_per_event=200, max_pairs_per_video=cap,
        min_gap_sec=min_gap, max_gap_sec=max_gap,
        lam=lam, sim_weight=sim_w, time_weight=time_w, top_k=top_k)

    sim1_all = truth @ q1
    sim2_all = truth @ q2
    all_rows = np.arange(200)
    video_of = all_rows % 4
    times_all = all_rows * 0.2

    per_video_top: list[tuple] = []
    for v in range(4):
        rows_v = all_rows[video_of == v]
        pairs = []
        for r1 in rows_v:
            for r2 in rows_v:
                dt = times_all[r2] - times_all[r1]
                if dt < min_gap or dt > max_gap:
                    continue
                decay = np.exp(-lam * (dt - min_gap))
                score = sim_w * (sim1_all[r1] + sim2_all[r2]) + time_w * decay
                pairs.append((score, int(r1), int(r2)))
        pairs.sort(key=lambda x: -x[0])
        per_video_top.extend(pairs[:cap])

    per_video_top.sort(key=lambda x: -x[0])
    want = [(r1, r2) for _, r1, r2 in per_video_top[:top_k]]

    assert len(res.chains) == top_k
    got = [(c.row1, c.row2) for c in res.chains]
    assert got == want
