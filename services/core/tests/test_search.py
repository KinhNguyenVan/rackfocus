"""Test 2-tier: recall trong subset, thứ hạng sau rerank, lọc tag, tombstone, diversity.

Xem docs/search-design.md §4. Ba tính chất then chốt được kiểm ở đây:

- EXACT_SUBSET phải khớp CHÍNH XÁC brute-force trên cùng refine store — nếu không thì
  "recall 100% trong subset" chỉ là lời nói.
- Tombstone lọc TRƯỚC khi rank (lọc sau thì top_k=10 với 3 điểm đã xoá trả về 7).
- `IDSelector` của FAISS thật sự giới hạn được kết quả trong tag đã chọn.
"""

from collections import Counter

import numpy as np
import pytest
from conftest import DIM, N
from searchcore import search as S


@pytest.fixture
def truth(snap):
    """Vector fp32 đọc từ CHÍNH refine store — ground truth phải cùng nguồn với cái đang
    được test, không phải fp32 gốc (fp32 không nằm trong snapshot)."""
    return np.asarray(snap.refine, dtype=np.float32)


def query(snap, row=0):
    return np.asarray(snap.refine[row], dtype=np.float32)


# --------------------------- choose_strategy ---------------------------
@pytest.mark.parametrize("cand,universe,exact_max,expect", [
    (500, 10_000, 20_000, S.STRATEGY_EXACT_SUBSET),
    (20_000, 100_000, 20_000, S.STRATEGY_EXACT_SUBSET),
    (20_001, 100_000, 20_000, S.STRATEGY_PRE),
    (100_000, 100_000, 20_000, S.STRATEGY_PRE),
])
def test_choose_strategy(cand, universe, exact_max, expect):
    assert S.choose_strategy(cand, universe, exact_max) == expect


def test_full_candidate_set_skips_filter():
    """Chọn HẾT tag -> candidate = N. Không short-circuit thì còn chậm hơn không filter,
    vì thêm selector match-everything vào mỗi bước duyệt đồ thị."""
    assert S.choose_strategy(1000, 1000, 20_000) == S.STRATEGY_PRE


def test_requested_strategy_wins():
    assert S.choose_strategy(10, 1000, 20_000, S.STRATEGY_POST) == S.STRATEGY_POST


# --------------------------- EXACT_SUBSET ---------------------------
@pytest.mark.parametrize("tags", [[0], [0, 3], [1, 2, 4]])
def test_exact_subset_matches_brute_force(snap, truth, tags):
    q = query(snap)
    res = S.search(snap, q, top_k=10, tags=tags, exact_max=10**9)
    assert res.strategy == S.STRATEGY_EXACT_SUBSET

    allow = np.where(np.isin(snap.tags, tags))[0]
    want = allow[np.argsort(-(truth[allow] @ q))[:10]]
    np.testing.assert_array_equal(res.rows, want)


def test_exact_subset_never_leaks_rows_outside_tags(snap):
    res = S.search(snap, query(snap), top_k=10, tags=[1], exact_max=10**9)
    assert set(snap.tags[res.rows].tolist()) == {1}


def test_candidate_count_matches_csr(snap):
    res = S.search(snap, query(snap), top_k=5, tags=[0, 2], exact_max=10**9)
    assert res.candidate_count == snap.tag_counts[0] + snap.tag_counts[2]
    assert res.tags_used == (0, 2)


def test_no_tags_searches_whole_corpus(snap):
    res = S.search(snap, query(snap), top_k=5, tags=None)
    assert res.candidate_count == N
    assert res.tags_used == ()


def test_forced_exact_subset_without_tags_brute_forces_whole_corpus(snap, truth):
    """UI 'exact': không tag (cand=None) + requested_strategy=EXACT_SUBSET ép brute-force
    TOÀN corpus, bỏ qua HNSW hoàn toàn. cand=None trước đây chỉ là sentinel cho nhánh
    HNSW -- ép EXACT_SUBSET không tag từng crash ở _rerank(snap, None, ...)."""
    q = query(snap)
    res = S.search(snap, q, top_k=10, tags=None,
                   requested_strategy=S.STRATEGY_EXACT_SUBSET)
    assert res.strategy == S.STRATEGY_EXACT_SUBSET

    want = np.argsort(-(truth @ q))[:10]
    np.testing.assert_array_equal(res.rows, want)


def test_forced_exact_subset_without_tags_still_filters_tombstone(snap):
    tomb = np.zeros((N + 7) // 8, dtype=np.uint8)
    tomb[0] |= np.uint8(1)  # row 0 đã xoá
    snap.tombstone = tomb

    res = S.search(snap, query(snap, row=0), top_k=N,
                   tags=None, requested_strategy=S.STRATEGY_EXACT_SUBSET)
    assert 0 not in res.rows.tolist()


# --------------------------- tombstone ---------------------------
def test_tombstone_filtered_before_ranking(snap, truth):
    """Xoá 3 row đứng đầu -> vẫn phải trả ĐỦ top_k, và không có row nào đã xoá."""
    q = query(snap)
    top = np.argsort(-(truth @ q))[:3]

    tomb = np.zeros((N + 7) // 8, dtype=np.uint8)
    for r in top:
        tomb[r >> 3] |= np.uint8(1 << (r & 7))
    snap.tombstone = tomb

    res = S.search(snap, q, top_k=10, tags=None, rerank_candidates=N)
    assert len(res.rows) == 10, "lọc SAU rank sẽ trả về 7"
    assert not set(res.rows.tolist()) & set(top.tolist())


def test_tombstone_applies_to_tagged_path(snap):
    tagged = snap.rows_for_tags([0])
    tomb = np.zeros((N + 7) // 8, dtype=np.uint8)
    for r in tagged[:5]:
        tomb[r >> 3] |= np.uint8(1 << (r & 7))
    snap.tombstone = tomb

    res = S.search(snap, query(snap), top_k=5, tags=[0], exact_max=10**9)
    assert res.candidate_count == len(tagged) - 5
    assert not set(res.rows.tolist()) & set(tagged[:5].tolist())


# --------------------------- HNSW + IDSelector ---------------------------
def test_hnsw_with_id_selector_confines_to_tags(snap):
    """Nhánh candidate lớn dùng IDSelectorBitmap — tức bitset quay lại trong hot path.

    Tuyên bố "vấn đề IdSet biến mất" CHỈ đúng ở nhánh EXACT_SUBSET.
    """
    res = S.search(snap, query(snap), top_k=5, tags=[0, 1],
                   exact_max=1, rerank_candidates=N)
    assert res.strategy == S.STRATEGY_PRE
    assert res.rows.size > 0
    assert set(snap.tags[res.rows].tolist()) <= {0, 1}


# --------------------------- fallback ---------------------------
def test_falls_back_to_untagged_when_tag_too_narrow(broken, monkeypatch):
    """Lọc tag là CỨNG: frame ngoài tag đã chọn không thể với tới ở mọi ef_search/top_k.

    Không có fallback thì LLM chọn sai tag cho một trang kết quả trông rất tự tin từ vùng
    corpus không chứa đáp án, và user không có cách nào phân biệt.
    """
    from conftest import DIM as D
    from conftest import ENCODER_NAME
    from searchcore.snapshot import TAGS_SENTINEL, Snapshot

    tags = np.full(N, TAGS_SENTINEL, dtype=np.uint16)
    tags[:3] = 2
    snap = Snapshot(broken(tags=tags), dim=D, encoder_name=ENCODER_NAME)

    res = S.search_with_fallback(snap, query(snap), top_k=10, tags=[2], exact_max=10**9)
    assert "tag_fallback" in res.warnings
    assert res.tags_used == ()
    assert len(res.rows) == 10


def test_no_fallback_when_tag_returns_enough(snap):
    res = S.search_with_fallback(snap, query(snap), top_k=5, tags=[0], exact_max=10**9)
    assert "tag_fallback" not in res.warnings
    assert res.tags_used == (0,)


# --------------------------- diversity ---------------------------
def test_diversify_caps_hits_per_video(snap):
    res = S.search(snap, query(snap), top_k=50, tags=None, rerank_candidates=N)
    rows, _ = S.diversify(snap, res.rows, res.scores, top_k=10, max_per_video=2)
    names = Counter(snap.payload.column("video_name").take(rows).to_pylist())
    assert max(names.values()) <= 2


def test_diversify_respects_time_gap(snap):
    res = S.search(snap, query(snap), top_k=50, tags=None, rerank_candidates=N)
    rows, _ = S.diversify(snap, res.rows, res.scores, top_k=10, min_time_gap_sec=5.0)
    times = snap.payload.column("keyframe_time").take(rows).to_pylist()
    names = snap.payload.column("video_name").take(rows).to_pylist()
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            if names[i] == names[j]:
                assert abs(times[i] - times[j]) >= 5.0


def test_diversify_noop_without_constraints(snap):
    res = S.search(snap, query(snap), top_k=20, tags=None, rerank_candidates=N)
    rows, _ = S.diversify(snap, res.rows, res.scores, top_k=5)
    np.testing.assert_array_equal(rows, res.rows[:5])


# --------------------------- lỗi đầu vào ---------------------------
def test_wrong_query_dim_raises(snap):
    with pytest.raises(ValueError, match="query dim"):
        S.search(snap, np.zeros(DIM + 1, dtype=np.float32), top_k=5)


def test_unknown_tag_propagates_as_value_error(snap):
    with pytest.raises(ValueError, match="ngoài phạm vi"):
        S.search(snap, query(snap), top_k=5, tags=[9999])
