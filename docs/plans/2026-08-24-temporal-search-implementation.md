# Temporal Search (TRAKE) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Wire up the already-stubbed `SearchTemporal` gRPC RPC end-to-end so a user can
type two ordered event descriptions and get back ranked candidate video chains, with a
FE toggle to switch between normal (KIS) and temporal search.

**Architecture:** BE encodes + (optionally) LLM-enriches both event texts in parallel,
unions their tags into one shared `Filter` (the proto only has one `Filter` per temporal
request, not one per event), and sends two pre-computed vectors to core's new
`SearchTemporal` handler. Core reuses the *existing* `search_with_fallback()` pipeline
per event to get a bounded candidate pool, joins both pools by `video_name`, and keeps
the single best-scoring valid (order + gap constrained) pair per video.

**Tech Stack:** Python (core: FAISS/gRPC/numpy/pyarrow, be: FastAPI/pydantic/grpc.aio),
TypeScript/React (fe), existing proto (no changes needed).

**Design doc:** `docs/superpowers/specs/2026-08-24-temporal-search-design.md` — read
this first if anything below is ambiguous; it has the full rationale.

---

## Before you start

Run `make proto` from the repo root once (regenerates gRPC stubs — they're gitignored
build artifacts, and `SearchTemporal`/`TemporalEvent`/`TemporalChain` already exist in
`proto/searchcore/v1/search.proto`, so no proto edits are needed, just stub generation).
Without this, `from .pb.searchcore.v1 import search_pb2 as pb` won't have
`pb.TemporalEvent` etc. available.

```bash
make proto
```

---

## Phase 1 — Core: config

### Task 1: Add `TRAKE_*` config knobs to `searchcore/config.py`

**Files:**
- Modify: `services/core/src/searchcore/config.py`
- Modify: `.env.example`

**Step 1: Add a `_float` helper**

There's `_int`/`_bool` already but no float helper (all existing numeric config is
integer). Add, right after `_bool`:

```python
def _float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))
```

**Step 2: Add the TRAKE fields to `Config`**

Insert after the `exact_subset_max` field (before the "── Đồng thời" comment block):

```python
    # ── TRAKE (docs/superpowers/specs/2026-08-24-temporal-search-design.md) ──────
    # Top-K candidate rows per event BEFORE joining by video — bounds join cost,
    # not the whole corpus (a full brute-force rerank per event was already measured
    # as too slow for a single query; doing it twice would be worse).
    trake_candidates_per_event: int = _int("TRAKE_CANDIDATES_PER_EVENT", 500)
    # Hard floor on (t2 - t1): pairs closer together (including negative = order
    # violated) are excluded entirely, never scored. Also doubles as the decay's
    # zero-penalty reference point.
    trake_min_gap_sec: float = _float("TRAKE_MIN_GAP_SEC", 5.0)
    # Hard ceiling on (t2 - t1): pairs further apart are excluded entirely.
    trake_max_gap_sec: float = _float("TRAKE_MAX_GAP_SEC", 120.0)
    trake_lambda: float = _float("TRAKE_LAMBDA", 0.00557)
    trake_sim_weight: float = _float("TRAKE_SIM_WEIGHT", 0.8)
    trake_time_weight: float = _float("TRAKE_TIME_WEIGHT", 0.2)
    trake_top_k_chains: int = _int("TRAKE_TOP_K_CHAINS", 20)
```

**Step 3: Add the same vars to `.env.example`**

Insert after the `EXACT_SUBSET_MAX=20000` line:

```bash
TRAKE_CANDIDATES_PER_EVENT=500
TRAKE_MIN_GAP_SEC=5.0
TRAKE_MAX_GAP_SEC=120.0
TRAKE_LAMBDA=0.00557
TRAKE_SIM_WEIGHT=0.8
TRAKE_TIME_WEIGHT=0.2
TRAKE_TOP_K_CHAINS=20
```

**Step 4: Verify it imports cleanly**

Run: `cd services/core && python -c "from src.searchcore.config import Config; c = Config(); print(c.trake_min_gap_sec, c.trake_lambda)"`

Expected: `5.0 0.00557` (no errors).

**Step 5: Commit**

```bash
git add services/core/src/searchcore/config.py .env.example
git commit -m "feat(core): add TRAKE_* config knobs for temporal search"
```

---

## Phase 2 — Core: matching algorithm (`searchcore/temporal.py`)

This replaces the 1-line docstring stub. Built incrementally, one behavior per task,
all driven by `services/core/tests/test_temporal.py` (currently also a 1-line stub).

### Task 2: Scaffolding + "pair within gap bounds gets scored"

**Files:**
- Modify: `services/core/src/searchcore/temporal.py`
- Modify: `services/core/tests/test_temporal.py`

**Step 1: Write the failing test**

Replace the whole content of `services/core/tests/test_temporal.py` with:

```python
"""Test ghép chuỗi TRAKE: đúng thứ tự, tôn trọng gap, một cặp tốt nhất mỗi video.

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
    # row 4 (V000, t=0.8s), row 12 (V000, t=2.4s) -> dt=1.6s
    res = T.search_temporal(
        snap, q(snap, 4), q(snap, 12), tags=None,
        candidates_per_event=50, min_gap_sec=0.1, max_gap_sec=5.0,
        lam=0.01, sim_weight=0.8, time_weight=0.2, top_k=10)

    assert len(res.chains) == 1
    c = res.chains[0]
    assert c.video_name == "L26_V000"
    assert c.row1 == 4
    assert c.row2 == 12
    assert c.t1 == 0.8 and c.t2 == 2.4
    assert c.score > 0
    assert res.warnings == []
```

**Step 2: Run test to verify it fails**

Run: `cd services/core && pytest tests/test_temporal.py -v`
Expected: FAIL (`AttributeError: module 'searchcore.temporal' has no attribute
'search_temporal'` or import error, since the file is still just a docstring).

**Step 3: Write minimal implementation**

Replace `services/core/src/searchcore/temporal.py`:

```python
"""TRAKE: search 2 sự kiện rồi ghép cặp theo thứ tự thời gian + giới hạn khoảng cách.

Xem docs/superpowers/specs/2026-08-24-temporal-search-design.md. Tái dùng
search_with_fallback cho từng event (KHÔNG brute-force toàn corpus — đã đo là quá chậm
nếu làm hai lần) rồi ghép theo video_name — mỗi video giữ đúng MỘT cặp tốt nhất, không
làm lại bước merge-overlapping-pairs của bản notebook (giải quyết bài toán khác: phân
tích khám phá nhiều cặp chồng lấn, không phải endpoint xếp hạng kết quả).
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
                    candidates_per_event: int, min_gap_sec: float, max_gap_sec: float,
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

    chains: list[Chain] = []
    for video in common:
        best: Chain | None = None
        for row1, sim1, time1 in by_video1[video]:
            for row2, sim2, time2 in by_video2[video]:
                dt = time2 - time1
                if dt < min_gap_sec or dt > max_gap_sec:
                    continue
                decay = float(np.exp(-lam * (dt - min_gap_sec)))
                score = sim_weight * (sim1 + sim2) + time_weight * decay
                if best is None or score > best.score:
                    best = Chain(video_name=video, row1=row1, row2=row2, score=score,
                                sim1=sim1, sim2=sim2, t1=time1, t2=time2)
        if best is not None:
            chains.append(best)

    if not chains:
        warnings.append("temporal_no_valid_gap")

    chains.sort(key=lambda c: -c.score)
    t["join_ms"] = (time.perf_counter() - t0) * 1000
    return TemporalResult(chains=chains[:top_k], warnings=warnings, timings_ms=t)
```

**Step 4: Run test to verify it passes**

Run: `cd services/core && pytest tests/test_temporal.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add services/core/src/searchcore/temporal.py services/core/tests/test_temporal.py
git commit -m "feat(core): implement TRAKE pair scoring within gap bounds"
```

### Task 3: Gap filter excludes pairs below min / above max

**Files:**
- Modify: `services/core/tests/test_temporal.py` (add tests, no implementation change expected)

**Step 1: Add the failing tests**

Append to `services/core/tests/test_temporal.py`:

```python
def test_pair_below_min_gap_excluded(snap):
    # same rows as above (dt=1.6s), but min_gap_sec=5.0 excludes it
    res = T.search_temporal(
        snap, q(snap, 4), q(snap, 12), tags=None,
        candidates_per_event=50, min_gap_sec=5.0, max_gap_sec=120.0,
        lam=0.01, sim_weight=0.8, time_weight=0.2, top_k=10)
    assert res.chains == []
    assert "temporal_no_valid_gap" in res.warnings


def test_pair_above_max_gap_excluded(snap):
    res = T.search_temporal(
        snap, q(snap, 4), q(snap, 12), tags=None,
        candidates_per_event=50, min_gap_sec=0.1, max_gap_sec=1.0,
        lam=0.01, sim_weight=0.8, time_weight=0.2, top_k=10)
    assert res.chains == []
    assert "temporal_no_valid_gap" in res.warnings


def test_order_violation_excluded_via_min_gap(snap):
    # event1 query matches the LATER row (12, t=2.4), event2 matches the EARLIER
    # row (4, t=0.8) -> dt = 0.8 - 2.4 = -1.6, always < a positive min_gap_sec
    res = T.search_temporal(
        snap, q(snap, 12), q(snap, 4), tags=None,
        candidates_per_event=50, min_gap_sec=0.1, max_gap_sec=120.0,
        lam=0.01, sim_weight=0.8, time_weight=0.2, top_k=10)
    assert res.chains == []
    assert "temporal_no_valid_gap" in res.warnings
```

**Step 2: Run to verify these already pass**

Run: `cd services/core && pytest tests/test_temporal.py -v`
Expected: PASS (Task 2's implementation already handles this — the gap filter was
built in from the start since it's core to the algorithm). If any of these fail,
the filter logic in `search_temporal` has a bug — fix it now before moving on.

**Step 3: Commit**

```bash
git add services/core/tests/test_temporal.py
git commit -m "test(core): cover TRAKE gap-filter and order-enforcement edge cases"
```

### Task 4: Best pair per video (not all valid pairs)

**Files:**
- Modify: `services/core/tests/test_temporal.py`

**Step 1: Write the failing test**

Append:

```python
def test_best_pair_per_video_selected_when_multiple_valid(snap):
    """Video V000 có nhiều row (0,4,8,...,196). event1 khớp mạnh nhất với row 4 (t=0.8)
    nhưng cũng có candidate ở row 0 (t=0.0); event2 khớp row 12 (t=2.4) và row 8
    (t=1.6). Cả (4,12) và (0,8) đều hợp lệ (dt=1.6 và 1.6) nhưng chỉ 1 chain/video
    được trả về -- chain có sim cao hơn thắng."""
    # Dùng chính row 4 làm query 1 -> top match tất định là row 4 (sim=1.0), các
    # row khác cùng video có sim thấp hơn nhưng vẫn > 0 vì vector ngẫu nhiên chuẩn hoá.
    res = T.search_temporal(
        snap, q(snap, 4), q(snap, 12), tags=None,
        candidates_per_event=200, min_gap_sec=0.1, max_gap_sec=5.0,
        lam=0.01, sim_weight=0.8, time_weight=0.2, top_k=10)

    v000_chains = [c for c in res.chains if c.video_name == "L26_V000"]
    assert len(v000_chains) == 1
    # sim1=1.0 đúng ở row=4 (query trùng khít) nên đây phải là cặp thắng.
    assert v000_chains[0].row1 == 4
```

**Step 2: Run to verify it fails or passes**

Run: `cd services/core && pytest tests/test_temporal.py -v`
Expected: PASS — the `if best is None or score > best.score` logic from Task 2
already enforces one-best-per-video. This test exists to lock that behavior in and
catch a future regression (e.g. someone "fixing" it to return all pairs). If it
fails, the dedup logic is broken — fix `search_temporal` before continuing.

**Step 3: Commit**

```bash
git add services/core/tests/test_temporal.py
git commit -m "test(core): lock in one-best-chain-per-video behavior"
```

### Task 5: No-common-video and no-candidates warnings

**Files:**
- Modify: `services/core/tests/test_temporal.py`

**Step 1: Write the failing tests**

Append:

```python
def test_no_common_video_returns_empty_with_warning(snap):
    # candidates_per_event=1 -> event1's only candidate is its own row (video
    # of row 0 = L26_V000), event2's only candidate is row 1 (video L26_V001).
    res = T.search_temporal(
        snap, q(snap, 0), q(snap, 1), tags=None,
        candidates_per_event=1, min_gap_sec=0.0, max_gap_sec=120.0,
        lam=0.01, sim_weight=0.8, time_weight=0.2, top_k=10)
    assert res.chains == []
    assert "temporal_no_common_video" in res.warnings


def test_no_candidates_for_event_returns_empty_with_warning(snap, monkeypatch):
    from searchcore.search import SearchResult

    empty = SearchResult(rows=np.empty(0, dtype=np.int64), scores=np.empty(0, dtype=np.float32))

    def fake_search_with_fallback(snap_, qvec, **kw):
        # trả rỗng cho MỌI lời gọi thứ hai trở đi (event2), giữ event1 bình thường
        fake_search_with_fallback.calls += 1
        if fake_search_with_fallback.calls == 2:
            return empty
        from searchcore.search import search_with_fallback as real
        return real(snap_, qvec, **kw)

    fake_search_with_fallback.calls = 0
    monkeypatch.setattr(T, "search_with_fallback", fake_search_with_fallback)

    res = T.search_temporal(
        snap, q(snap, 4), q(snap, 12), tags=None,
        candidates_per_event=50, min_gap_sec=0.1, max_gap_sec=5.0,
        lam=0.01, sim_weight=0.8, time_weight=0.2, top_k=10)
    assert res.chains == []
    assert "temporal_no_candidates_event2" in res.warnings
```

**Step 2: Run test to verify it fails first**

Run: `cd services/core && pytest tests/test_temporal.py -v`
Expected: `test_no_common_video_returns_empty_with_warning` should already PASS
(same reasoning as Task 4). `test_no_candidates_for_event_returns_empty_with_warning`
should also PASS since `monkeypatch.setattr(T, "search_with_fallback", ...)` patches
the name `temporal.search_with_fallback` that `search_temporal()` calls (it was
imported via `from .search import search_with_fallback` in Task 2, so it's a module
attribute of `temporal.py` and patchable this way).

If `test_no_candidates_for_event_returns_empty_with_warning` fails because
`search_temporal()` doesn't check `r1`/`r2` emptiness before grouping — that
check already exists in Task 2's implementation, so this should pass immediately.

**Step 3: Commit**

```bash
git add services/core/tests/test_temporal.py
git commit -m "test(core): cover TRAKE no-common-video and no-candidates warnings"
```

---

## Phase 3 — Core: gRPC wiring (`SearchTemporal` RPC)

### Task 6: Wire `SearchTemporal` in `searchcore/server.py`

**Files:**
- Modify: `services/core/src/searchcore/server.py`
- Modify: `services/core/tests/test_server.py`

**Step 1: Write the failing test**

Append to `services/core/tests/test_server.py`:

First, add a fixture right after the existing `stubs` fixture in
`test_server.py` — the default `Config()` used by `stubs` has `TRAKE_MIN_GAP_SEC`
default to 5.0, but the `snap` fixture's rows only span 0.2s apart per row, so a
dedicated small-gap config is needed to exercise a real chain in this test corpus
(`Config` is `@dataclass(frozen=True)`, so overriding specific fields at
construction works out of the box):

```python
@pytest.fixture
def stubs_small_gap(snap):
    """Như `stubs` nhưng TRAKE_MIN_GAP_SEC nhỏ để test được với dt=1.6s của
    fixture snap có sẵn (video cách nhau keyframe_time=0.2s/row)."""
    holder = IndexHolder()
    holder.swap(snap)
    cfg = Config(trake_min_gap_sec=0.1, trake_max_gap_sec=5.0)
    sock = f"unix://{tempfile.mkdtemp()}/sc.sock"

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    pb_grpc.add_SearchCoreServiceServicer_to_server(
        SearchCoreServiceServicer(holder, FakeEncoder(), cfg), server)
    server.add_insecure_port(sock)
    server.start()

    channel = grpc.insecure_channel(sock)
    yield pb_grpc.SearchCoreServiceStub(channel), snap
    channel.close()
    server.stop(0)
```

Then append the tests themselves:

```python
# --------------------------- SearchTemporal ---------------------------
def test_search_temporal_returns_chain_with_two_hits(stubs_small_gap):
    search, snap = stubs_small_gap
    r = search.SearchTemporal(pb.SearchTemporalRequest(
        events=[
            pb.TemporalEvent(query=[pb.QueryPart(vector=_q(snap, 4))]),
            pb.TemporalEvent(query=[pb.QueryPart(vector=_q(snap, 12))]),
        ],
        with_payload=True))

    assert len(r.chains) == 1
    c = r.chains[0]
    assert len(c.hits) == 2
    assert c.hits[0].payload.video_name == "L26_V000"
    assert c.hits[0].index_row == 4
    assert c.hits[1].index_row == 12
    assert c.span_sec == pytest.approx(1.6, abs=1e-6)
    assert c.score > 0


def test_search_temporal_rejects_wrong_event_count(stubs):
    search, _, snap = stubs
    with pytest.raises(grpc.RpcError) as ex:
        search.SearchTemporal(pb.SearchTemporalRequest(
            events=[pb.TemporalEvent(query=[pb.QueryPart(vector=_q(snap, 0))])]))
    assert ex.value.code() == grpc.StatusCode.INVALID_ARGUMENT


def test_search_temporal_unavailable_before_snapshot_loaded():
    holder = IndexHolder()
    sock = f"unix://{tempfile.mkdtemp()}/sc.sock"
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    pb_grpc.add_SearchCoreServiceServicer_to_server(
        SearchCoreServiceServicer(holder, None, Config()), server)
    server.add_insecure_port(sock)
    server.start()
    try:
        ch = grpc.insecure_channel(sock)
        with pytest.raises(grpc.RpcError) as ex:
            pb_grpc.SearchCoreServiceStub(ch).SearchTemporal(
                pb.SearchTemporalRequest(events=[
                    pb.TemporalEvent(query=[pb.QueryPart(text="x")]),
                    pb.TemporalEvent(query=[pb.QueryPart(text="y")]),
                ]))
        assert ex.value.code() == grpc.StatusCode.UNAVAILABLE
        ch.close()
    finally:
        server.stop(0)
```

**Step 2: Run test to verify it fails**

Run: `cd services/core && pytest tests/test_server.py -v -k temporal`
Expected: FAIL with `AttributeError: 'SearchCoreServiceServicer' object has no
attribute 'SearchTemporal'` (the generated grpc base class has a default
UNIMPLEMENTED stub, so it'll actually fail with a `grpc.RpcError` /
`StatusCode.UNIMPLEMENTED` instead — either way, not the behavior we want).

**Step 3: Write minimal implementation**

In `services/core/src/searchcore/server.py`:

1. Add the import near the top (alongside `from . import search as S`):

```python
from . import temporal as T
```

2. Add the `SearchTemporal` method to `SearchCoreServiceServicer`, right after the
   existing `Search` method (before `Encode`):

```python
    def SearchTemporal(self, request, context):
        t_all = time.perf_counter()
        snap = self._snap_or_abort(context)

        if len(request.events) != 2:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT,
                          "v1 SearchTemporal chỉ nhận đúng 2 event")

        try:
            qvec1 = _query_vector(request.events[0], self.encoder, snap)
            qvec2 = _query_vector(request.events[1], self.encoder, snap)
        except ValueError as ex:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(ex))

        params = request.params
        tags = list(request.filter.tags) if request.HasField("filter") else []
        requested_strategy = request.filter.strategy if request.HasField("filter") else 0
        cfg = self.cfg

        res = T.search_temporal(
            snap, qvec1, qvec2, tags=tags,
            candidates_per_event=cfg.trake_candidates_per_event if cfg else 500,
            min_gap_sec=cfg.trake_min_gap_sec if cfg else 5.0,
            max_gap_sec=cfg.trake_max_gap_sec if cfg else 120.0,
            lam=cfg.trake_lambda if cfg else 0.00557,
            sim_weight=cfg.trake_sim_weight if cfg else 0.8,
            time_weight=cfg.trake_time_weight if cfg else 0.2,
            top_k=min(request.top_k, cfg.trake_top_k_chains if cfg else 20)
                if request.top_k else (cfg.trake_top_k_chains if cfg else 20),
            exact_max=cfg.exact_subset_max if cfg else 20_000,
            rerank_candidates=params.rerank_candidates or (cfg.rerank_candidates if cfg else 800),
            requested_strategy=requested_strategy,
        )

        chains = []
        for c in res.chains:
            rows = np.array([c.row1, c.row2])
            payloads = _build_payloads(snap, rows) if request.with_payload else [None, None]
            hits = [
                cpb.SearchHit(
                    id=int(snap.idmap[c.row1]), index_row=int(c.row1),
                    score_exact=float(c.sim1), score_fused=float(c.sim1), rank=0,
                    tier=TIER_KEYFRAME,
                    **({"payload": payloads[0]} if payloads[0] is not None else {})),
                cpb.SearchHit(
                    id=int(snap.idmap[c.row2]), index_row=int(c.row2),
                    score_exact=float(c.sim2), score_fused=float(c.sim2), rank=1,
                    tier=TIER_KEYFRAME,
                    **({"payload": payloads[1]} if payloads[1] is not None else {})),
            ]
            chains.append(pb.TemporalChain(
                hits=hits, score=c.score, span_sec=c.t2 - c.t1))

        total_ms = (time.perf_counter() - t_all) * 1000
        metrics.incr("temporal_queries_total")
        return pb.SearchTemporalResponse(
            chains=chains,
            meta=cpb.ResponseMeta(
                request_id=request.ctx.request_id, snapshot_ver=snap.version,
                timings=cpb.Timings(total_ms=total_ms), warnings=res.warnings),
        )
```

Note: `_build_payloads(snap, rows)` returns a list aligned with `rows`, so
`payloads[0]`/`payloads[1]` correspond to `c.row1`/`c.row2` respectively — this
reuses the existing helper unmodified.

**Step 4: Run test to verify it passes**

Run: `cd services/core && pytest tests/test_server.py -v -k temporal`
Expected: PASS (all 3 new tests)

**Step 5: Run the full core test suite to check nothing broke**

Run: `cd services/core && pytest -q`
Expected: all pass (previous test count + new temporal tests)

**Step 6: Commit**

```bash
git add services/core/src/searchcore/server.py services/core/tests/test_server.py
git commit -m "feat(core): wire SearchTemporal gRPC handler to temporal.search_temporal"
```

---

## Phase 4 — BE: API endpoint

### Task 7: Add `search_temporal()` to `clients/searchcore.py`

**Files:**
- Modify: `services/be/src/app/clients/searchcore.py`

**Step 1: Write the failing test** (this one is tested indirectly through Task 8's
endpoint test, since `clients/searchcore.py` has no existing dedicated unit test file
— it's exercised through `api/search.py`'s tests via the real gRPC test server. Skip
straight to implementation; Task 8's test will catch any mistake here.)

**Step 2: Add the function**

Append to `services/be/src/app/clients/searchcore.py` (after the existing `search()`
function):

```python
async def search_temporal(vec1, vec2, *, tags=None, request_id: str = "",
                          exact: bool = False, top_k: int | None = None,
                          with_payload: bool = True, timeout: float = 2.0):
    """SearchTemporalRequest chỉ có MỘT Filter cho cả request (không phải 1/event) --
    `tags` ở đây phải là hợp của tag cả 2 event (BE tự union trước khi gọi hàm này).
    `exact=True` áp cho CẢ hai nhánh tìm candidate mỗi event, giống search() thường."""
    req = pb.SearchTemporalRequest(
        ctx=_ctx(request_id),
        events=[
            pb.TemporalEvent(query=[pb.QueryPart(vector=cpb.Vector(values=list(vec1)))]),
            pb.TemporalEvent(query=[pb.QueryPart(vector=cpb.Vector(values=list(vec2)))]),
        ],
        top_k=top_k or 0,
        filter=cpb.Filter(
            tags=list(tags or []),
            strategy=cpb.FILTER_STRATEGY_EXACT_SUBSET if exact else cpb.FILTER_STRATEGY_UNSPECIFIED,
        ),
        with_payload=with_payload,
    )
    return await _search.SearchTemporal(req, timeout=timeout)
```

**Step 3: Commit**

```bash
git add services/be/src/app/clients/searchcore.py
git commit -m "feat(be): add search_temporal() gRPC client function"
```

### Task 8: New `POST /api/search/temporal` endpoint

**Files:**
- Create: `services/be/src/app/api/search_temporal.py`
- Modify: `services/be/src/app/api/router.py`
- Create: `services/be/tests/test_search_temporal.py`

**Step 1: Write the failing test**

Create `services/be/tests/test_search_temporal.py`:

```python
"""Test POST /api/search/temporal: 2 event, union tag khi bật LLM, chain 2 hit.

Xem docs/superpowers/specs/2026-08-24-temporal-search-design.md. conftest.py's snap
fixture: 200 row, video luân phiên theo i%4, keyframe_time=i*0.2s/row.
"""


def search_temporal(client, **body):
    body.setdefault("event1", "người đàn ông cầm micro")
    body.setdefault("event2", "khán giả vỗ tay")
    r = client.post("/api/search/temporal", json=body)
    assert r.status_code == 200, r.text
    return r.json()


def test_temporal_search_returns_chains_with_two_hits_each(client, llm):
    d = search_temporal(client)
    assert isinstance(d["chains"], list)
    if d["chains"]:
        c = d["chains"][0]
        assert len(c["hits"]) == 2
        assert c["hits"][0]["video_name"] == c["video_name"]
        assert c["span_sec"] >= 0


def test_temporal_use_llm_false_skips_llm(client, llm):
    d = search_temporal(client, use_llm=False)
    assert llm.calls == 0
    assert "warnings" in d


def test_temporal_use_llm_true_unions_both_events_tags(client, llm):
    calls = []

    async def reply(**kwargs):
        text = kwargs["messages"][1]["content"]
        calls.append(text)
        payload = {"tags": [0]} if len(calls) == 1 else {"tags": [1]}
        import json
        import types
        msg = types.SimpleNamespace(content=json.dumps({**payload, "enriched": ""}))
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])

    import sys
    import types as _types
    sys.modules["litellm"] = _types.SimpleNamespace(acompletion=reply)

    d = search_temporal(client, use_llm=True)
    assert len(calls) == 2
    assert d  # phản hồi vẫn hợp lệ dù mỗi event ra tag khác nhau


def test_temporal_empty_event_rejected(client):
    assert client.post("/api/search/temporal",
                       json={"event1": "", "event2": "x"}).status_code == 422
```

**Step 2: Run test to verify it fails**

Run: `cd services/be && pytest tests/test_search_temporal.py -v`
Expected: FAIL with 404 (route doesn't exist yet)

**Step 3: Write the endpoint**

Create `services/be/src/app/api/search_temporal.py`:

```python
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

from .search import Hit
from ..clients import searchcore
from ..config import get_settings
from ..services import enrich as enrich_svc
from ..services import tagvocab

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
    snapshot_ver: str
    timings_ms: dict[str, float]


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
        chains=chains, warnings=warnings,
        snapshot_ver=resp.meta.snapshot_ver or snap_ver,
        timings_ms={
            "encode_and_enrich_parallel": round(parallel_ms, 2),
            "core_total": round(resp.meta.timings.total_ms, 2),
            "total": round(total_ms, 2),
        },
    )
```

Wire it into the router — modify `services/be/src/app/api/router.py`:

```python
"""Gom sub-router. /healthz + /readyz ở root (Caddyfile route riêng), còn lại /api."""
from fastapi import APIRouter

from . import browse, health, search, search_temporal

root_router = APIRouter()
root_router.include_router(health.router, tags=["health"])

api_router = APIRouter(prefix="/api")
api_router.include_router(search.router, tags=["search"])
api_router.include_router(search_temporal.router, tags=["search"])
api_router.include_router(browse.router, tags=["browse"])
```

**Step 4: Run test to verify it passes**

Run: `cd services/be && pytest tests/test_search_temporal.py -v`
Expected: PASS

**Step 5: Run the full BE test suite to check nothing broke**

Run: `cd services/be && pytest -q`
Expected: all pass

**Step 6: Commit**

```bash
git add services/be/src/app/api/search_temporal.py services/be/src/app/api/router.py \
        services/be/tests/test_search_temporal.py
git commit -m "feat(be): add POST /api/search/temporal endpoint"
```

---

## Phase 5 — FE: types, client, hook

### Task 9: FE types + API client function

**Files:**
- Modify: `services/fe/src/api/types.ts`
- Modify: `services/fe/src/api/client.ts`

**Step 1: Add types**

Append to `services/fe/src/api/types.ts`:

```typescript
// Khớp services/be/src/app/api/search_temporal.py.
export type TemporalSearchRequest = {
	event1: string;
	event2: string;
	use_llm?: boolean;
	exact?: boolean;
	top_k?: number;
};

export type TemporalChain = {
	video_name: string;
	score: number;
	span_sec: number;
	hits: SearchHit[]; // luôn đúng 2 phần tử, theo thứ tự event1 -> event2
};

export type TemporalSearchResponse = {
	chains: TemporalChain[];
	warnings: string[];
	snapshot_ver: string;
	timings_ms: Record<string, number>;
};
```

**Step 2: Add client function**

Append to `services/fe/src/api/client.ts`:

```typescript
import type { TemporalSearchRequest, TemporalSearchResponse } from "./types";

export async function searchTemporal(
	request: TemporalSearchRequest,
	signal?: AbortSignal,
): Promise<TemporalSearchResponse> {
	const response = await fetch("/api/search/temporal", {
		method: "POST",
		headers: { "content-type": "application/json" },
		body: JSON.stringify(request),
		signal,
	});

	if (!response.ok) {
		const detail = await response.text();
		throw new Error(detail || `Temporal search failed (${response.status})`);
	}

	return response.json() as Promise<TemporalSearchResponse>;
}
```

(Merge this import with the existing `import type { NeighborsResponse, SearchRequest,
SearchResponse } from "./types";` line at the top of the file into one combined
import statement rather than adding a second one.)

**Step 3: Verify it compiles**

Run: `cd services/fe && npm run build`
Expected: succeeds (runs `tsc --noEmit && vite build` — this is the only
"test" FE has today, per the repo's existing convention)

**Step 4: Commit**

```bash
git add services/fe/src/api/types.ts services/fe/src/api/client.ts
git commit -m "feat(fe): add temporal search API types and client function"
```

### Task 10: `useTemporalSearch` hook

**Files:**
- Create: `services/fe/src/hooks/useTemporalSearch.ts`

**Step 1: Write it**

Modeled directly on the existing `useSearch.ts` (same debounce/abort/request-id-race
pattern):

```typescript
import { useEffect, useRef, useState } from "react";
import { searchTemporal } from "../api/client";
import type { TemporalChain } from "../api/types";

export function useTemporalSearch(
	event1: string,
	event2: string,
	useLlm = false,
	exact = false,
) {
	const [chains, setChains] = useState<TemporalChain[]>([]);
	const [totalMs, setTotalMs] = useState<number | null>(null);
	const [warnings, setWarnings] = useState<string[]>([]);
	const [error, setError] = useState<string | null>(null);
	const [loading, setLoading] = useState(false);
	const requestId = useRef(0);

	useEffect(() => {
		const e1 = event1.trim();
		const e2 = event2.trim();
		if (!e1 || !e2) {
			setChains([]);
			setTotalMs(null);
			setWarnings([]);
			setError(null);
			setLoading(false);
			return;
		}

		const controller = new AbortController();
		const currentRequest = ++requestId.current;
		setLoading(true);
		setError(null);

		searchTemporal({ event1: e1, event2: e2, use_llm: useLlm, exact }, controller.signal)
			.then((data) => {
				if (currentRequest !== requestId.current) return;
				setChains(data.chains ?? []);
				setTotalMs(data.timings_ms?.total ?? null);
				setWarnings(data.warnings ?? []);
			})
			.catch((cause: unknown) => {
				if (controller.signal.aborted || currentRequest !== requestId.current) return;
				setError(cause instanceof Error ? cause.message : "Temporal search failed");
				setChains([]);
				setTotalMs(null);
				setWarnings([]);
			})
			.finally(() => {
				if (currentRequest === requestId.current) setLoading(false);
			});

		return () => controller.abort();
	}, [event1, event2, useLlm, exact]);

	return { chains, totalMs, warnings, error, loading };
}
```

**Step 2: Verify it compiles**

Run: `cd services/fe && npm run build`
Expected: succeeds

**Step 3: Commit**

```bash
git add services/fe/src/hooks/useTemporalSearch.ts
git commit -m "feat(fe): add useTemporalSearch hook"
```

---

## Phase 6 — FE: components + App.tsx wiring

### Task 11: `TemporalQueryBuilder` component

**Files:**
- Modify: `services/fe/src/components/TemporalQueryBuilder.tsx` (replace the 1-line stub)

**Step 1: Write it**

```tsx
type Props = {
	event1: string;
	event2: string;
	onEvent1Change: (value: string) => void;
	onEvent2Change: (value: string) => void;
};

// 2 ô nhập sự kiện có thứ tự cho TRAKE — thay ô query đơn của KIS khi bật temporal mode.
// Gap tối thiểu/tối đa giữa 2 sự kiện là cấu hình phía core (TRAKE_MIN_GAP_SEC/
// TRAKE_MAX_GAP_SEC), không phơi ra UI.
export function TemporalQueryBuilder({ event1, event2, onEvent1Change, onEvent2Change }: Props) {
	return (
		<div className="mb-3">
			<div className="mb-2">
				<label className="form-label small fw-semibold">Sự kiện 1 (xảy ra trước)</label>
				<input
					type="text"
					className="form-control"
					placeholder="VD: người đàn ông cầm micro phát biểu"
					value={event1}
					onChange={(e) => onEvent1Change(e.target.value)}
				/>
			</div>
			<div>
				<label className="form-label small fw-semibold">Sự kiện 2 (xảy ra sau)</label>
				<input
					type="text"
					className="form-control"
					placeholder="VD: khán giả đứng dậy vỗ tay"
					value={event2}
					onChange={(e) => onEvent2Change(e.target.value)}
				/>
			</div>
		</div>
	);
}
```

**Step 2: Verify it compiles**

Run: `cd services/fe && npm run build`
Expected: succeeds (component isn't imported anywhere yet, but it must type-check on
its own)

**Step 3: Commit**

```bash
git add services/fe/src/components/TemporalQueryBuilder.tsx
git commit -m "feat(fe): add TemporalQueryBuilder component"
```

### Task 12: `TemporalChainCard` component

**Files:**
- Create: `services/fe/src/components/TemporalChainCard.tsx`

**Step 1: Write it**

```tsx
import type { TemporalChain } from "../api/types";

type Props = {
	chain: TemporalChain;
	onUseChain: (chain: TemporalChain) => void;
};

// Một chain: video, điểm, khoảng cách thời gian, 2 thumbnail theo đúng thứ tự.
// "Dùng chuỗi này" đẩy cả 2 hit vào sidebar Selected có sẵn -- exportResult() nhánh
// "trake" đã kỳ vọng frame cùng 1 video trong `selected`, nên không cần đổi gì ở đó.
export function TemporalChainCard({ chain, onUseChain }: Props) {
	const [hit1, hit2] = chain.hits;
	return (
		<div className="card mb-2">
			<div className="card-body p-2">
				<div className="d-flex justify-content-between align-items-center mb-1">
					<small className="fw-semibold">{chain.video_name}</small>
					<small className="text-muted">
						score={chain.score.toFixed(3)} · span={chain.span_sec.toFixed(1)}s
					</small>
				</div>
				<div className="d-flex gap-2">
					{[hit1, hit2].map((hit, i) => (
						<div key={i} className="text-center" style={{ flex: 1 }}>
							<img
								src={hit.keyframe_url || "https://placehold.co/200x120"}
								className="img-fluid rounded"
								alt={`event ${i + 1}`}
							/>
							<small className="d-block text-muted">
								#{i + 1} · {hit.keyframe_time.toFixed(1)}s
							</small>
						</div>
					))}
				</div>
				<button
					className="btn btn-sm btn-outline-primary w-100 mt-2"
					onClick={() => onUseChain(chain)}
				>
					Dùng chuỗi này
				</button>
			</div>
		</div>
	);
}
```

**Step 2: Verify it compiles**

Run: `cd services/fe && npm run build`
Expected: succeeds

**Step 3: Commit**

```bash
git add services/fe/src/components/TemporalChainCard.tsx
git commit -m "feat(fe): add TemporalChainCard component"
```

### Task 13: Wire everything into `App.tsx`

**Files:**
- Modify: `services/fe/src/App.tsx`

This is the integration task — adding a `searchMode` toggle that swaps the query
input and the results section, without touching the existing KIS behavior.

**Step 1: Add imports**

At the top of `App.tsx`, alongside the existing imports:

```tsx
import { TemporalQueryBuilder } from "./components/TemporalQueryBuilder";
import { TemporalChainCard } from "./components/TemporalChainCard";
import { useTemporalSearch } from "./hooks/useTemporalSearch";
import type { TemporalChain } from "./api/types";
```

**Step 2: Add state**

Inside `SearchPage`, right after the existing `const [exactMode, setExactMode] =
useState(false);` line, add:

```tsx
	const [searchMode, setSearchMode] = useState<"kis" | "temporal">("kis");
	const [event1, setEvent1] = useState("");
	const [event2, setEvent2] = useState("");
	const [submittedEvent1, setSubmittedEvent1] = useState("");
	const [submittedEvent2, setSubmittedEvent2] = useState("");
```

**Step 3: Wire the temporal hook**

Right after the existing `const { hits, totalMs, strategy, loading, error } =
useSearch(...)` block, add:

```tsx
	const {
		chains,
		totalMs: temporalTotalMs,
		warnings: temporalWarnings,
		loading: temporalLoading,
		error: temporalError,
	} = useTemporalSearch(submittedEvent1, submittedEvent2, useLlm, exactMode);
```

**Step 4: Update the persisted-state effects**

In the `useEffect` that restores from `localStorage` (the one reading
`searchState`), add restoring `searchMode` alongside the existing fields:

```tsx
      setSearchMode(saved.searchMode ?? "kis");
```

And in the effect that saves to `localStorage`, add `searchMode` to the saved
object and to its dependency array:

```tsx
        searchMode,
```

(both go next to the existing `exactMode`/`exactMode,` entries respectively)

**Step 5: Update the search-submit handler**

Replace the existing `function search(event: FormEvent)`:

```tsx
  function search(event: FormEvent) {
    event.preventDefault();
    if (searchMode === "temporal") {
      if (event1.trim() && event2.trim()) {
        setSelected([]);
        setSubmittedEvent1(event1.trim());
        setSubmittedEvent2(event2.trim());
      }
      return;
    }
    if (query.trim()) {
      setSelected([]);
      setSubmitted(query.trim());
    }
  }
```

**Step 6: Add a mode toggle and swap the query box**

In the JSX, find the `<form onSubmit={search}>` block containing the query
`<input>`. Wrap the mode choice around it: add a toggle immediately above the
`<form>` tag:

```tsx
                    <div className="btn-group btn-group-sm mb-2" role="group">
                      <input
                        type="radio"
                        className="btn-check"
                        name="searchTopMode"
                        id="modeKis"
                        checked={searchMode === "kis"}
                        onChange={() => setSearchMode("kis")}
                      />
                      <label className="btn btn-outline-secondary" htmlFor="modeKis">
                        KIS (1 câu)
                      </label>
                      <input
                        type="radio"
                        className="btn-check"
                        name="searchTopMode"
                        id="modeTemporal"
                        checked={searchMode === "temporal"}
                        onChange={() => setSearchMode("temporal")}
                      />
                      <label className="btn btn-outline-secondary" htmlFor="modeTemporal">
                        Temporal (2 sự kiện)
                      </label>
                    </div>
```

Then, inside the `<form onSubmit={search}>`, conditionally render either the
existing input-group (KIS) or `TemporalQueryBuilder` (temporal) — replace:

```tsx
                      <div className="input-group mb-3">
                        <input
                          type="text"
                          className="form-control"
                          placeholder="Search by text..."
                          value={query}
                          onChange={(e) => setQuery(e.target.value)}
                        />
                        ... (rest of existing input-group, unchanged)
                      </div>
```

with:

```tsx
                      {searchMode === "temporal" ? (
                        <TemporalQueryBuilder
                          event1={event1}
                          event2={event2}
                          onEvent1Change={setEvent1}
                          onEvent2Change={setEvent2}
                        />
                      ) : (
                        <div className="input-group mb-3">
                          <input
                            type="text"
                            className="form-control"
                            placeholder="Search by text..."
                            value={query}
                            onChange={(e) => setQuery(e.target.value)}
                          />
                          ... (rest of existing input-group, unchanged)
                        </div>
                      )}
                      <button className="btn btn-outline-secondary" type="submit">
                        <i className="bi bi-search"></i>
                      </button>
```

(Keep the submit button and Clear button outside the conditional — they apply to
both modes. The `useLlm` switch and rerank/exact toggle below stay as-is, unchanged,
per the design: they're shared between both modes.)

**Step 7: Add a handler to load a chain into `selected`**

Right after the existing `const toggle = (result: Result) => { ... };` function,
add:

```tsx
  const useChain = (chain: TemporalChain) => {
    const results: Result[] = chain.hits.map((hit) => ({
      ...hit,
      video: hit.video_name,
      url:
        hit.keyframe_url ||
        `https://placehold.co/640x360/e9eef0/354d58?text=${hit.video_name}%23${hit.frame}`,
    }));
    setSelected(results);
  };
```

**Step 8: Render chain cards instead of the flat grid in temporal mode**

Find the "Images results" card block (the one rendering `results.map(...)` in a
grid). Wrap its body in a conditional — replace the `<div className="row
row-cols-2...">{results.map(...)}</div>` block's surrounding card body content
so that when `searchMode === "temporal"`, it renders chain cards instead:

```tsx
                  {searchMode === "temporal" ? (
                    <>
                      {temporalLoading && (
                        <div className="text-center my-3">
                          <div className="spinner-border text-primary" role="status">
                            <span className="visually-hidden">Loading...</span>
                          </div>
                        </div>
                      )}
                      {temporalError && <p className="text-danger">{temporalError}</p>}
                      {temporalWarnings.length > 0 && (
                        <p className="text-muted small">{temporalWarnings.join(", ")}</p>
                      )}
                      {chains.map((chain, i) => (
                        <TemporalChainCard key={i} chain={chain} onUseChain={useChain} />
                      ))}
                      {temporalTotalMs !== null && (
                        <small className="text-muted d-block mt-3">
                          {chains.length} chains · {temporalTotalMs.toFixed(2)} ms
                        </small>
                      )}
                    </>
                  ) : (
                    <>
                      {/* existing loading/error/grid/totalMs block, unchanged */}
                    </>
                  )}
```

**Step 9: Verify it compiles**

Run: `cd services/fe && npm run build`
Expected: succeeds — fix any type errors (most likely: `Result` type needs all
`SearchHit` fields, which `chain.hits[i]` already has since `TemporalChain.hits`
is `SearchHit[]`, same as normal `hits`).

**Step 10: Manual smoke test**

Run: `cd services/fe && npm run dev`, open the app, switch to "Temporal (2 sự
kiện)" mode, type two event descriptions, submit, and confirm chain cards render
(or an appropriate warning/empty state if the dev backend has no matching data —
check `SC_STUB_MODE`/snapshot availability per `CLAUDE.md` if results look wrong).

**Step 11: Commit**

```bash
git add services/fe/src/App.tsx
git commit -m "feat(fe): wire temporal search mode toggle into the search page"
```

---

## Phase 7 — Final verification

### Task 14: Full test suite + lint pass

**Files:** none (verification only)

**Step 1: Run core tests**

Run: `cd services/core && pytest -q`
Expected: all pass

**Step 2: Run be tests**

Run: `cd services/be && pytest -q`
Expected: all pass

**Step 3: Run fmt/lint**

Run: `ruff format services/be/src services/core/src services/ingest/src && ruff check services/be/src services/core/src services/ingest/src`
Expected: no changes needed / no errors (fix anything ruff flags in the new files)

**Step 4: FE build**

Run: `cd services/fe && npm run build`
Expected: succeeds

**Step 5: Commit any formatting fixes**

```bash
git add -A
git commit -m "chore: ruff format temporal search files"
```

(skip this commit if `ruff format` made no changes)

---

## Summary of new/changed files

- `services/core/src/searchcore/config.py` — TRAKE_* config
- `services/core/src/searchcore/temporal.py` — matching algorithm (was a stub)
- `services/core/src/searchcore/server.py` — `SearchTemporal` RPC handler
- `services/core/tests/test_temporal.py` — algorithm tests (was a stub)
- `services/core/tests/test_server.py` — RPC-level tests
- `.env.example` — TRAKE_* vars
- `services/be/src/app/clients/searchcore.py` — `search_temporal()` client fn
- `services/be/src/app/api/search_temporal.py` — new endpoint (new file)
- `services/be/src/app/api/router.py` — register new router
- `services/be/tests/test_search_temporal.py` — endpoint tests (new file)
- `services/fe/src/api/types.ts` — Temporal* types
- `services/fe/src/api/client.ts` — `searchTemporal()` fn
- `services/fe/src/hooks/useTemporalSearch.ts` — new hook (new file)
- `services/fe/src/components/TemporalQueryBuilder.tsx` — 2-event input (was a stub)
- `services/fe/src/components/TemporalChainCard.tsx` — chain result card (new file)
- `services/fe/src/App.tsx` — mode toggle + wiring
