# Temporal search (TRAKE) — design

Date: 2026-08-24
Status: approved, pending implementation plan

## Summary

Add a real temporal ("TRAKE") search mode: the user types two ordered event
descriptions ("a car crashes", "police arrive"), and the system returns
candidate video chains where both events occur in the right order within a
plausible time gap. This is currently unimplemented — the proto RPC
(`SearchTemporal`), and stub files in core/be/fe (`temporal.py`,
`TemporalRequest`, `TemporalQueryBuilder.tsx`) exist but do nothing.

Not in scope: N>2 event chains (proto models this generically for the future,
but the matching algorithm here is 2-events-only, matching what was asked),
reordering/`allow_reorder`, cross-video chains.

## Background / what already exists

- `proto/searchcore/v1/search.proto` already defines `SearchTemporal`,
  `TemporalEvent{query, min_gap_sec, max_gap_sec, optional}`,
  `SearchTemporalRequest`, `TemporalChain{hits, score, video_id, span_sec,
  missing_event_indices}`. No proto changes needed for this feature — the
  handler and BE/FE just aren't wired up yet.
- `Payload.keyframe_time` (proto field 15) is already populated in ingest
  (`services/ingest/src/ingest/stages/media.py`: `idx / fps` per frame),
  already flows through `payload.parquet` → core → BE's `Hit.keyframe_time` →
  FE's `SearchHit.keyframe_time`. No new data plumbing needed for timing —
  S3 only holds the keyframe/clip image files, not timing metadata.
- `temporal_search.py` (repo root, untracked prototype/notebook script) proved
  the algorithm shape: encode 2 text queries, top-K frames per query,
  intersect on shared videos, score every frame pair by
  `0.8*(sim1+sim2) + 0.2*decay(Δt)` with order enforcement, merge overlapping
  pairs into clusters. This design reuses the scoring idea but replaces
  full-corpus brute force with the existing bounded search pipeline, and
  drops the clustering step (see Decisions).
- `docs/runbook.md` flags a real risk with TRAKE under the corpus's tag
  partitioning: "một tag sai giết cả chuỗi" (one wrong tag kills the whole
  chain), since tag filtering is a hard partition, not a re-ranking signal.
  This design does not resolve that risk — it exposes an explicit `use_llm`
  toggle defaulting **off** for temporal mode, so the tag-partition risk is
  opt-in rather than always-on.

## Architecture & data flow

```
FE (search mode: KIS | Temporal)
  Temporal mode: 2 event text inputs + use_llm toggle (default off) +
  rerank/exact toggle (both reused from KIS mode)
        │  POST /api/search/temporal
        ▼
BE (api/search_temporal.py, new)
  asyncio.gather: encode(event1), enrich(event1), encode(event2), enrich(event2)
  → 2 vectors + a SINGLE shared tag set (union of both events' enriched tags,
    empty if use_llm=false) — SearchTemporalRequest has one Filter for the
    whole request (proto §"NB" below), not one per event
        │  gRPC SearchTemporal(events=[{vector}, {vector}], filter.tags=[...])
        ▼
core (searchcore/temporal.py, new)
  for each event: search_with_fallback(vector, tags, top_k=TRAKE_CANDIDATES_PER_EVENT)
    (same shared `tags` applied to both calls — reuses existing search.py's
    HNSW/exact/tombstone/tag-fallback machinery unmodified)
  group both events' candidate rows by video_name
  for each video present in both groups:
    find all valid pairs: t2 > t1, min_gap_sec <= (t2-t1) <= max_gap_sec
    score = sim_weight*(s1+s2) + time_weight*decay(t2-t1)
    keep its top TRAKE_MAX_PAIRS_PER_VIDEO pairs by score
  pool ALL videos' surviving pairs, sort by score, cut to the global top_k
        │
        ▼
BE maps TemporalChain (proto) → TemporalChain (pydantic, reuses existing Hit model)
        │
        ▼
FE renders chain-cards (video, score, span, 2 thumbnails in order);
"Use this chain" button loads both hits into the existing `selected` sidebar,
so the existing TRAKE submission-export path works unmodified.
```

BE always does encoding/enrichment (as it does today for normal search) —
core's `SearchTemporal` handler receives two pre-computed vectors via
`QueryPart.vector`, never raw text. This preserves the "core is CPU-only
search, BE owns LLM/encoding orchestration" split. Gap bounds
(`min_gap_sec`/`max_gap_sec`) are **not** part of the request — they're
core-side config only (see below), same tier as `exact_subset_max`.

**NB, correcting an earlier draft of this section:** `proto/searchcore/v1/search.proto`'s
`SearchTemporalRequest` has exactly one `Filter filter` field for the whole
request — `TemporalEvent` does NOT carry its own per-event tags. So when
`use_llm=true`, BE enriches event1 and event2 independently (they describe
different things, so each gets its own LLM call) but then takes the
**union** of both tag sets and sends that single union as the one shared
`filter.tags`, applied identically to both events' candidate searches inside
core. This matches the wire contract exactly — no proto changes needed —
at the cost of both events searching a (possibly) slightly broader tag set
than either alone would pick.

## Matching algorithm

Approach chosen (over full-corpus brute force, and over generic N-event DP):
bounded candidate-join, hardcoded to exactly 2 events. Reuses the existing
`search()`/`search_with_fallback()` pipeline per event instead of scanning
the whole corpus (a full brute-force rerank per event over ~2M vectors was
already measured as too slow for a single query, per `docs/search-design.md`
— doing it twice would blow well past any usable latency). Generic N-event
chaining was considered and rejected as unrequested complexity — the proto
still models `repeated TemporalEvent` for future extension, but the matching
logic itself only needs to handle 2.

Simplification versus the `temporal_search.py` prototype: no
overlap-merging/clustering step (`merge_pairs_one_video`) — that existed to
support exploratory notebook analysis of many overlapping candidate pairs,
which a ranked-results endpoint doesn't need. The prototype's other idea,
keeping multiple pairs per video (its `PAIRS_PER_VIDEO`), IS kept: each
video contributes up to `TRAKE_MAX_PAIRS_PER_VIDEO` of its best-scoring valid
pairs (not just one), all videos' surviving pairs are pooled into one list,
sorted by score, and cut to the final `top_k`. A single video can appear
more than once in the result if several of its pairs score well enough —
each pair still renders as its own chain-card.

**Hard filter (not a soft penalty):** a candidate pair is excluded entirely —
never scored, never returned — if `t2 - t1 < TRAKE_MIN_GAP_SEC` or
`t2 - t1 > TRAKE_MAX_GAP_SEC`.

**Score, for pairs that pass the filter:**
```
decay(dt) = exp(-TRAKE_LAMBDA * (dt - TRAKE_MIN_GAP_SEC))   # dt >= min_gap_sec by construction, so decay(dt) <= 1.0
score     = TRAKE_SIM_WEIGHT * (sim1 + sim2) + TRAKE_TIME_WEIGHT * decay(dt)
```
`TRAKE_MIN_GAP_SEC` doubles as both the hard floor and the decay's
zero-penalty reference point — an earlier draft had a separate
`TRAKE_GRACE_SEC` for the decay's no-penalty window, but with
`grace_sec <= min_gap_sec` that window would only ever be evaluated on
already-rejected pairs (dead code), so it was folded into `min_gap_sec`.

## New core config (`searchcore/config.py` + `.env`)

Following the existing `_int`/`_bool`/`os.getenv` pattern already used for
`RERANK_CANDIDATES`, `EXACT_SUBSET_MAX`, etc.

| Env var | Default | Meaning |
|---|---|---|
| `TRAKE_CANDIDATES_PER_EVENT` | 500 | top-K per event before the video join (bounds join cost) |
| `TRAKE_MAX_PAIRS_PER_VIDEO` | 5 | how many of a video's best valid pairs survive into the pool before the global top_k cut |
| `TRAKE_MIN_GAP_SEC` | 5.0 | hard floor on `t2-t1`; also the decay's zero-penalty point |
| `TRAKE_MAX_GAP_SEC` | 120.0 | hard ceiling on `t2-t1` |
| `TRAKE_LAMBDA` | 0.00557 | decay rate beyond `min_gap_sec` (from the prototype) |
| `TRAKE_SIM_WEIGHT` | 0.8 | weight on similarity in the blended score |
| `TRAKE_TIME_WEIGHT` | 0.2 | weight on temporal fit in the blended score |
| `TRAKE_TOP_K_CHAINS` | 20 | max chains returned |

Order (`t2 > t1`) and same-video are always enforced — not configurable, no
proto fields (`allow_reorder`/`require_same_video`) exposed, since that's the
definition of this feature, not a tunable.

## BE API

`services/be/src/app/api/search_temporal.py` (new router):

```python
class TemporalSearchRequest(BaseModel):
    event1: str = Field(min_length=1)
    event2: str = Field(min_length=1)
    use_llm: bool = False          # default OFF for temporal (tag-partition risk, see Background)
    exact: bool = False            # reuses rerank/exact toggle semantics from normal search
    top_k: int | None = None       # chains to return, capped by TRAKE_TOP_K_CHAINS

class TemporalChain(BaseModel):
    video_name: str
    score: float
    span_sec: float
    hits: list[Hit]                # reuses the EXISTING Hit model from api/search.py, exactly 2, in order

class TemporalSearchResponse(BaseModel):
    chains: list[TemporalChain]
    warnings: list[str]           # merged event1_*/event2_* warnings from both events, plus temporal_* ones
    tags_used: list[int]          # union of both events' resolved tags (only meaningful when use_llm=True)
    snapshot_ver: str
    timings_ms: dict[str, float]

@router.post("/search/temporal", response_model=TemporalSearchResponse)
async def search_temporal(req: TemporalSearchRequest) -> TemporalSearchResponse: ...
```

`services/be/src/app/clients/searchcore.py` gets a new `search_temporal()`
function alongside the existing `search()`, building a
`pb.SearchTemporalRequest` with 2 `TemporalEvent`s (each
`QueryPart(vector=...)` + per-event tags) — no gap fields on the wire
request, since those are core-config-only.

`services/be/src/app/schemas/search.py`'s `TemporalRequest` stub gets
replaced by the above (or the request/response models simply live in
`api/search_temporal.py` directly, matching how `api/search.py` currently
defines its own `SearchRequest`/`Hit`/`SearchResponse` inline rather than in
`schemas/`).

## FE wiring

- `App.tsx`: new `searchMode: "kis" | "temporal"` state. Selecting "temporal"
  swaps the query-box section for 2 event inputs (per approved UI decision:
  same panel, swap the box) and swaps the results section from the flat grid
  to the chain-card list. The existing `useLlm` and `exactMode` toggles stay
  visible and apply to both modes.
- `TemporalQueryBuilder.tsx` (currently a 1-line stub): the 2 event text
  inputs.
- `useTemporalSearch.ts` (new): mirrors `useSearch.ts`'s debounce/
  abort-controller/request-id-race pattern, calls a new `searchTemporal()` in
  `api/client.ts`.
- `TemporalChainCard.tsx` (new): one chain — video name, score, span, two
  thumbnails in order with their `keyframe_time`s, and a "Use this chain"
  button that pushes both hits into the existing `selected` array via the
  current `toggle()` mechanism — so `exportResult()`'s existing "trake"
  branch (which already expects same-video `selected` frames) works
  unmodified.
- `api/types.ts`: new `TemporalSearchRequest`/`TemporalChain`/
  `TemporalSearchResponse` types mirroring the BE pydantic models.

## Error handling

All non-fatal cases return an empty/partial result with a `warnings` entry,
matching the existing pattern (`tag_empty`, `llm_failed`, `tag_fallback` in
`api/search.py`) rather than raising:

- Empty candidate pool for event 1 or 2 → `warnings: ["temporal_no_candidates_event1"]` (or `_event2`), `chains: []`.
- No video appears in both candidate pools → `warnings: ["temporal_no_common_video"]`, `chains: []`.
- Shared videos exist but no pair passes the gap filter → `warnings: ["temporal_no_valid_gap"]`, `chains: []` (distinguishes "right video, wrong timing" from "never found the video," useful for debugging query wording).
- LLM failure for one event (when `use_llm=true`) → because `SearchTemporalRequest` has a SINGLE shared `Filter` for both events (see the "NB" note above), there is no per-event untagged fallback: the surviving event's tags are applied to BOTH events, including the one whose enrichment failed. BE adds an `llm_failed_event1`/`llm_failed_event2` warning so this is visible, but the failed event still searches under the other event's tags rather than untagged.
- Core/gRPC errors (`UNAVAILABLE`, `DEADLINE_EXCEEDED`, `INVALID_ARGUMENT`) → same HTTP status mapping `api/search.py` already uses (503/504/400/502).

## Testing plan

- `services/core/tests/test_temporal.py` (currently a 1-line stub): direct
  calls to `search_temporal()` against a small synthetic snapshot (same
  `conftest.py` fixture pattern as `test_search.py`). Cases: order
  enforcement (t2<t1 pair excluded), gap filtering (dt<5 and dt>120
  excluded, boundary values), top-`TRAKE_MAX_PAIRS_PER_VIDEO`-per-video
  capping when a video has more valid pairs than the cap, global top_k
  cut across pooled pairs from multiple videos, empty-candidate case,
  no-common-video case.
- `services/be/tests/test_search_temporal.py` (new): through the real
  gRPC-backed test client (no mocks, matching the pattern from commit
  `86153e6`). Cases: endpoint end-to-end happy path, `use_llm` toggle
  behavior, error-code-to-HTTP-status mapping.
- FE: no new automated tests — matches the current FE testing level (`tsc
  --noEmit` + `vite build` only, no FE test suite exists in this repo today).

## Open items for the implementation plan

- Exact config-reading mechanism in `searchcore/config.py` for the new
  `TRAKE_*` vars (same `_int`/`_bool`/plain-float pattern as existing fields —
  needs a `_float` helper, which doesn't exist yet, since existing numeric
  config is all int).
- Whether `schemas/search.py`'s stub `TemporalRequest` comment gets deleted
  or repurposed — current repo convention (`api/search.py`) defines request/
  response models inline in the router file, not in `schemas/`, so this
  design follows that precedent rather than the stub file's original intent.
