# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Multimodal (text→video) search platform. Query budget for the search path: **100–200ms**.
Reference scale: ~2M vectors × 3072-dim (actual measured dim on current corpus is **1152**,
not 3072 — see `docs/search-design.md` §1/§2, an earlier handoff doc had this wrong).

Repo and docs are written in **Vietnamese**; code identifiers and comments follow the same
convention in `services/core` and `services/be`. Match that when editing those services.

### Three non-negotiable principles (from README.md)

1. **Offline and online are strictly separated.** Heavy models run on rented-by-the-hour GPU,
   then shut down. The hot (query) path is CPU-only.
2. **The hot path lives on one machine.** BE, search core, and encoder share a host and talk
   over a Unix socket (~0.1ms). Every network hop adds 50–200ms.
3. **Embeddings exist in three forms**: fp32 on R2 (rebuild source), SQ8 in RAM (coarse
   search), fp16 on NVMe (mmap, exact rerank).

## Repository layout

```
proto/          gRPC contract between be and core — freeze/review before changing either side
services/core/  Search core: gRPC + FAISS + ONNX text encoder. Cold start 2-3 min, deployed rarely
services/be/    BE gateway: FastAPI, LLM tag enrichment, fusion. Deployed continuously
services/ingest/Offline pipeline (ASR, shot/scene detect, embedding, tag build). Not in prod compose
services/fe/    React frontend
sql/            Postgres schema for the offline ingest pipeline only — see note below
scripts/        Ops scripts (proto codegen, snapshot pull/swap, bench, hardware check)
ops/            Deploy/ops docs *and* code: RunPod Dockerfile/start.sh, encbench.py, gpu_parity.py,
                RUNBOOK.md, SERVERLESS.md, its own Caddyfile, Grafana dashboards, backup/monitoring
run_prompt.py   Manual harness: runs the temporal segmentation prompt over a hardcoded
                Vietnamese query list via litellm (see Temporal search below) — the
                tag-enrichment prompt is built inline in Python, not from a prompt file
docs/decisions/ ADRs — read before proposing an architecture change (currently stubs, see below)
docs/search-design.md   The actual, reviewed design for core+BE search (read this first)
docs/runbook.md         What's done vs. what's missing to run the system for real
```

`src/rackfocus/` at the repo root is a leftover `uv init` scaffold (a lone `print("Hello...")`),
not real code. Deploy has two workflows beyond `ci.yml`: `.github/workflows/deploy.yml` (VPS —
builds and pushes three images to GHCR) and `runpod.yml` (one merged image for RunPod).

**Why core is a separate container but the same repo**: `.proto` is the shared contract — one
PR changes both sides instead of cross-repo PRs. Core loads a 6.7GB index + warmup (2-3 min);
BE deploys several times a day. Same container would mean reloading the index on every BE fix.
Same host: Unix socket is 0.1ms; splitting providers loses that entirely.

**Postgres/Redis are not in the hot path.** `docker-compose.yml` doesn't run either: BE's
`db/{models,queries,session}.py` and `services/{prefilter,hydrate}.py` are unimplemented
stubs (no asyncpg/sqlalchemy call is ever made), and query caching is in-process RAM
(`services/cache.py`), not Redis. Display payload (video_name, frame, keyframe_time, ...)
comes straight back from core as `hit.payload` — read from the snapshot's
`payload.parquet` — so there's no separate hydrate step to reason about. Postgres is only
still needed offline, by `ingest/db.py`, to assign `video_id`/`scene_idx` during the embed
pipeline. `sql/` reflects that offline schema, not a BE dependency.

## Commands

`make` targets wrap `docker compose` for the containerized workflow. A native (no-Docker)
workflow also works and is what production/dev machines actually run day-to-day — see
README.md's "Cách A — chạy trực tiếp bằng venv" for the full walkthrough (three terminals:
core, be, fe; `uv sync --extra core --extra be` or `pip install -r
services/{be,core}/requirements.txt`; core does **not** auto-load `.env`, so export it by
hand; `SEARCHCORE_TARGET` must be `localhost:50051` natively vs. the Unix socket in
compose). That section also has a common-errors table and covers the S3 snapshot/encoder
cache: it's marked done via an empty `.s3_download_done` file per cache dir and **never
diffed against S3 again** — rebuilding a snapshot in place (same `snapshots/vN` path)
leaves every machine with a marker silently serving the stale build; bump the version
directory instead of overwriting it.

```bash
cp .env.example .env      # fill in AWS/S3 keys, LLM key (PG_PASS only matters for offline ingest)
make proto                # regenerate gRPC stubs — ALWAYS run first, and after any .proto edit
make dev                  # docker-compose.yml + docker-compose.dev.yml, hot reload, debug ports
make up / make down       # full stack, detached
make logs S=be            # tail logs for one service
make test                 # pytest inside be + searchcore containers
make fmt / make lint      # ruff format / check on services/{be,core,ingest}/src
make bench                # latency bench, expects p50<60ms p95<150ms without LLM
make snapshot-pull VER=v3 / make snapshot-swap VER=v3   # pull/hot-swap a new index snapshot
make check-hw             # verify NVMe + AVX before deploying core
```

There's no `make migrate`/`make warm` — see the Postgres/Redis note above; there's nothing
to migrate and no hydration cache to warm.

### Running tests directly (matches CI)

Tests run per-service with plain `pytest`, not via `make test` — CI's real command is:

```bash
pip install grpcio-tools && bash scripts/gen_proto.sh   # required first for core/be (gitignored stub artifacts)
pytest services/core/tests -q
pytest services/be/tests -q
pytest services/ingest/tests -q       # does NOT need proto stubs generated first

# single test
pytest services/core/tests/test_temporal.py::test_name -q
```

Each service's `tests/conftest.py` inserts `../src` (and, for BE, `../../core/src`) onto
`sys.path` directly — there's no installed package, no `pytest.ini`, no shared root conftest.
BE's test fixtures spin up a **real** core gRPC server over a Unix socket with a small
synthetic snapshot (not a mocked stub) — only the LLM (litellm) and the ONNX text encoder are
mocked, because those are the expensive/networked pieces. This is intentional: most BE bugs are
in the seam between the two services (wrong proto field, wrong gRPC→HTTP status mapping), which
a mocked core would hide. When adding tests that touch search, prefer extending the existing
`_write_snapshot`/`write_snapshot` fixture helpers in `conftest.py` over hand-rolling FAISS
index files.

`ci.yml` installs each service's deps by hand (not `pip install -e .`) into a matrix job per
service, `fail-fast: false` so all three report instead of stopping at the first failure. If
you add an import used at module load time (e.g. BE's `clients/s3.py` importing `boto3`), add
the package to that service's `deps:` line in the matrix or its tests fail to import at all,
not just the tests touching that code path.

FE: `cd services/fe && npm run dev` / `npm run build` (runs `tsc --noEmit` first) / `npm run preview`.

## Architecture

### Request flow (text search, v1 scope = KIS only — see below)

```
BE  POST /api/search {text, top_k, use_llm}
     ├─ (parallel) ─┬─ gRPC Encode(text=ORIGINAL user text)   → core, ONNX SigLIP text tower
     │              └─ LLM(text + tag_vocab) → {tags[]}        → litellm, 80-300ms
     ├─ wait for both: total = max(LLM, encode), not sum
     └─ gRPC Search(vector=<encoded>, filter.tags=tags, top_k) → core
core ├─ candidate = union(csr_bucket[t] for t in tags)  |  everything if tags empty
     ├─ filter tombstones BEFORE ranking (not after top_k)
     ├─ choose_strategy(len(candidate)): EXACT_SUBSET (brute force on subset) vs 2-tier HNSW
     ├─ diversity: cosine dedup + max_per_shot + min_time_gap
     └─ ResponseMeta{tags_used, candidate_count, filter_strategy_used, snapshot_ver}
```

Encoding the user's **original** query text in parallel with LLM tag selection (rather than
having the LLM rewrite the query) is deliberate: it saves 80–300ms and preserves the user's
own phrasing, which matters for domain-specific terms.

### Tag filtering is a hard partition, not a soft rerank

Each frame carries exactly **one** tag (`domain_id`, 13 fixed values — not the ~50-value
`topic_id`). Candidates for a tag-filtered query are a union of CSR buckets, not a corpus scan.
This means: **a wrong LLM-chosen tag makes the correct frame permanently unreachable** for that
query, at any `ef_search`/`top_k`. System recall = P(LLM picked the right tag) × recall-within-
subset. This is why `ResponseMeta` must always surface `tags_used`/`candidate_count`/
`filter_strategy_used`, and why core auto-falls-back to an untagged search when the tagged
search returns fewer than `top_k` hits above `min_score` (flagged via `warnings: ["tag_fallback"]`).
`Filter.tags` is AND'd with every other `Filter` field (`allow`/`deny`, `video_ids`,
time bounds, etc.) — don't special-case it.

`tag_fallback` firing on a healthy tag is usually `FAISS_EF_SEARCH` set too low, not a bug:
HNSW only visits ~`ef_search` nodes before applying the `IDSelector`, so a tag covering
6–10% of the corpus can yield far fewer than `top_k` hits at low `ef_search` even though
that tag has plenty of matches. Measured on the real corpus (613k points / 13 tags,
`top_k=300`): `ef_search=2000` → 40–71ms core p50 but only 69–79% recall and frequent
fallback; `ef_search=10000` → 195–219ms p50, ~0 fallback. Untagged search recall is
~flat (99.6–100%) across that whole range — only the tagged path is `ef_search`-sensitive.
Tags under `EXACT_SUBSET_MAX` route through brute-force `exact_subset` instead and aren't
affected, which is why this bug looks tag-dependent ("works for some topics, not others").

### Snapshot: the offline↔online contract (second contract after `.proto`)

Produced by `aic-embed-siglip-2026.ipynb`; `services/core` only ever reads it. Lives at
`snapshots/v{N}/`:

| file | purpose |
|---|---|
| `manifest.json` | dim, encoder_name, count, groups, per-file sha256 checksums |
| `visual.faiss` | `IndexHNSWSQ` (HNSW32,SQ8), `METRIC_INNER_PRODUCT` — coarse, in RAM |
| `visual.f16` | raw fp16 N×dim, **no header** — exact rerank, mmap'd from NVMe |
| `idmap.npy` | int64[N] row → point_id |
| `payload.parquet` | row-aligned metadata (video_name, frame, keyframe_time, ...), no vectors |
| `tombstone.bin` | bitset, LSB-first, ceil(N/8) bytes |
| `tags.npy` / `tag_vocab.json` | uint16[N] row → tag_id (65535 = unassigned), and the vocab |

`core/snapshot.py` does **fail-closed** validation on load (rejects, never degrades): dim +
encoder_name match config, sha256 of every file matches the manifest (the fp16 file has no
header, so a size check alone would accept a mismatched build), raw file size, row-alignment
across all 5-7 files, tombstone length, tags within vocab range, and `‖v‖ ≈ 1` across the
**entire** file (not sampled — a bad L2-norm subset would otherwise dominate every query
regardless of relevance). `visual.f16` must fit in page cache alongside everything else within
the container memory limit, or gather latency blows up from major page faults — see the sizing
math in `docs/search-design.md` §2 before changing `count`/`dim`/container memory limits.

fp16 rerank scores are "accurate enough", not exact — flips only occur between near-tied
candidates (error is 50-170× smaller than the rank10/rank11 score gap). Don't treat
`debug.force_exact` as fp32 ground truth; the refine store *is* fp16, there's no fp32 in the
snapshot.

### v1 scope

In scope: text KIS (keyword instance search) and TRAKE (temporal/ordered event chains via
`SearchTemporal` — implemented and tested on both sides: vectorized pair scoring in
`services/core/src/searchcore/temporal.py`, `POST /api/search/temporal` in
`services/be/src/app/api/search_temporal.py`, covered by `test_temporal.py` /
`test_search_temporal.py`). Temporal defaults to **`use_llm=False`**, the opposite of normal
search: each event is tag-enriched independently and the results are unioned into one hard
`Filter`, so a wrong tag on any single event kills the whole chain. **Out of scope**: VKIS (needs a vision tower in
core — only the text tower is bundled), QA (`has_ocr`/`objects` stages not implemented, so
`require_ocr` returns 0 results rather than "no such scene"), tier SCENE (only KEYFRAME is
embedded), sparse/BM25. Check `docs/search-design.md` §9 before assuming a proto field (e.g.
`Filter.require_ocr`, `SearchSimilar`) actually does anything server-side — many exist in
`proto/` for the full product vision but aren't wired up yet.

### Temporal search: prepare-then-search two-step flow

TRAKE has a second endpoint, `POST /api/search/temporal/prepare`
(`services/be/src/app/api/search_temporal.py`), that runs *before* `/api/search/temporal`
and is explicitly exempt from the 100–200ms hot-path budget — the user still has to review
segments and pick two events after it returns. It fans out two independent LLM calls in
parallel (`asyncio.gather`, latency = max, not sum):

- `services/segment.py` splits one Vietnamese query into N English CLIP-ready segments
  (prompt: `services/be/src/app/services/segment_prompt.txt`, moved here from the old
  root-level `prompt.yaml`, which no longer exists).
- `services/enrich.py` tag-selects the same query, same as normal search.

Both services share a contract: **never raise**. Any LLM failure/timeout/garbage JSON
degrades to a single fallback segment (the original query verbatim) or empty tags, plus an
`error` field that surfaces to the client as a `warnings` entry (`llm_failed_segment` /
`llm_failed_tags`) — never an HTTP error.

FE's `TemporalPrepare.tsx` lets the user edit segment text and tick/untick tags before
calling `search_temporal` with `tags` set explicitly, which **bypasses** LLM tag selection
on that second call (`used_llm = req.tags is None and req.use_llm` in
`search_temporal.py`). `tags: []` (user unticked everything → search the whole corpus) and
`tags: None` (skipped prepare → decide by `use_llm`) are different states and must not be
merged — see the comment on `TemporalSearchRequest.tags`. When `use_llm=False` (TRAKE's
default), FE swaps in `TemporalQueryBuilder.tsx` (two plain text boxes) instead of
`TemporalPrepare`. Design doc:
`docs/superpowers/specs/2026-08-28-temporal-llm-segmentation-design.md`.

### Ingest pipeline (offline, GPU)

`services/ingest` turns raw video into per-scene embeddings + metadata, run manually on
rented GPU / Kaggle — not part of the production compose stack. Stages (`src/ingest/stages/`):
probe → shot detect (TransNetV2) → keyframe extraction → scene grouping (BaSSL) → ASR
(chunkformer, Vietnamese) → scene cut. A separate, CPU-only, independently-run job
(`python -m ingest.domain`) does LLM-based domain/topic segmentation against transcripts and
materializes results into MongoDB (`domain_jobs`/`domain_analyses`/`scene_domain_map`) — this
is what `ingest/build_tags.py` joins against (via `AIC_KeyframeSceneMap.ipynb`'s validated
frame→scene map, **not** the unvalidated `payload.scene_idx` column) to produce the snapshot's
`tags.npy`. See `services/ingest/README.md` for the full stage-by-stage contract and MongoDB
schema, and `docs/runbook.md` §2 for the current blockers to a working end-to-end index.

### Proto conventions (`proto/`)

- Adding a field/message is safe. Renumbering, renaming, or deleting fields breaks
  compatibility — use `reserved`. CI's `buf breaking` job enforces this on every PR against
  `main`; `buf lint` runs on every push/PR regardless.
- After any `.proto` change, run `make proto` (or `bash scripts/gen_proto.sh`) to regenerate
  stubs into `services/{core,be}/src/.../pb/` — these are gitignored build artifacts, not
  checked in, and both services fail to import without them.
- `RequestContext` wraps every request (retrofitting it later is expensive). `ResponseMeta`
  carries `tags_used`/`snapshot_ver`/`warnings`/`Timings` back on every response — surface these
  in the BE API rather than dropping them.

### Config

`services/core/src/searchcore/config.py` and `services/be/src/app/config.py` both read
straight from environment / `.env` (core: plain dataclass + `os.getenv`; BE: pydantic-settings).
`.env.example` is the authoritative list of every env var and its default — check there before
guessing a variable name. Notable ones: `SC_STUB_MODE=1` makes core return fake results without
loading a snapshot (useful for BE/FE dev without a real corpus); `SNAPSHOT_DIR`/`ENCODER_PATH`
(bind mount) win over `SNAPSHOT_S3`/`ENCODER_S3` if both are set; `VECTOR_DIM=0` means "trust
the manifest, don't check". `CLOUDFRONT_DOMAIN` is currently unset/unused — BE's S3 client
(`services/be/src/app/clients/s3.py`, see `docs/s3-client.md`) goes straight to S3, not CDN.

## Working conventions

- Branch naming: `feat/<service>-<short-description>`, e.g. `feat/core-temporal-search`.
- Changes to `proto/` need review from whoever owns BE *and* whoever owns core (see
  `CODEOWNERS`/`.github/CODEOWNERS`).
- Never commit `.faiss`, `.f16`, `.npy`, `.parquet` — already blocked in `.gitignore`; these are
  large generated/data artifacts, not source.
- Architecture decisions go in `docs/decisions/` as ADRs, not chat/PR descriptions (the ADR
  files currently exist as stubs — `0001` monorepo/container split, `0002` FAISS vs pgvector,
  `0003` two-tier coarse+rerank, `0004` scene-as-index-unit — fill these in as those decisions
  get re-litigated or need explaining to someone new).
- `docs/search-design.md` is the source of truth for search behavior and has already been
  through adversarial multi-agent design review (37 logged decisions, 5 rejected objections) —
  prefer it over `Handoff_core_be.md`, which it explicitly corrects in several places (latency
  budget, vector dim, int8 quantization rejection, tombstone filter ordering).
