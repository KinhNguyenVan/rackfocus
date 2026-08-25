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
services/be/    BE gateway: FastAPI, LLM tag enrichment, fusion, hydrate. Deployed continuously
services/ingest/Offline pipeline (ASR, shot/scene detect, embedding, tag build). Not in prod compose
services/fe/    React frontend
sql/            Postgres schema + hand-tuned queries
scripts/        Ops scripts (proto codegen, snapshot pull/swap, bench, hardware check)
docs/decisions/ ADRs — read before proposing an architecture change (currently stubs, see below)
docs/search-design.md   The actual, reviewed design for core+BE search (read this first)
docs/runbook.md         What's done vs. what's missing to run the system for real
```

**Why core is a separate container but the same repo**: `.proto` is the shared contract — one
PR changes both sides instead of cross-repo PRs. Core loads a 6.7GB index + warmup (2-3 min);
BE deploys several times a day. Same container would mean reloading the index on every BE fix.
Same host: Unix socket is 0.1ms; splitting providers loses that entirely.

## Commands

All commands assume Docker Compose; there's no local (non-container) dev workflow for
be/core/fe. `make` targets wrap `docker compose`:

```bash
cp .env.example .env      # fill in PG_PASS, S3 keys, LLM key
make proto                # regenerate gRPC stubs — ALWAYS run first, and after any .proto edit
make dev                  # docker-compose.yml + docker-compose.dev.yml, hot reload, debug ports
make up / make down       # full stack, detached
make logs S=be            # tail logs for one service
make migrate              # alembic upgrade head (via be container)
make test                 # pytest inside be + searchcore containers
make fmt / make lint      # ruff format / check on services/{be,core,ingest}/src
make bench                # latency bench, expects p50<60ms p95<150ms without LLM
make warm                 # warm hydration cache
make snapshot-pull VER=v3 / make snapshot-swap VER=v3   # pull/hot-swap a new index snapshot
make check-hw             # verify NVMe + AVX before deploying core
```

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
`SearchTemporal`, ~implemented — see `services/core/src/searchcore/temporal.py` and
`services/be/src/app/api/search_temporal.py`). **Out of scope**: VKIS (needs a vision tower in
core — only the text tower is bundled), QA (`has_ocr`/`objects` stages not implemented, so
`require_ocr` returns 0 results rather than "no such scene"), tier SCENE (only KEYFRAME is
embedded), sparse/BM25. Check `docs/search-design.md` §9 before assuming a proto field (e.g.
`Filter.require_ocr`, `SearchSimilar`) actually does anything server-side — many exist in
`proto/` for the full product vision but aren't wired up yet.

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
the manifest, don't check".

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
