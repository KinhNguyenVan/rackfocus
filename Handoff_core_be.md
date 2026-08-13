# rackfocus — Bàn giao kỹ thuật: Search Core & BE Gateway

> Tài liệu này viết để đưa vào Claude Code làm context. Nó chứa đủ ràng buộc,
> con số và bẫy đã biết để implement mà không cần đọc lại lịch sử thảo luận.
>
> **Đọc mục 2 (Bất biến) và mục 8 (Bẫy đã biết) trước khi viết dòng code nào.**

---

## 1. Bối cảnh và trạng thái hiện tại

**Sản phẩm**: hệ thống truy vấn video bằng ngôn ngữ tự nhiên cho AI Challenge HCMC 2026 (thể thức VBS/LSC), thiết kế để đóng gói thành sản phẩm sau cuộc thi.

**Quy mô tham chiếu**:
- ~500 giờ video → ~300k scene, ~2M keyframe
- Vector visual 3072-dim, vector text (OCR/ASR) 1024-dim
- Ngân sách latency phần search: **100–200ms** (đã đo thiết kế: 39–110ms)

**Đã xong**:
- Scaffold monorepo (151 file), `docker compose` chạy được
- `proto/searchcore/v1/{common,search,points,admin}.proto` — 745 dòng, `buf lint` sạch
- `scripts/gen_proto.sh` sinh stub cho cả core và be
- Walking skeleton: BE → gRPC → core (stub mode) trả kết quả giả, đã verify

**Chưa xong**: toàn bộ phần implement thật của core và BE.

**Cấu trúc thư mục liên quan**:
```
proto/searchcore/v1/*.proto          hợp đồng, đã chốt
services/core/src/searchcore/        <- implement ở đây
services/be/src/app/                 <- và ở đây
sql/init/*.sql                       schema Postgres
```

---

## 2. Bất biến — không được vi phạm

| # | Bất biến | Lý do |
|---|---|---|
| I1 | **Model nặng không bao giờ chạy trong hot path.** Encoder query là ngoại lệ duy nhất, và phải là ONNX/CPU đã load sẵn trong RAM. | Ingest chạy trên GPU thuê theo giờ. Hot path chỉ CPU. |
| I2 | **Core và BE cùng máy.** Giao tiếp qua Unix socket, fallback TCP. Không bao giờ tách sang hai nhà cung cấp. | Unix socket đo được 0.017ms. Vượt biên network là +50–200ms. |
| I3 | **Postgres không nằm trong hot path.** Hydrate metadata lấy từ Redis. Postgres chỉ dùng cho pre-filter và ghi log. | 15ms → 0.3ms. |
| I4 | **Vector không bao giờ đi qua Postgres.** Không dùng pgvector (giới hạn 2000 chiều cho `vector`, 3072 vượt ngưỡng). | |
| I5 | **Embedding tồn tại 3 bản**: fp32 trên R2 (nguồn sự thật, để rebuild), SQ8 trong RAM (coarse), fp16 trên NVMe mmap (rerank). SQ8 không decompress ngược được. | |
| I6 | **Snapshot bất biến, có version.** Đổi index = load snapshot mới + atomic swap. Không mutate index đang serve. | |
| I7 | **Warmup là bắt buộc** sau mỗi lần load snapshot. Không phải tuỳ chọn. | Query đầu chậm 10–50× nếu bỏ qua. Trong thi đấu query đầu là query tính điểm. |
| I8 | **Refine store phải nằm trên NVMe.** HDD hoặc network storage làm sụp thiết kế 2 tầng. | Mỗi query đọc random ~10MB. HDD ~100 IOPS → hàng trăm ms. |
| I9 | **Logging là fire-and-forget.** Không nằm trong đường trả response. | |
| I10 | **Đổi `.proto` phải bump v2 nếu phá tương thích.** CI có `buf breaking` canh gác. | |

---

## 3. Search Core — đặc tả implement

### 3.1 Trách nhiệm

Core làm **đúng ba việc**: encode query → vector search → trả hit có payload.
Core **không** biết gì về LLM, không gọi Postgres, không quyết định nghiệp vụ.

### 3.2 Kiến trúc 2 tầng — thuật toán lõi

Đây là phần quan trọng nhất. Mọi thứ khác xoay quanh nó.

```
Query vector (3072-dim, đã L2 normalize)
        │
        ▼
┌─ TẦNG 1: COARSE ─────────────────────────────────┐
│  FAISS IndexHNSWFlat + ScalarQuantizer(SQ8)      │
│  Toàn bộ trong RAM (~6.7 GB cho 2M × 3072)       │
│  index.hnsw.efSearch = ef_search (32-256)        │
│  -> trả về rerank_candidates (500-1000) row id   │
│  ~1-5ms                                           │
└──────────────────────────────────────────────────┘
        │ rows: np.ndarray[int64]
        ▼
┌─ TẦNG 2: EXACT RERANK ───────────────────────────┐
│  refine.f16 = np.memmap(fp16, shape=(N, dim))    │
│  cand = refine[rows].astype(np.float32)          │
│  exact = cand @ qvec        <- BLAS, 1 matmul    │
│  order = np.argsort(-exact)[:top_k]              │
│  ~1-5ms  (gather ~10MB random từ NVMe)           │
└──────────────────────────────────────────────────┘
        │
        ▼
   idmap[rows] -> scene_id, join payload
```

**Vì sao gather thủ công chứ không dùng `faiss.IndexRefineFlat`**: refine store là mmap fp16, không phải index FAISS trong RAM. Tự gather cho phép kiểm soát chính xác lượng đọc, cache được, và không tốn thêm 22.9 GB RAM cho `IndexFlat` fp32.

**Vì sao fp16 đủ để gọi là "exact"**: sai số fp16 ~0.0001, nhỏ hơn nhiều so với khoảng cách phân biệt ngữ nghĩa giữa các candidate. Không đổi thứ hạng top-10 trong 800 candidate. Khác hẳn SQ8 (256 mức/chiều) — sai số đủ lớn để đổi thứ hạng, nên SQ8 chỉ dùng ở tầng thô.

**"Exact" ở đây nghĩa là**: không bỏ sót candidate nào trong tập 800 đã lọc. Khác với "approximate" của HNSW (đi tắt trên đồ thị, có thể bỏ sót). Hai trục độc lập: *thuật toán* (exact vs approximate) và *độ chính xác số học* (fp32/fp16/SQ8).

### 3.3 Module và trách nhiệm

```
services/core/src/searchcore/
├── main.py           Entrypoint: config -> load snapshot -> warmup -> gRPC serve
├── config.py         Env vars, dataclass frozen
├── server.py         3 servicer: SearchCoreService, PointService, AdminService
├── snapshot.py       Snapshot bất biến: đọc manifest, validate, load faiss + mmap + idmap
├── holder.py         IndexHolder: atomic pointer swap, giữ ref cũ cho request đang chạy
├── search.py         2-tier search, filter, fusion đa vector
├── sparse.py         Inverted index BM25 (FAISS không làm sparse)
├── fusion.py         RRF, DBSF, weighted sum
├── diversity.py      MMR, max_per_video, dedup theo cosine
├── temporal.py       TRAKE: ghép chuỗi sự kiện theo ràng buộc thời gian
├── region.py         Sub-region search: lưới tĩnh 5 ô + fuse IoU-distance
├── points.py         CRUD: upsert, delete (tombstone), scroll, count
├── collections.py    Quản lý nhiều collection, lazy load
├── warmup.py         N query giả sau load
├── metrics.py        Latency percentile theo stage
└── encoder/
    ├── base.py       Interface
    ├── text.py       ONNX text tower
    └── image.py      ONNX image tower (cho VKIS)
```

### 3.4 Snapshot — định dạng và load

```
snapshots/v3/
├── manifest.json      {version, dim, metric, encoder_name, count, built_at, checksums, vectors:[...]}
├── visual.faiss       HNSW32,SQ8 — keyframe tier
├── visual.f16         raw fp16 row-major — refine store
├── visual_scene.faiss HNSW32,SQ8 — scene tier
├── visual_scene.f16
├── ocr.faiss          1024-dim
├── ocr.f16
├── asr.faiss
├── asr.f16
├── sparse.bin         inverted index đã serialize
├── payload.bin        payload store dạng cột (struct-of-arrays)
├── idmap.npy          int64[N] -> scene_id
└── tombstone.bin      bitset
```

`manifest.json` là **hợp đồng giữa offline và online**. Core phải **từ chối load** nếu `dim`, `metric`, hoặc `encoder_name` không khớp config — chống lỗi âm thầm khi trộn snapshot của model khác.

```python
class Snapshot:
    """Bất biến. Swap bằng đổi pointer, không mutate."""
    def __init__(self, path: str, cfg: Config):
        man = json.load(open(f"{path}/manifest.json"))
        if man["dim"] != cfg.dim or man["encoder_name"] != cfg.encoder_name:
            raise ValueError(f"snapshot không khớp config: {man}")
        self.version = man["version"]
        self.indexes = {}   # {(vector_name, tier): faiss.Index}
        self.refine  = {}   # {(vector_name, tier): np.memmap}
        self.idmap   = np.load(f"{path}/idmap.npy")
        self.payload = PayloadStore.load(f"{path}/payload.bin")
        self.tombstone = np.fromfile(f"{path}/tombstone.bin", dtype=np.uint8)
        # ...

class IndexHolder:
    def __init__(self): self._snap = None; self._lock = threading.Lock()
    def swap(self, new: Snapshot):
        with self._lock:
            old, self._snap = self._snap, new    # atomic
        del old   # request đang chạy vẫn giữ ref cũ, GC dọn sau
    @property
    def snap(self) -> Snapshot: return self._snap
```

**Hot swap cần RAM tạm gấp đôi index nóng** (6.7 × 2 = 13.4 GB). Đây là một lý do chọn máy 64 GB.

### 3.5 Filter — chọn chiến lược theo cardinality

HNSW + filter chọn lọc cao là bài toán khó: khi hầu hết neighbor bị loại, thuật toán đi trên đồ thị bị lạc, phải duyệt rất xa mới gom đủ `top_k` → latency tăng vọt, recall tụt.

```python
def choose_strategy(cardinality: int, universe: int, requested) -> FilterStrategy:
    if requested != FILTER_STRATEGY_UNSPECIFIED:
        return requested
    ratio = cardinality / universe
    if cardinality <= 10_000:      return EXACT_SUBSET  # quét vét cạn nhanh hơn
    if ratio < 0.01:               return EXACT_SUBSET  # quá chọn lọc, HNSW sẽ lạc
    if ratio > 0.5:                return POST          # filter rộng, lọc sau rẻ hơn
    return PRE                                          # dùng IDSelector của FAISS
```

Trả `Timings.filter_strategy_used` về để tune bằng số liệu thật.

**IdSet có 3 dạng mã hoá.** BE chọn dạng theo kích thước; core phải giải mã cả ba:
- `LIST` khi < ~50k ID
- `BITSET` khi lớn hơn (2M bit = 250KB thay vì 16MB — nhỏ hơn 64×)
- `ROARING` khi thưa

Dùng `cardinality` để chọn chiến lược **trước khi** giải mã, không phải sau.

### 3.6 Multi-vector và hybrid fusion

Mỗi `vector_name` là một FAISS index riêng, chung không gian ID.

```python
def search_multi(snap, parts: list[QueryPart], params) -> list[Hit]:
    # 1. Nhóm part theo vector_name
    by_name = groupby(parts, key=lambda p: p.vector_name or "visual")

    # 2. Search song song từng index (ThreadPoolExecutor — FAISS nhả GIL)
    results = {}
    with ThreadPoolExecutor(max_workers=len(by_name)) as ex:
        futs = {name: ex.submit(search_one, snap, name, ps, params)
                for name, ps in by_name.items()}
        results = {n: f.result() for n, f in futs.items()}

    # 3. Sparse chạy riêng (inverted index, không phải FAISS)
    if any(p.HasField("sparse") for p in parts):
        results["sparse"] = snap.sparse.search(...)

    # 4. Fusion
    return fuse(results, mode=params.fusion, weights=params.vector_weights)
```

Search song song nên latency là **max**, không phải tổng (~3-6ms cho 3 index).

**Fusion mặc định nên là RRF**, không phải weighted sum. Lý do: điểm cosine của dense và điểm BM25 của sparse ở hai thang hoàn toàn khác nhau, cộng thẳng là vô nghĩa. RRF chỉ dùng thứ hạng nên an toàn:

```python
def rrf(ranked_lists: dict[str, list[int]], weights: dict[str, float], k: int = 60):
    scores = defaultdict(float)
    for name, ids in ranked_lists.items():
        w = weights.get(name, 1.0)
        for rank, doc_id in enumerate(ids):
            scores[doc_id] += w / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: -x[1])
```

### 3.7 Sparse index — phần FAISS không làm

Đây là hạng mục tốn công nhất (~2–3 ngày). Yêu cầu tối thiểu:

```python
class SparseIndex:
    """Inverted index BM25 trên token OCR + ASR."""
    # postings: dict[term_id, np.ndarray[(doc_id, weight)]]
    # Lưu dạng CSR để nén: indptr + indices + data
    def search(self, query_terms: dict[int, float], top_k: int,
               allowlist: np.ndarray | None) -> list[tuple[int, float]]:
        # WAND hoặc block-max WAND để cắt sớm; naive scan cũng chấp nhận được
        # ở quy mô 7M posting nếu cần ra kết quả nhanh
```

**Nếu cần cắt phạm vi giai đoạn 1**: bỏ sparse trong core, để BE fuse bằng Postgres GIN như thiết kế ban đầu. Proto đã hỗ trợ cả hai, đổi ý không phải sửa hợp đồng.

### 3.8 CRUD và tombstone

**Ràng buộc FAISS cứng: `IndexHNSW` không hỗ trợ `remove_ids`.**

```python
def delete_points(snap, ids):
    for i in ids:
        row = snap.id_to_row[i]
        set_bit(snap.tombstone, row)      # chỉ đánh dấu
    # KHÔNG xoá khỏi index. Lọc lúc search.

def search(...):
    rows = index.search(...)
    rows = rows[~get_bits(snap.tombstone, rows)]   # lọc tombstone
```

**Chi phí không nằm ở latency tức thời** (kiểm bitset O(1)/candidate, <0.1ms) **mà ở sự trôi dần**:
- Thêm điểm vào HNSW không rebuild → chất lượng đồ thị giảm
- Đồ thị vẫn đi qua node chết → phải tăng `ef_search` để bù
- Sau ~20% điểm bị thêm/xoá: latency có thể +30–50%, recall tụt

→ `AdminService.Optimize` (server streaming, trả tiến độ) rebuild và nén. **Chiếm 2× RAM index lúc chạy.**

### 3.9 Collections — lazy load

Mỗi collection load đồng thời nhân toàn bộ index lên (1 collection = 8.35 GB, 3 = 25 GB).

**Quyết định**: lazy load. Chỉ giữ collection đang dùng trong RAM; collection khác để trên đĩa, load khi có request đầu tiên (chấp nhận cold start 2–3 phút). LRU evict khi vượt ngưỡng RAM cấu hình.

### 3.10 Cấu hình runtime bắt buộc

```python
faiss.omp_set_num_threads(4)   # KHÔNG để mặc định — oversubscribe làm p99 dao động

server = grpc.server(
    futures.ThreadPoolExecutor(max_workers=8),
    options=[("grpc.max_send_message_length", 32*1024*1024),
             ("grpc.max_receive_message_length", 32*1024*1024)],
)
server.add_insecure_port(f"unix://{cfg.socket_path}")
server.add_insecure_port(f"[::]:{cfg.tcp_port}")
server.start()
os.chmod(cfg.socket_path, 0o666)   # BE chạy uid khác, không có dòng này sẽ Permission denied
```

### 3.11 Warmup

```python
def warmup(snap, encoder, n: int = 50):
    """Bắt buộc. Warm HNSW graph vào cache CPU, warm mmap page cache, init BLAS."""
    rng = np.random.default_rng(0)
    for _ in range(n):
        q = rng.standard_normal((1, snap.dim), dtype=np.float32)
        q /= np.linalg.norm(q)
        two_tier_search(snap, q, top_k=50, n_cand=800)
```

Cân nhắc thêm `vmtouch -t refine.f16` sau deploy để prefault page cache — ổn định p99.

---

## 4. BE Gateway — đặc tả implement

### 4.1 Trách nhiệm

BE giữ phần **orchestration**: LLM enrich, dựng allowlist từ Postgres, gọi core, hydrate metadata, ghi log, phục vụ FE.

### 4.2 Hot path — pattern bắt buộc

```python
@app.post("/api/search")
async def search(req: SearchReq):
    t0 = time.perf_counter()

    # 1. Cache
    qhash = blake2b(normalize(req.text).encode(), digest_size=16).hexdigest()
    if cached := await redis.get(f"res:{qhash}:{req.top_k}"):
        return msgpack.unpackb(cached)

    # 2. LLM enrich và pre-filter CHẠY SONG SONG — không nối tiếp
    tasks = []
    if req.use_llm:  tasks.append(llm_enrich(req.text))
    if req.filters:  tasks.append(build_allowlist(req.filters))
    enrich, allowlist = await asyncio.gather(*tasks) if tasks else (None, None)

    # 3. gRPC — core tự encode text, tiết kiệm 1 round-trip
    resp = await sc_stub.Search(pb.SearchRequest(
        ctx=pb.RequestContext(request_id=rid, session_id=req.session_id,
                              task_type=req.task_type, deadline_ms=200),
        query=[pb.QueryPart(text=req.text, vector_name="visual", weight=1.0)],
        top_k=req.top_k,
        params=pb.SearchParams(ef_search=64, rerank_candidates=800,
                               fusion=pb.FUSION_MODE_RRF),
        filter=build_filter(allowlist),
        diversity=pb.Diversity(max_per_video=3, dedup_threshold=0.97),
        with_payload=True,
    ))

    # 4. Hydrate từ Redis hot cache — KHÔNG query Postgres
    hits = await hydrate_from_redis([h.id for h in resp.hits])

    out = {...}
    await redis.setex(f"res:{qhash}:{req.top_k}", 300, msgpack.packb(out))
    asyncio.create_task(log_search(req, out))   # fire-and-forget
    return out
```

### 4.3 Hydration cache — tối ưu có tỉ lệ lợi ích/công sức cao nhất

Nạp sẵn display payload của toàn bộ scene vào Redis:

```
key: h:{scene_id}
val: msgpack {vid, kf, clip, s, e, title}   ~200 bytes
2M × 200B = ~400 MB
```

Loại Postgres hoàn toàn khỏi hot path: 15ms → 0.3ms.

Postgres chỉ còn hai việc: **pre-filter** (dựng allowlist) và **ghi log**.

Nếu buộc phải hydrate qua Postgres, **bắt buộc batch một query**, không N+1:

```sql
SELECT s.id, s.video_id, s.start_sec, s.end_sec, s.keyframe_s3, v.title
FROM scenes s JOIN videos v ON v.id = s.video_id
WHERE s.id = ANY($1::bigint[])
ORDER BY array_position($1::bigint[], s.id);
```

### 4.4 Pre-filter — chọn mã hoá IdSet

```python
async def build_allowlist(filters) -> pb.IdSet:
    ids = await db.fetch_ids(filters)          # SQL WHERE ... -> scene_id
    n = len(ids)
    if n < 50_000:
        return pb.IdSet(encoding=LIST, ids=ids, cardinality=n, universe_size=UNIVERSE)
    bitset = pack_bitset(ids, UNIVERSE)         # 2M bit = 250KB
    return pb.IdSet(encoding=BITSET, packed=bitset, cardinality=n, universe_size=UNIVERSE)
```

### 4.5 API surface

```
POST   /api/search                KIS/AVS
WS     /ws/search                 streaming progressive
POST   /api/search/temporal       TRAKE
POST   /api/search/qa             search + VLM đọc scene trả lời
POST   /api/search/region         text + bounding box
POST   /api/search/similar        VKIS
POST   /api/search/feedback       relevance feedback -> rerank vòng sau
GET    /api/scenes/{id}           chi tiết + OCR/ASR/objects
GET    /api/scenes/{id}/neighbors temporal context
GET    /api/videos/{id}/timeline
POST   /api/videos/upload-url     presigned R2
POST   /api/videos/{id}/ingest    enqueue
GET    /api/videos/{id}/status
POST   /api/submit                nộp bài + log
POST   /api/admin/snapshot/load
GET    /api/admin/stats
GET    /healthz  /readyz
```

### 4.6 Fusion lexical (nếu giữ sparse ở BE thay vì core)

Chỉ tính BM25 **trên top-K đã lọc bằng vector**, không quét toàn kho: O(2M) → O(1000).

---

## 5. Ngân sách latency — mục tiêu nghiệm thu

| Giai đoạn | Mục tiêu |
|---|---|
| Redis cache hit | 0.5ms → trả luôn |
| Encode query (ONNX CPU) | 10–30ms |
| Pre-filter Postgres | 5–15ms ⟍ song song |
| LLM enrich (tuỳ chọn) | 80–300ms ⟋ |
| gRPC round-trip (Unix socket) | 0.1ms (đo được 0.017ms) |
| Coarse search (3 index song song) | 3–6ms |
| Sparse BM25 | 5–15ms |
| Rerank fp16 | 3–10ms |
| Fusion RRF | 2–5ms |
| Diversity / MMR | 2–5ms |
| Hydrate Redis | 1–3ms |
| Serialize | 2–5ms |
| **Tổng không LLM** | **39–110ms** |

**Nghiệm thu**: `make bench` cho **p50 < 60ms, p95 < 150ms** khi không dùng LLM.

Nếu lệch, đọc `Timings` per-stage trong response để biết chính xác chỗ nào chậm — đừng đoán.

---

## 6. Ngân sách RAM

| Thành phần | RAM | mmap NVMe |
|---|---|---|
| visual keyframe (2M × 3072) SQ8 | 6.71 GB | 12.29 GB |
| visual scene (300k × 3072) SQ8 | 1.01 GB | 1.84 GB |
| ocr dense (150k × 1024) | 0.20 GB | 0.31 GB |
| asr dense (200k × 1024) | 0.26 GB | 0.41 GB |
| sparse inverted index | 0.10 GB | — |
| payload store | 0.07 GB | — |
| text encoder ONNX | 1.00 GB | — |
| text encoder #2 (nếu ocr/asr dùng model khác) | 1.20 GB | — |
| image encoder ONNX (VKIS) | 1.50 GB | — |
| process Python + gRPC | 1.50 GB | — |
| **Core tổng** | **~13.5 GB** | **14.85 GB** |
| BE + Postgres + Redis + OS | 7.00 GB | — |
| **Thường trú** | **~20.5 GB** | |
| + chỗ hot swap (2× index nóng) | +7.71 GB | |

**Máy khuyến nghị: 64 GB + NVMe.** 32 GB chạy được nhưng không đủ chỗ hot swap khi index đầy.

**Cân nhắc tiết kiệm 1.2 GB + 10–30ms**: dùng **cùng một text encoder** cho cả visual/ocr/asr, chấp nhận OCR/ASR search kém tối ưu hơn chút.

---

## 7. Thứ tự implement

**Core**
1. `config.py` + `snapshot.py` — load được 1 index visual, validate manifest
2. `holder.py` + `warmup.py` — atomic swap, warmup 50 query
3. `search.py` — 2-tier với 1 vector name, chưa filter. **Bench ngay tại đây.**
4. `server.py` — nối vào gRPC, tắt `SC_STUB_MODE`
5. Filter + `choose_strategy` + giải mã IdSet 3 dạng
6. Multi-vector + `fusion.py` (RRF)
7. `diversity.py` (MMR, max_per_video, dedup)
8. `temporal.py` (TRAKE)
9. `points.py` (CRUD + tombstone) + `Optimize`
10. `sparse.py` (đắt nhất — cân nhắc hoãn sang giai đoạn 2)
11. `region.py`, `collections.py`

**BE**
1. `db/models.py` + migration + `sql/init/*.sql`
2. `clients/searchcore.py` — wrap stub, retry, deadline
3. `api/search.py` đường đơn giản nhất, không LLM không filter. **Xác nhận p50 < 60ms.**
4. `tools/warm_hydration_cache.py` + `services/hydrate.py`
5. `tools/bench.py`
6. `services/prefilter.py` (allowlist + chọn mã hoá IdSet)
7. `services/enrich.py` (LLM, chạy song song)
8. `api/ws.py` (streaming progressive)
9. `api/search.py` phần temporal/qa/region/similar
10. `services/ranking.py` (học lại trọng số từ click log)

**Nguyên tắc xuyên suốt**: mỗi bước chạy end-to-end với dữ liệu nhỏ trước khi mở rộng. Bench sau mỗi bước có ảnh hưởng latency.

---

## 8. Bẫy đã biết — đã vấp phải hoặc đã xác định

| Bẫy | Xử lý |
|---|---|
| **`IndexHNSW` không hỗ trợ `remove_ids`** | Tombstone bitset + lọc lúc search + `Optimize()` rebuild định kỳ |
| **Query đầu chậm 10–50×** | `warmup()` bắt buộc, 20–50 query giả. `vmtouch -t` cho mmap |
| **Unix socket `Permission denied`** | `os.chmod(socket_path, 0o666)` sau `server.start()`. Core uid 10001, BE uid 10002 |
| **Container core bị restart vòng lặp** | `start_period: 180s` trong healthcheck. Load 6.7GB + warmup mất 2–3 phút |
| **p99 dao động thất thường** | `faiss.omp_set_num_threads(4)` + giới hạn gRPC worker, tránh oversubscribe |
| **allowlist 1.2M ID = ~10MB/request** | `IdSet` dạng BITSET (250KB, nhỏ hơn 64×) |
| **HNSW lạc khi filter quá chọn lọc** | `FilterStrategy.EXACT_SUBSET` khi cardinality < 10k hoặc ratio < 1% |
| **Fusion dense + sparse bằng weighted sum** | Sai — hai thang điểm khác nhau. Dùng RRF (chỉ dựa thứ hạng) |
| **`docker compose` không nội suy `${}` trong `env_file`** | Viết thẳng giá trị vào `.env`, không dùng `${PG_PASS}` trong `DATABASE_URL` |
| **protoc sinh import tuyệt đối** | `gen_proto.sh` có bước `sed 's/^from searchcore\.v1 import/from . import/'` |
| **Tên image Docker phải viết thường** | `ghcr.io/kinhnguyenvan/...` dù username GitHub có chữ hoa |
| **pgvector giới hạn 2000 chiều** | Không dùng pgvector. FAISS quản vector hoàn toàn |
| **N+1 query khi hydrate** | Batch `WHERE id = ANY($1)` + `array_position` giữ thứ tự |
| **CRUD làm index trôi dần** | Sau ~20% thay đổi: latency +30–50%. Lên lịch `Optimize()` |
| **Nhiều collection nhân RAM** | Lazy load + LRU evict |

---

## 9. Biến môi trường

```bash
# Core
SC_SOCKET_PATH=/var/run/searchcore/sc.sock
SC_TCP_PORT=50051
SC_STUB_MODE=0                    # 1 = trả kết quả giả, chưa cần index
SNAPSHOT_DIR=/data/snapshots/current
ENCODER_PATH=/models/siglip-text.onnx
ENCODER_NAME=siglip-so400m-p14-384
VECTOR_DIM=3072
OMP_NUM_THREADS=4
FAISS_EF_SEARCH=64
RERANK_CANDIDATES=800
WARMUP_QUERIES=50
MAX_COLLECTIONS_IN_RAM=1

# BE
SEARCHCORE_TARGET=unix:///var/run/searchcore/sc.sock   # hoặc searchcore:50051
DATABASE_URL=postgresql+asyncpg://vs:<pass>@postgres:5432/rackfocus
REDIS_URL=redis://redis:6379/0
LLM_API_KEY=
S3_ENDPOINT= S3_BUCKET= S3_ACCESS_KEY= S3_SECRET_KEY= S3_PUBLIC_BASE=
```

---

## 10. Kiểm tra nghiệm thu

```bash
# 1. Core lên và load được snapshot thật
docker compose logs searchcore | grep -E "loaded|warmup|ready"

# 2. Health trả đúng version và point count
grpcurl -plaintext localhost:50051 searchcore.v1.AdminService/Health

# 3. Đường ống thông
curl -XPOST localhost:8000/api/search -H 'content-type: application/json' \
  -d '{"text":"người đi trên đường phố","top_k":10}' | jq '.timings'

# 4. Bench đạt mục tiêu
make bench       # p50 < 60ms, p95 < 150ms

# 5. Đối chiếu recall thật của HNSW
#    Chạy cùng query với debug.force_exact=true, so overlap top-10.
#    Kỳ vọng >= 95% với ef_search=64. Nếu thấp hơn, tăng ef_search hoặc rerank_candidates.
```