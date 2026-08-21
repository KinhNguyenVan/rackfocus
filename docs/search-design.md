# Thiết kế Search — core + BE (v1)

Đã qua design review đa tác nhân (Skeptic / Constraint Guardian / User Advocate).
Decision Log: 37 quyết định, 5 objection bị từ chối. Tài liệu này là bản **sau** review —
mọi con số ở đây đã được đo hoặc tính, không phải ước lượng.

Phạm vi v1: **text KIS**. VKIS/QA/TRAKE **ngoài phạm vi** (xem §9) — nói rõ để không ai
tưởng chúng chạy được.

---

## 1. Ngân sách latency — đã sửa

Handoff §5 khai "encode 10–30ms, tổng không LLM 39–110ms" và §10 gate `p50 < 60ms`.
**Cả ba đều không đạt được** và đã bị thay bằng số dưới đây.

SigLIP so400m text tower = 449,3M param (hidden 1152, ffn 4304, 27 layer).
FLOPs = **53,3 GFLOP/query**, **cố định** vì SigLIP buộc `padding="max_length", max_length=64`
— query 3 chữ trả giá y như query 60 token.

| Thành phần | Ngân sách |
|---|---|
| LLM enrich + chọn tag | 80–300 ms (song song với encode, xem §3) |
| Encode text (so400m fp32, 4 core) | **170–420 ms** |
| Vector search (EXACT_SUBSET 8,5k, đo được) | 7 ms |
| Vector search (2-tier, không tag) | 5–15 ms |
| Hydrate | 0 (payload inline, §6) |
| **Tổng** | **250–500 ms** |

`deadline_ms` phải suy từ p99 encode đo thật, **không** để 200 như Handoff §4.2 (sẽ khiến
100% request DEADLINE_EXCEEDED). Quyết định: giữ so400m, không re-embed, nới ngân sách.

### Số đo THẬT (bundle đã export, 4 performance core Apple Silicon)

| | p50 | p95 | GFLOP/s | kích thước |
|---|---|---|---|---|
| fp32 | **688 ms** | **2215 ms** | 77 | 1716 MB |
| int8 (dynamic) | **251 ms** | 272 ms | 213 | 431 MB |

fp32 **dao động rất mạnh** (max đo được 3301ms; ba lần đo riêng cho 296 / 688 / 1024 ms).
Model 1,7GB nhạy với áp lực bộ nhớ. Scaling thread cũng tệ: 1→4 thread chỉ nhanh 1,44×,
và 8 thread *chậm hơn* 4 thread — GEMM gầy (M=64, batch 1) + 27 rào đồng bộ, bound bởi
bộ nhớ chứ không phải compute. Đừng kỳ vọng thêm core sẽ cứu được.

Bác bỏ ba mitigation đã xét:
- Cache theo hash query: **hit rate 0%** trên request quan trọng nhất — Handoff I7 viết
  "trong thi đấu query đầu là query tính điểm", query thi đấu là mới theo định nghĩa.
- **int8: ĐÃ ĐO, KHÔNG DÙNG ĐƯỢC.** Nhanh 3,07× (251ms) nhưng làm lệch kết quả. Đo trên
  1897 keyframe embedding thật với 10 query tiếng Việt: **overlap@10 chỉ 55%**,
  overlap@50 64%, **top-1 khớp 2/10**. Cosine giữa text embedding fp32 và int8 thấp nhất
  0,939 — đủ để đẩy query ra khỏi manifold mà image embedding fp32 được dựng trên đó.
- Text tower nhỏ hơn: hai tower SigLIP train chung, **không hoán đổi được** → là re-embed.

---

## 2. Snapshot — định dạng và kiểm tra khi load

Do notebook `aic-embed-siglip-2026.ipynb` build. Core **chỉ đọc**.

```
snapshots/v{N}/
├── manifest.json     hợp đồng offline↔online
├── visual.faiss      IndexHNSWSQ (HNSW32,SQ8) METRIC_INNER_PRODUCT   — coarse, in RAM
├── visual.f16        raw fp16 row-major N×dim, KHÔNG header          — rerank, mmap
├── idmap.npy         int64[N] row -> point_id
├── payload.parquet   metadata, row-aligned, không có vector
├── tombstone.bin     bitset ceil(N/8) byte
└── tags.npy          uint16[N] row -> tag_id. Sentinel 65535 = chưa gán   ← thêm
    tag_vocab.json    {tag_id: {name, description}}                        ← thêm
```

**Không** có `tag_index.npz`. CSR dựng lúc load bằng `argsort(tags)` + `bincount` —
~30–50ms ở N=500k, không đáng thành file thứ ba phải pin integrity.

`metric: "cosine"` trong manifest nhưng index là `METRIC_INNER_PRODUCT`. Hai cái **chỉ**
tương đương vì vector đã L2-normalize. Core **không được** normalize lại.

### Kiểm tra bắt buộc khi load (fail thì từ chối, không degrade)

| Kiểm tra | Vì sao |
|---|---|
| `manifest.dim == cfg.dim` và `manifest.encoder_name == cfg.encoder_name` | Hợp đồng. `config.py` hiện **không có** hai field này → check đang là no-op, phải thêm |
| **sha256 mọi file khớp `manifest.checksums`** | `visual.f16` không header → size check pass với *bất kỳ* file cùng độ dài. Trộn `visual.faiss` build #2 với `visual.f16` build #1 cho mọi điểm là dot product với vector của frame khác, **mọi validation khác đều pass**. Đo: 1,9GB = 1–4s trên start_period 180s (0,6–2%) → **không phải tuỳ chọn** |
| `getsize(visual.f16) == count*dim*2` | Bắt file raw bị cắt |
| `len(idmap) == len(tags) == payload.num_rows == count` | Row-alignment giữa 5 file |
| `len(tombstone) == ceil(count/8)` | Thiếu → IndexError phụ thuộc dữ liệu, chỉ ở một số query |
| `tags.max() < len(vocab)` hoặc `== 65535` | Sentinel không được đè lên tag hợp lệ |
| `‖v‖ ≈ 1` trên **toàn bộ** `visual.f16` | Group nào embed bằng bundle chưa bake L2 có norm ~15 và **thống trị mọi query** bất kể nội dung. Lấy mẫu sẽ bỏ sót vài nghìn row trên 500k → check hết, 1 pass |
| `manifest.groups` khớp danh sách group kỳ vọng | Snapshot 3/30 group vẫn tự nhất quán → core không thể tự phát hiện |

### tags.npy phải join qua point_id, không theo vị trí

Row order = `sorted(glob(embed_*.parquet))`, **không** ghi trong manifest (giờ đã ghi
`row_order` + `groups` + `point_id_scheme`). Rebuild lệch một video là mọi tag lệch mà
`len(tags) == count` vẫn pass — sai âm thầm, không exception, không metric nào động.
Nên: sinh `tags.npy` bằng join `point_id` từ `idmap.npy`, và đưa `tags.npy` +
`tag_vocab.json` vào `manifest.checksums`.

**Đã implement** (`ingest/build_tags.py`, tag = `domain_id`), ba chặng:

```
(payload.video_name, payload.frame) --[Maps_*/maps/keyframe_scene_map.json]--> scene_id
scene_id                            --[DomainRepository.active_domain_by_scene]--> domain_id
domain_id                           --[Domain enum, thứ tự cố định]--> tag_id
```

Chặng 1 đọc **map đã validate** do `AIC_KeyframeSceneMap.ipynb` sinh (bisect theo
`start_frame`, đối chiếu chéo timestamp, tự `assert nearest == 0`/`unmatched == 0` trước
khi ghi file) — KHÔNG dùng thẳng cột `payload.scene_idx`, vì cột đó (`assign_scene_idx`
trong notebook embed) là một vòng lặp không có bước kiểm nào: không phát hiện được
keyframe rơi ngoài mọi scene hay lệch do fps lẻ.

`video_name`/`frame` đọc **trực tiếp từ `payload.parquet` của chính snapshot đang build
tag cho nó** ở chặng 1 — không rebuild danh sách theo thứ tự riêng rồi ghép lại. Do đó
không rơi vào bẫy "sai theo vị trí" ở trên: không có bước trung gian nào mà row order có
thể lệch khỏi payload/idmap hiện tại, vì tag được tính ngay trên chính cột đã row-aligned
đó. Vẫn tự thêm `tags.npy`+`tag_vocab.json` vào `manifest.checksums` như yêu cầu.

### Bất đẳng thức bắt buộc

```
count × dim × 2  ≤  (giới hạn cgroup − anon RSS)
```
`visual.f16` gather chỉ nhanh khi nằm trong page cache. Không cached: ~4800 major fault
× ~80µs = **~380ms**. Ngoài bất đẳng thức này **mọi số latency trong tài liệu vô nghĩa**.

Ở N=500k, dim=1152: anon ~4,2 GiB + f16 cache 1,07 GiB = **5,24 GiB / 12 GiB = 44%** → đủ.
Ở 2M×1152 là 117% kể cả hot-swap → phải nâng limit hoặc chốt lại N.

RAM mỗi row (đừng chia Handoff §6 cho một hệ số — các hạng mục cố định không co theo dim):
`SQ8 = dim` + `graph HNSW = 272 B` + `payload Arrow = 268 B` + `idmap 8` + `tags 2`.
Ở dim=1152 các hạng mục cố định chiếm **48%**.

---

## 3. Luồng — encode song song với LLM

```
BE  POST /api/search {text, top_k, use_llm=true}
     ├─ (song song) ─┬─ gRPC Encode(text=text_GỐC_của_user)
     │               └─ LLM(text + tag_vocab) -> {tags[]}          80–300ms
     ├─ chờ cả hai:  tổng = max(LLM, encode), KHÔNG phải tổng cộng
     └─ gRPC Search(vector=<đã encode>, filter.tags=tags, top_k)
core ├─ candidate = concat(csr[t] for t in tags)   |  toàn bộ nếu tags rỗng
     ├─ lọc tombstone TRƯỚC khi rank
     ├─ choose_strategy(len(candidate))
     ├─ EXACT_SUBSET: refine_fp16[cand] @ q -> argpartition top_k
     │  hoặc 2-tier:  HNSW(ef) -> gather fp16 -> argpartition
     ├─ diversity: dedup cosine + max_per_shot + min_time_gap
     └─ meta{tags_used, candidate_count, filter_strategy_used, snapshot_ver}
```

Hai lợi ích của việc encode **query gốc** song song:
1. Tổng latency giảm 80–300ms.
2. Giữ đúng cách diễn đạt của người dùng. LLM chỉ dùng để **chọn tag**, không viết lại
   query — một competitor biết domain ("Honda Cub xanh, thấy biển số") sẽ phụ thuộc vào
   chính chữ họ gõ.

---

## 4. Tag filter

Mỗi frame đúng **một** tag → tag *phân hoạch* corpus. Candidate = nối các bucket CSR,
không scan N, không bitset.

**Rút lại tuyên bố "vấn đề IdSet biến mất"** — chỉ đúng ở nhánh nhỏ. Nhánh lớn dùng
`IDSelectorBitmap`, tức bitset **quay lại trong hot path**, dựng lại mỗi request.

### Định tuyến theo cardinality

Một ngưỡng duy nhất `EXACT_SUBSET_MAX` (mặc định 20 000, **phải đo mà chốt** — trước đây
tài liệu ghi 20k rồi lại nói "đúng như §3.5 (≤10 000)", hai số khác nhau trình bày như một).

Số đo thật (dim=1152, gather fp16 + matmul + top-100, Apple Silicon; server 4 vCPU chậm hơn):

| candidate | thời gian | GFLOP/s |
|---|---|---|
| 1 000 | 0,6 ms | 4,1 |
| 8 500 | **7,4 ms** | 2,7 |
| 20 000 | 12,6 ms | 3,6 |
| 50 000 | 63,1 ms | 1,8 |
| 250 000 | 256,5 ms | 2,2 |

Chi phí là **băng thông bộ nhớ**, không phải FLOPs: gather 19,6MB fp16 → ép fp32 39MB →
matmul đọc lại 39MB. Nên chỉ đạt 2,7 GFLOP/s. Quét toàn bộ 250k mất 256ms → **đường
không-tag buộc phải dùng HNSW**.

### Phân bố tag không đều

"1,7k frame/tag" là **trung bình**. Phân bố thực tế head-heavy, **và** LLM chọn tag rộng
thường xuyên hơn vì tag rộng liên quan tới nhiều query hơn. Nên:
- Đo phân bố thật, log `candidate_count` mỗi query.
- Short-circuit: `len(candidate) == count` → bỏ filter, đi thẳng đường không-tag (nếu
  không sẽ chậm hơn cả không filter, vì thêm selector match-everything).
- Row chưa gán (sentinel 65535) **không thuộc bucket nào** → mọi query có tag đều loại
  100% số đó. Phải log tỉ lệ phủ tag.

### Ngữ nghĩa `Filter.tags`

`tags` **AND** với mọi field khác của `Filter` (`allow`/`deny`, `video_ids`,
`min_start_sec`, …). Trước đây pseudocode chỉ honour `tags` và **âm thầm bỏ `allow`**.
Bounds-check: `tags` là uint32 còn `tags.npy` là uint16 → LLM ảo giác tag 70000 phải trả
`INVALID_ARGUMENT`, không phải IndexError → 500.

### Tag sai làm frame đúng KHÔNG THỂ với tới

Đây là tính chất xấu nhất của phân hoạch: filter là cứng, không phải hạ hạng. Recall hệ
thống = P(tag đúng) × 100%, và **P(tag đúng) hoàn toàn chưa đo**. `EXACT_SUBSET` vẫn trả
đủ `top_k` từ subset sai → trang kết quả tự tin từ 1,7% corpus không chứa đáp án.

Bắt buộc có:
1. `ResponseMeta` trả `tags_used`, `candidate_count`, `filter_strategy_used` → BE hiển thị.
2. **Tự động fallback**: tagged trả < `top_k` hit trên `min_score` → chạy lại untagged,
   đánh dấu `warnings=["tag_fallback"]`.
3. Giữ cờ `use_llm` (Handoff §4.2 vốn có) để người dùng tắt hẳn tag.

---

## 5. Encoder text — cần export mới

`export_siglip_onnx.py` dòng 8: *"Chỉ export `get_image_features` (không export nhánh text)"*.
Bundle hiện tại **không encode được text**.

```
models/siglip-so400m-patch14-384-onnx/       vision — đã có, cho ingest
models/siglip-so400m-patch14-384-text-onnx/  text   — CẦN THÊM, cho core
    model.onnx   input_ids[batch,64] -> text_embeds (L2-normalize bake trong graph)
    tokenizer.json / spiece.model + tokenizer_config.json
```

Ba bẫy:
1. `padding="max_length", max_length=64` — **bắt buộc**, khác CLIP. Dynamic padding ra vector sai.
2. L2-normalize bake vào graph, giống `_ImageEncoder`.
3. **Canonicalization**: SigLIP chuẩn hoá text (lowercase, dấu câu) trước tokenize. Bỏ
   qua thì query lệch nhẹ khỏi manifold của 500k image embedding — không lỗi, không crash,
   rất dễ bị quy oan cho "tag tệ".

Gate verify **bằng số**, không phải "cosine phải cao": cosine thật của cặp ảnh–caption
khớp ở SigLIP là **0,05–0,15** (có logit scale + bias), nên "cao" là bẫy hai chiều. Dùng:
- PyTorch vs ONNX ≥ 0,999 (giống gate vision đã có), **và**
- retrieval accuracy trên bộ ảnh–caption đã biết (top-1 phải đúng).

---

## 6. Payload trả inline — bỏ hydration Redis ở v1

`payload.parquet` row-aligned, `keyframe_key`/`clip_key` **là URL S3 đầy đủ** (do
frame-cut notebook sinh; lưu ý `media.py` của pipeline cũ sinh path tương đối — hai
đường lệch nhau, cần gộp). Core gather payload theo row → BE không cần Postgres lẫn Redis.

268 B/row dạng Arrow columnar; **~1300 B/row nếu dựng dict Python**. Bắt buộc truy cập
zero-copy columnar. 69% là hai URL, trong đó 110 B/row là prefix lặp.

### Proto: thêm field vào `Payload`

`Payload` hiện có `video_id` (uint64) nhưng payload thật chỉ có `video_name` (string).
Thiếu field làm 4 tính năng hợp đồng thành no-op hoặc trả sai:

| Thêm | Vì sao |
|---|---|
| `video_name` (string) | Payload thật dùng string. **Không có `video_id` thì `Diversity.max_per_video` group theo 0 → `top_k=50` trả về 3.** `Filter.video_ids`, `SearchGrouped(group_by=video_id)`, `exclude_same_video` cũng vô hiệu |
| `frame` (int64) | Cần để nộp bài. `point_id` là blake2b64 **không phân giải được** |
| `keyframe_time` (double) | `start_sec`/`end_sec` là thời gian của **scene**. Scene gồm nhiều shot → seek theo `start_sec` lệch hàng chục giây, đúng lúc verify là bottleneck thật của KIS |
| `shot_id` có nguồn | Đang có trong proto nhưng payload không sinh → `Diversity.max_per_shot` không tính được. Lấy từ `shots.csv` |

### `tag_vocab` cần RPC

`tag_vocab.json` nằm trong snapshot của core, nhưng BE cần nó để gọi LLM. Thêm
`AdminService.GetTagVocab` trả vocab **kèm `snapshot_ver`** để BE pin đúng bản.
Đọc trực tiếp thư mục snapshot từ BE là phá service boundary và phá atomic swap.

Lưu ý chi phí chưa tính: 500 mô tả tag ≈ **15–25k prompt token mỗi query**.

**Đã chốt: tag = `domain_id`, tức chỉ 13 giá trị cố định** (`Domain` enum, không phải
`topic_id` ~50 giá trị) — chi phí prompt thực tế chỉ còn vài trăm token/query, không phải
15–25k. Đánh đổi: filter thô hơn hẳn (mỗi tag gộp nhiều chủ đề con), candidate/tag lớn hơn
nhiều so với giả định "1,7k frame/tag" ở §4 (13 tag chia đều 500k thì ~38k frame/tag) —
`EXACT_SUBSET_MAX` (mặc định 20 000, xem §4) sẽ bị vượt bởi hầu hết tag đơn, nên phần lớn
query có tag vẫn đi nhánh HNSW hai tầng, không phải nhánh brute-force rẻ hơn.

---

## 7. Đồng thời, warmup, swap

| Vấn đề | Xử lý |
|---|---|
| `ef_search` là per-request trong proto nhưng chỉ áp được bằng `index.hnsw.efSearch = N` — **mutable state trên object dùng chung 8 thread**. Race: request audit ef=256 chạy thật ở ef=32, báo 71% recall, không exception, làm gate §10 không thể falsify | **Không cho per-request**. Đặt theo snapshot lúc load. Nếu cần per-request thì phải lock quanh (set + search) |
| 8 gRPC worker × 4 BLAS thread trên 4 vCPU = **8× oversubscription**, cộng encoder 53 GFLOP mỗi request. `OMP_NUM_THREADS` **không** điều khiển OpenBLAS (cần `OPENBLAS_NUM_THREADS`) | Semaphore giới hạn **1–2 encode đồng thời**; khai báo queue depth + policy khi bão hoà; set cả hai biến env |
| Warmup §3.11 chỉ warm 2-tier — **không** warm ONNX encoder (chỗ đắt nhất, first-run còn phải cấp arena + pre-pack weight) và **không** warm nhánh EXACT_SUBSET | Warm cả ba. Nghịch lý phải chấp nhận: warm encoder tử tế cộng thêm ~10–20s startup |
| `download_dir` cache marker **không có invalidation**; prefix `snapshots/v1` hardcode → re-upload cùng prefix cho split-brain: host có marker serve bản cũ mãi mãi, `snapshot_ver` báo "1" ở cả hai | Prefix **version-immutable**; cache path key theo version; swap luôn là thư mục **mới**; load **single-flight** (hai lần load chồng nhau giữ 3× snapshot → OOM) |
| Cả hai mount trong compose đều `:ro` → `download_dir` rơi về `/tmp` ephemeral → tải lại 3,7–9,4GB mỗi lần recreate; nếu `/tmp` là tmpfs thì lượng đó tính vào cgroup 12GiB → nổ | **Named RW volume** cho `MODEL_CACHE_DIR`; thêm `MODEL_CACHE_DIR`, `SNAPSHOT_S3`, `ENCODER_S3`, `ENCODER_NAME`, `VECTOR_DIM` vào `.env.example` |
| Tombstone lọc **sau** `top_k` → `top_k=10` với 3 điểm đã xoá trả về 7 | Lọc **trước** khi rank. Chốt bit order LSB-first, ghi vào manifest |

---

## 8. fp16 — "đủ chính xác", không phải "exact"

Đo được (dim=1152, 100 query):

| cosine nội bộ | N | đổi TẬP top-10 | đổi THỨ TỰ | gap rank10–11 | sai số fp16 |
|---|---|---|---|---|---|
| 0,0 | 50k | 2/100 | 5/100 | 4,1e-4 | 4,1e-6 |
| 0,8 | 50k | 0/100 | 5/100 | 2,0e-4 | 4,1e-6 |

Sai số nhỏ hơn gap rank10–11 **50–170×** → flip chỉ giữa candidate gần như đồng điểm.
Dùng được, nhưng:
- "exact" chỉ có nghĩa **không bỏ sót candidate**, không phải chính xác số học.
- **Không** dùng điểm fp16 làm `min_score` tuyệt đối mà bỏ qua sai số này.
- Gate §10 dùng `debug.force_exact` làm ground truth là **sai**: refine store *chính là*
  fp16, fp32 không nằm trong snapshot. Nó đo HNSW-vs-fp16, trộn lẫn sai số ANN với sai số
  lượng tử hoá. Muốn có fp32 truth phải đọc lại parquet.
- Keyframe liên tiếp trong một shot chênh nhau ~1e-5 = trong khoảng nhiễu fp16 → cái nào
  sống sót `dedup_threshold` là gần như tuỳ ý; test pin `score_exact` sẽ flake.

---

## 9. Ngoài phạm vi v1 (nói rõ để không ai tưởng đã có)

| Task | Thiếu gì |
|---|---|
| VKIS | Cần vision tower trong core để encode ảnh/clip mẫu. §5 chỉ bundle text tower. `SearchSimilar(image/clip)` và `RegionSelector` không có encoder phía sau |
| QA | `objects=[]`, `has_ocr=False` toàn bộ (stage chưa code) → `Filter.require_ocr` trả **0 kết quả** trên corpus 500k, không phân biệt được với "không có scene nào như vậy" |
| TRAKE | Cần `SearchTemporal`. Trên corpus phân hoạch, **mọi** event phải rơi vào bucket đã chọn để chuỗi sống → một tag sai giết cả chuỗi |
| Tier SCENE | Chỉ mới embed tier KEYFRAME |
| sparse / BM25 | Chưa code |

Rủi ro thứ tự: đường **không-tag** chạy được ngay (chưa có dữ liệu tag) nên nó là đường
được test; đường **có tag** mới là đường sẽ live. Hai đường khác nhau về recall, latency
và nguồn gốc điểm số.

---

## 10. Nghiệm thu

1. Core load snapshot thật, **từ chối** đúng khi: sai dim, sai encoder_name, checksum lệch,
   `visual.f16` sai kích thước, vector không normalize, thiếu group.
2. `Encode` trả vector dim khớp manifest; PyTorch vs ONNX ≥ 0,999; retrieval top-1 đúng
   trên bộ ảnh–caption mẫu.
3. Search không tag: p50/p95 đo được, so recall với brute-force fp16 trên **cùng** refine
   store (không dùng `force_exact` làm fp32 truth).
4. Search có tag: `candidate_count` và `filter_strategy_used` khớp `indptr`; fallback
   untagged kích hoạt đúng khi tagged trả thiếu.
5. Đồng thời: 8 request song song không vượt semaphore encode; p99 không sụp.
6. Swap: load snapshot mới trong khi có request đang chạy → request cũ vẫn đúng, không
   crash, không đọc lẫn file.
