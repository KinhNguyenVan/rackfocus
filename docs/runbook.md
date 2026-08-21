# Runbook — chạy search core + BE

Thiết kế: [search-design.md](search-design.md). Tài liệu này chỉ trả lời hai câu:
**đã có gì** và **còn thiếu gì để server chạy thật**.

---

## 1. Đã xong

| Phần | Trạng thái |
|---|---|
| `proto/` | `Payload.{video_name,frame,keyframe_time}`, `Filter.tags`, `ResponseMeta.tags_used`, `AdminService.GetTagVocab`. Chỉ thêm field nên `buf breaking` pass |
| `core/config.py` | Có `dim` + `encoder_name` (trước đây thiếu → check hợp đồng snapshot là no-op) |
| `core/snapshot.py` | Load + **10 kiểm tra fail-closed** (dim, encoder, checksum, kích thước raw, row-alignment, tombstone, tag ngoài vocab, dtype, L2-norm, thiếu file) |
| `core/search.py` | Định tuyến theo cardinality, EXACT_SUBSET, 2-tier + `IDSelector`, tombstone **trước** rank, fallback untagged, diversity |
| `core/holder.py` | Atomic swap + single-flight |
| `core/encoder/` | ONNX text tower + canonicalization SigLIP + semaphore giới hạn encode |
| `core/{server,main,warmup,metrics}.py` | gRPC servicer, khởi động, warmup cả encoder lẫn EXACT_SUBSET, percentile |
| `be/` | `enrich.py` (litellm), `tagvocab.py` (cache pin theo `snapshot_ver`), `clients/searchcore.py`, `api/{search,health}.py` |
| `ingest/export_siglip_text_onnx.py` | Export text tower + gate verify bằng số |
| Test | **94 test**: core 58, BE 14, ingest 22. CI đã bật `pytest`, tắt `fail-fast` |
| Compose | Volume RW `sc_cache`, `start_period` 600s |

Notebook embed đã sửa: `load_shards` zero-copy (9.2× → 1×), `concat_tables` permissive,
check L2-norm, chặn snapshot thiếu group, không xoá local khi upload lỗi, ghi vector fp32.

---

## 2. Còn thiếu để chạy thật

Theo thứ tự. Không làm xong (1) và (2) thì server **không** search được.

### (0) Chốt lại encoder — CHƯA GIẢI QUYẾT

Bundle text tower **đã export và verify xong** (cosine PyTorch vs ONNX = 1,000000, norm
= 1,0). Nhưng đo thật thì latency **không đạt cả ngân sách đã nới**:

| | p50 | p95 | kích thước |
|---|---|---|---|
| fp32 | **688 ms** | 2215 ms | 1716 MB |
| int8 | 251 ms | 272 ms | 431 MB |

- fp32 dao động dữ dội (ba lần đo: 296 / 688 / 1024 ms, max 3301 ms). Thêm thread không
  cứu được: 1→4 thread chỉ nhanh 1,44×, 8 thread còn *chậm hơn*.
- **int8 nhanh 3× nhưng làm hỏng kết quả.** Đo trên 1897 keyframe embedding thật:
  overlap@10 chỉ **55%**, top-1 khớp **2/10**. Không dùng được.

Bốn đường còn lại, cần bạn chọn:

1. Chấp nhận ~700 ms – 1 s cho encode (vượt ngân sách 250–500 ms đã duyệt).
2. Thử `onnxruntime.transformers.optimizer` (fuse attention) — có thể 1,5–2× mà **không**
   đổi số học như int8. Chưa đo.
3. Đổi sang SigLIP nhỏ hơn (base: 12 layer, hidden 768, ~13 GFLOP) → phải **re-embed toàn
   corpus**.
4. Encoder chạy GPU — sửa bất biến I1 (luật do nhóm tự đặt).

Trong lúc chưa chốt, hệ thống vẫn chạy được: BE có thể tự gửi `vector`, hoặc dùng bundle
fp32 và chịu latency.

### (1) Bundle ONNX text tower — ĐÃ XONG (còn upload)

`export_siglip_onnx.py` cố ý chỉ export vision (`get_image_features`). Core **không encode
được text** cho tới khi có bundle text.

```bash
cd services/ingest
PYTHONPATH=src python -m ingest.export_siglip_text_onnx \
    --model google/siglip-so400m-patch14-384 \
    --output-dir /tmp/siglip-text-onnx \
    --sample-image /đường/dẫn/ảnh-thật.webp \
    --captions "caption đúng của ảnh" "caption sai 1" "caption sai 2" \
    --s3-uri s3://aic-bucket-2026/models/siglip-so400m-patch14-384-text-onnx
```

`--sample-image` **không phải tuỳ chọn**: so PyTorch-vs-ONNX ≥0.999 chỉ chứng minh export
trung thực với PyTorch, **không** chứng minh đã export đúng nhánh. Gate retrieval mới bắt
được lỗi đó. Đừng kỳ vọng cosine lớn — SigLIP có logit scale + bias nên cặp ảnh–caption
khớp chỉ ~0.05–0.15.

Cần package **`onnx`**, không phải chỉ `onnxruntime`: `torch.onnx.export(dynamo=False)`
gọi tới nó và báo `OnnxExporterError: Module onnx is not installed!` nếu thiếu.
`requirements.txt` trước đây không khai — đã thêm. Lỗi này áp dụng cho **cả**
`export_siglip_onnx.py` (vision) đã có sẵn trong repo.

Tải model lần đầu ~3,3 GB vào `~/.cache/huggingface`.

### (2) Snapshot có đủ group — BLOCKER

Chạy cell build-index trong `aic-embed-siglip-2026.ipynb` với `SNAPSHOT_GROUPS` là **toàn
bộ** group, rồi upload lên `snapshots/v{N}/`.

Notebook giờ sẽ raise nếu số video trong snapshot khác số video thật trên S3 — trước đây
chạy tuần tự từ trên xuống cho snapshot chỉ 3/30 group mà **mọi** con số vẫn tự nhất quán.

### (3) `tags.npy` + `tag_vocab.json` — không chặn

Chưa có thì core vẫn chạy, chỉ là mọi query search toàn bộ corpus (`tags_used=[]`).

Tag = `domain_id` (13 giá trị cố định của `Domain` enum trong
`ingest/domain/models.py`, không phải `topic_id` — quyết định vì domain enrichment chỉ
gán domain/topic theo **scene**, và tag phải phân hoạch corpus nên chọn tầng thô hơn,
ổn định hơn). Sinh bằng:

```bash
python -m ingest.build_tags --snapshot-dir /path/to/snapshots/v1
```

(`services/ingest/src/ingest/build_tags.py`, cần `MONGO_URI`/`MONGO_DB` trong `.env` —
đọc `domain_jobs`/`scene_domain_map` do `python -m ingest.domain` ghi ra.)

- Join bằng `(payload.video_name, payload.scene_idx)` → `scene_domain_map` của
  **analysis đang active** cho video đó (`DomainRepository.active_domain_by_scene`) — không
  qua `idmap.npy`/`point_id`, vì `scene_idx` đã cùng không gian id với `scene_id` mà domain
  enrichment dùng (`assign_scene_idx` ở notebook embed), không cần nội suy theo frame.
- Row/video/scene không có analysis active nào → sentinel **65535** (không phải 255; vocab
  chỉ 13 tag nhưng sentinel giữ nguyên theo hợp đồng của core).
- In ra tỉ lệ phủ tag + phân bố theo domain — chạy xong đọc log này trước khi tin dữ liệu.
- Tự thêm `tags.npy` + `tag_vocab.json` vào `manifest.checksums` (dùng `--no-manifest` nếu
  muốn tự làm bước đó).
- **Phải chạy `python -m ingest.domain` (phân đoạn domain) cho toàn bộ group xong trước** —
  video nào chưa được domain enrichment chạm tới thì toàn bộ row của video đó thành sentinel,
  tức là bị loại khỏi 100% query có tag.

### (4) Cấu hình `.env`

```bash
SC_STUB_MODE=0                 # 1 = trả kết quả giả, /readyz phơi cờ này ra
SNAPSHOT_S3=s3://aic-bucket-2026/snapshots/v1
ENCODER_S3=s3://aic-bucket-2026/models/siglip-so400m-patch14-384-text-onnx
ENCODER_NAME=siglip-so400m-patch14-384
VECTOR_DIM=1152                # đã đo trên dữ liệu thật, KHÔNG phải 3072
MODEL_CACHE_DIR=/var/cache/searchcore
LLM_MODEL=groq/llama-3.3-70b-versatile
LLM_API_KEY=...
```

`SNAPSHOT_DIR` (bind mount) **thắng** `SNAPSHOT_S3` nếu có cả hai — core log cảnh báo.

### (5) Chạy

```bash
make proto        # stub gRPC là artifact gitignore, thiếu là core/be không import được
make up
curl localhost:8000/readyz          # đợi ready=true (khởi động lạnh: tải ~3.7GB + warmup)
curl localhost:8000/api/tags
curl -X POST localhost:8000/api/search \
     -H 'content-type: application/json' \
     -d '{"text":"cầu thủ ăn mừng","top_k":10}'
```

---

## 3. Số đo trên dữ liệu thật

Từ `embed_000.parquet` (1897 keyframe, video L29_V001):

| Đại lượng | Giá trị |
|---|---|
| **dim** | **1152** — Handoff ghi 3072, sai. RAM §6 phải tính lại |
| L2-norm | lệch tối đa **1,05e-06** → loader chấp nhận |
| Sau khi ép fp16 | lệch **1,09e-04**, vẫn dưới ngưỡng 1e-3 |
| Cosine nội bộ | 0,63 — embedding tập trung, đúng như dự đoán |
| `objects` | `list<null>` (stage chưa code) → **bắt buộc** `promote_options="permissive"` khi concat |
| Vector trong parquet | `list<double>` — xem §4 |

Đo hiệu năng (dim=1152, gather fp16 + matmul, Apple Silicon; server 4 vCPU chậm hơn):
8,5k candidate = **7,4 ms**; 20k = 12,6 ms; 50k = 63 ms; 250k = **256 ms**.
Chi phí là **băng thông bộ nhớ**, không phải FLOPs → đường không-tag buộc phải dùng HNSW.

---

## 4. Nợ kỹ thuật đã biết

| Vấn đề | Ảnh hưởng |
|---|---|
| **Scene dài trung vị 130,6s, tối đa 157,6s** (1897 keyframe chỉ gom thành **13 scene**) | `keyframe_time` lệch `start_sec` trung vị **59,6s**, tối đa **157,2s**. Player seek theo `start_sec` sẽ nhảy tới **2,6 phút** khỏi frame đã khớp. `Filter.min_start_sec/max_end_sec` gần như vô dụng, `has_speech` thành gần như per-video. Đây là vấn đề **chất lượng gom scene của BaSSL**, không phải của search |
| Parquet ghi vector `list<double>` | Đã sửa writer sang fp32 (gộp được với shard cũ). Shard **đã ghi** vẫn fp64 → 4,29 GB thay vì 2,15 GB ở 500k, và load phải ép kiểu (2× bộ nhớ). Re-embed hoặc chấp nhận |
| Notebook build snapshot chưa track git | Format snapshot là hợp đồng offline↔online thứ hai, ngang `.proto`. `.proto` có `buf breaking` canh; snapshot chỉ có test canh. `build_index.py` vẫn là stub 1 dòng |
| `media.py` sinh path tương đối, notebook sinh URL tuyệt đối | Hai đường lệch nhau, cần gộp |
| Ngân sách latency | 250–500 ms, **không** phải 100–200 ms như Handoff §5. `deadline_ms=200` ở §4.2 sẽ cho 100% DEADLINE_EXCEEDED |
| Ngoài phạm vi v1 | VKIS (cần vision tower trong core), QA (`has_ocr` toàn False), TRAKE, tier SCENE, sparse/BM25 |
