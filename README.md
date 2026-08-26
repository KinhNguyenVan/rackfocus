# Multimodal Video Search Platform

Truy vấn video bằng ngôn ngữ tự nhiên. Ngân sách latency phần search: **100–200ms**.
Quy mô tham chiếu: ~2M vector × 3072-dim.

## Ba nguyên tắc không thương lượng

1. **Offline và online tách tuyệt đối.** Model nặng chạy trên GPU thuê theo giờ rồi tắt. Hot path chỉ có CPU.
2. **Hot path nằm trên một máy.** BE, search core, encoder cùng host, nói chuyện qua Unix socket (~0.1ms). Mỗi lần vượt biên network là cộng 50–200ms.
3. **Embedding có ba bản.** fp32 trên R2 để rebuild. SQ8 trong RAM để search thô. fp16 trên NVMe để rerank chính xác.

## Cấu trúc

```
proto/          Hợp đồng gRPC giữa BE và core — chốt trước, đổi phải review cả 2 phía
services/core/  Search core: gRPC + FAISS + encoder. Khởi động 2-3 phút, ít deploy
services/be/    BE gateway: FastAPI, LLM enrich, fusion, hydrate. Deploy liên tục
services/ingest/Pipeline offline. Không chạy trong compose production
services/fe/    Frontend React
sql/            Schema + query đã tối ưu
scripts/        Thao tác vận hành
docs/decisions/ ADR — vì sao chọn thế này, đọc trước khi đề xuất đổi
```

## Vì sao core tách container riêng mà không tách repo

- **Cùng repo**: `.proto` là hợp đồng chung, đổi API sửa một PR thay vì PR chéo hai repo.
- **Khác container**: core load 6.7GB index + warmup mất 2–3 phút. BE deploy vài lần một ngày. Chung container thì mỗi lần sửa BE phải reload index.
- **Cùng máy**: Unix socket 0.1ms. Tách sang hai nhà cung cấp là mất trắng mọi tối ưu.

## Chạy local

### 0. Chuẩn bị (bắt buộc, cả hai cách bên dưới)

```bash
cp .env.example .env      # rồi mở .env điền key — xem bảng "Biến phải tự điền"
make proto                # sinh stub gRPC vào services/{core,be}/.../pb
```

`make proto` phải chạy **trước tiên**: stub gRPC là artifact sinh lúc build, đã
gitignore. Thiếu nó thì core và be không import được gì cả.

**Không cần build lại snapshot.** Snapshot + tags đã nằm trên S3
(`s3://aic-bucket-2026/snapshots/v1`, 612.975 vector / 785 video). Core tự tải và
cache vào `MODEL_CACHE_DIR` — chỉ máy đầu tiên build và upload, mọi máy khác chỉ tải về.
Lần chạy đầu tải ~4GB (1.8GB encoder + snapshot) mất vài phút, các lần sau đọc cache.

> **Cache không tự invalidate.** Mỗi thư mục cache được đánh dấu bằng file rỗng
> `.s3_download_done`; thấy marker là core dùng luôn bản local, **không** so lại với S3
> ([encoder/base.py](services/core/src/searchcore/encoder/base.py)). Nếu nội dung trên S3
> đổi mà đường dẫn giữ nguyên — build đè lên `snapshots/v1`, việc đã từng xảy ra ở dự án
> này — thì máy có marker sẽ serve bản cũ vĩnh viễn, im lặng, không log gì.
>
> Vì vậy khi build lại snapshot: **bump version** (`snapshots/v2`) chứ đừng ghi đè, hoặc
> xoá tay `rm -rf .cache/searchcore` trên từng máy.

Đặt `MODEL_CACHE_DIR` vào đường dẫn **bền** (như `.cache/` trong repo, đã gitignore).
Trỏ vào `/tmp` thì cache mất mỗi lần reboot/dọn temp và phải tải lại 4GB.

#### Biến phải tự điền vào `.env`

| Biến | Vì sao cần | Thiếu thì sao |
|---|---|---|
| `AWS_ACCESS_KEY`, `AWS_SECRET_KEY` | Core tải snapshot/encoder; BE `/api/neighbors` | Core không nạp nổi snapshot, `/api/neighbors` trả 502 |
| `GROQ_API_KEY` (hoặc `LLM_API_KEY`) | LLM chọn tag + enrich query | Vẫn chạy: BE fallback search toàn corpus, không lọc tag |
| `PG_PASS` | Postgres của compose | Chỉ cần khi chạy compose |

Không có LLM key vẫn search được — bật toggle **"Search toàn bộ, không lọc — tắt LLM"**
trên UI, hoặc đặt `LLM_ENABLED=0`.

### Cách A — chạy trực tiếp bằng venv (nhanh, dùng khi dev/thi)

Ba service, ba terminal. Chạy **từ thư mục gốc repo** (BE đọc `.env` tương đối với CWD).

```bash
python3.11 -m venv .venv && . .venv/bin/activate
pip install -r services/be/requirements.txt -r services/core/requirements.txt
```

Bước này tải khoảng 2GB: bundle encoder trên S3 chưa có `tokenizer.json` nên core phải
dùng `transformers` + `sentencepiece` thay cho `tokenizers` (nhẹ hơn nhiều). Thêm
`tokenizer.json` vào bundle là bỏ được cả hai — xem chú thích trong
`services/core/requirements.txt`.

**Terminal 1 — core** (khởi động 2–3 phút: tải + nạp index + warmup):

```bash
set -a; . ./.env; set +a          # core KHÔNG tự đọc .env, phải export tay
export SC_STUB_MODE=0             # 0 = dùng snapshot thật, KHÔNG phải kết quả giả
export SC_TCP_PORT=50051          # local không có Unix socket dùng chung
export MODEL_CACHE_DIR="$PWD/.cache/searchcore"   # thay cho /var/cache (cần quyền root)
# ingest/src để core dùng chung ingest.storage.download_dir; thiếu nó thì
# encoder/base.py tự tải bằng boto3 (cùng quy ước marker .s3_download_done).
export PYTHONPATH="$PWD/services/core/src:$PWD/services/ingest/src"
python -m searchcore.main
```

Lần đầu core tải ~1.8GB encoder rồi tới snapshot, **không in log trong lúc tải** — theo
tiến độ bằng `du -sh .cache/searchcore`. Chỉ dùng `AWS_ACCESS_KEY`/`AWS_SECRET_KEY`
(không phải tên chuẩn `AWS_ACCESS_KEY_ID` của boto3).

**Terminal 2 — BE** (phải đợi core sẵn sàng *trước khi* chạy BE):

BE lấy tag vocab từ core **một lần duy nhất lúc startup**. Chạy BE trước core thì log ghi
`khởi động: 0 tag từ snapshot ?` kèm cảnh báo *"search sẽ không lọc tag"*, và lọc theo
topic trên UI sẽ im lặng không có tác dụng cho tới khi restart BE. Đúng thứ tự phải là
core sẵn sàng → BE → FE.

```bash
export SEARCHCORE_TARGET=localhost:50051     # ghi đè Unix socket trong .env
export PYTHONPATH="$PWD/services/be/src"
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

**Terminal 3 — FE:**

```bash
cd services/fe && npm install && npm run dev   # http://localhost:5173
```

### Cách B — docker compose

```bash
make dev     # hot reload
make down    # dừng
```

Compose dùng Unix socket giữa BE và core (`SEARCHCORE_TARGET` mặc định trong
`.env.example`), nên **không** đổi sang `localhost:50051` như cách A. Nếu chạy FE
trong compose thì đặt `VITE_BE_TARGET=http://be:8000`.

### Xác nhận chạy đúng

```bash
curl -s localhost:8000/healthz                     # BE sống
curl -s localhost:8000/readyz                      # core đã nạp snapshot chưa
curl -s -X POST localhost:8000/api/search \
  -H 'content-type: application/json' \
  -d '{"text":"cầu thủ bóng đá ăn mừng","top_k":5}' | head -c 400
```

`/readyz` phải trả `"ready": true`, **`"stub_mode": false`** và `point_count` khác 0
(corpus hiện tại 612.975). `stub_mode: true` nghĩa là core đang trả kết quả giả — đó là
lý do phổ biến nhất khiến search "chạy" mà kết quả vô nghĩa.

`/healthz` và `/readyz` khác nhau có chủ ý: BE sống ngay, nhưng core mất vài phút tải +
validate snapshot rồi warmup; trong khoảng đó search trả 503.

### Lỗi hay gặp

| Triệu chứng | Nguyên nhân |
|---|---|
| Kết quả trả về nhưng vô nghĩa | `SC_STUB_MODE=1` — core trả dữ liệu giả, chưa nạp snapshot. Kiểm bằng `stub_mode` trong `/readyz` |
| BE `UNAVAILABLE` / `DEADLINE_EXCEEDED` khi search | Core chưa xong warmup (2–3 phút), hoặc `SEARCHCORE_TARGET` còn trỏ Unix socket khi chạy cách A |
| `ModuleNotFoundError: app.pb` / `searchcore.pb` | Chưa chạy `make proto` |
| FE gọi `/api/*` ra 500/ECONNREFUSED | BE chưa chạy, hoặc `VITE_BE_TARGET` trỏ `http://be:8000` khi chạy ngoài compose |
| `/api/neighbors` trả 502 | Thiếu `AWS_ACCESS_KEY`/`AWS_SECRET_KEY` |
| Core chết vì không ghi được `MODEL_CACHE_DIR` | Mặc định `/var/cache/searchcore` cần root — trỏ vào thư mục trong repo |
| Lọc theo topic trên UI không có tác dụng | Hai nguyên nhân khác nhau: (a) BE khởi động trước core → nạp 0 tag, restart BE; (b) response có `warnings: ["tag_fallback"]` và `tags_used: []` → `FAISS_EF_SEARCH` quá thấp, xem bên dưới |
| Core im lặng nhiều phút, không log gì | Bình thường: đang tải từ S3. Theo `du -sh .cache/searchcore` |

### `tag_fallback` — lọc tag tắt trong im lặng

Nếu response có `warnings: ["tag_fallback"]`, `tags_used: []` và `candidate_count` bằng
cả corpus, tức là lọc tag **đã bị vô hiệu**: core tìm bằng tag ra ít hơn `top_k` nên tự
rơi về search toàn bộ ([search.py](services/core/src/searchcore/search.py#L154-L170) —
fallback này là cố ý, để LLM chọn sai tag không trả về một trang kết quả tự tin từ vùng
corpus không chứa đáp án).

Nguyên nhân thường gặp là `FAISS_EF_SEARCH` quá thấp. HNSW chỉ thăm khoảng `ef_search`
node rồi mới áp IDSelector, nên với tag chiếm 6–10% corpus thì `ef_search=64` chỉ ra
~4–6 kết quả — luôn nhỏ hơn `top_k` mà FE xin (300).

Đo trên corpus thật (613k point / 13 tag, `top_k=300`, `RERANK_CANDIDATES=1000`;
recall = overlap@300 lấy ef=10000 làm mốc):

| `ef_search` | core p50 (có tag) | recall search có tag | `tag_fallback` |
|---|---|---|---|
| 2000 | 40–71ms | 69–79% | 1/18 case |
| 4000 | 89–94ms | 82–84% | 1/18 case |
| **10000** | **195–219ms** | mốc | **0/120 query** |

Search **không** tag gần như không phụ thuộc `ef_search` (recall 99,6–100% ở mọi mức) —
chỉ search **có** tag mới tụt. Và phần chênh ~100ms là không đáng kể so với LLM enrich ở
BE (400–3700ms), nên đổi 100ms lấy 20% recall là đáng.

Lọc tag chỉ vô hiệu ở nhánh `pre`. Tag nhỏ hơn `EXACT_SUBSET_MAX` (20.000) đi đường
`exact_subset` nên vẫn đúng kể cả khi `ef_search` thấp — đó là lý do lỗi này trông
"lúc được lúc không" tuỳ tag.

## Thứ tự làm việc

| # | Việc | Ai |
|---|---|---|
| 1 | Chốt `proto/searchcore/v1/search.proto` | cả nhóm review |
| 2 | Schema Postgres + migration | be |
| 3 | Ingest chạy đúng **một** video end-to-end | ingest |
| 4 | `build_index.py` → snapshot đầy đủ cho 5 video | ingest |
| 5 | Core: load snapshot, 2-tier search, warmup, bench | core |
| 6 | BE `/api/search` đường đơn giản nhất, xác nhận p50 < 60ms | be |
| 7 | FE grid view + keyframe từ R2 | fe |
| 8 | Thêm dần: pre-filter → fusion → LLM → TRAKE → region | cả nhóm |

**Nguyên tắc**: mỗi bước phải chạy end-to-end với dữ liệu nhỏ trước khi mở rộng.
Đừng build cả pipeline rồi mới test — lỗi ở stage 2 sẽ lộ ra sau 20 giờ GPU đã cháy.

## Quy ước

- Branch: `feat/<service>-<mô-tả>`, ví dụ `feat/core-temporal-search`.
- Đổi `proto/` phải có review của cả người giữ BE lẫn core.
- Không commit `.faiss`, `.f16`, `.npy`, `.parquet` — đã chặn trong `.gitignore`.
- Quyết định kiến trúc ghi thành ADR trong `docs/decisions/`, không chôn trong chat.