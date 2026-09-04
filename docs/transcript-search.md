# Transcript keyword search (tìm theo lời thoại)

Tìm kiếm **keyword trên lời thoại (ASR) của từng scene** rồi gợi ý kiểu Google. Bổ trợ cho
vector/image search: vector hợp câu "trên màn hình có X", transcript hợp câu "có người **nói**
về X" (phát biểu, phỏng vấn, tường thuật, dạy nấu ăn…). Tính năng **cộng thêm, opt-in** —
không set `TRANSCRIPT_DB_PATH` thì endpoint trả 503, phần còn lại chạy như cũ.

## Kiến trúc

```
offline:  payload.parquet (snapshot)  +  Transcripts_*/…json (ASR)
                         │  ghép theo thời gian (midpoint segment ∈ [scene.start,end])
                         ▼
              ingest/build_transcript_index.py  ──►  transcript.sqlite (FTS5)
online:   FE dropdown ──GET /api/transcript/suggest──►  BE mở sqlite read-only (nạp ở lifespan)
```

- **Vì sao là artifact riêng**: text `scene.script` KHÔNG có trong snapshot (`payload.parquet`
  chỉ có cờ `has_speech`) và online KHÔNG có Postgres. Nên đóng gói lời thoại thành 1 SQLite
  FTS5 bất biến, BE mở lúc khởi động.
- **Khớp key**: `clip_key`/`keyframe_key` lấy thẳng từ `payload.parquet` (URL S3 tuyệt đối)
  nên click 1 gợi ý mở **đúng scene clip** mà vector search cũng trỏ tới.
- **Ghép ASR→scene**: sao đúng `stages/asr.py::assign_script_to_scenes` — script của scene =
  nối text mọi segment có **midpoint** `(start+end)/2` nằm trong `[start_sec, end_sec]` của
  scene → khớp cờ `has_speech` của snapshot.

## Nguồn dữ liệu trên S3 (`aic-bucket-2026`, ap-southeast-1)

| Prefix | Nội dung |
|---|---|
| `snapshots/vN/payload.parquet` | ranh giới scene + `clip_key`/`keyframe_key` (URL tuyệt đối) |
| `Transcripts_<batch>/transcripts/<video>.json` | ASR: `{"video_id": <tên video>, "segments":[{start,end,text}]}` |

Không có `Videos_*` (video đầy đủ) → phát bằng scene clip (`Keyscence_*`).

## Dựng index

Một lệnh, không cần Postgres (tự tải payload + mọi `Transcripts_*` từ S3, ghép, dựng sqlite).

macOS/Linux (bash/zsh) — chạy từ gốc repo:

```bash
export PYTHONPATH="$PWD/services/ingest/src"
export SNAPSHOT_S3="s3://aic-bucket-2026/snapshots/v2"   # trỏ bản snapshot muốn dùng
python -m ingest.build_transcript_index --from-s3 --db "$PWD/.tmp/transcript_full.sqlite"
```

Windows (PowerShell) — thay `C:/rackfocus` bằng đường dẫn repo của bạn:

```powershell
$env:PYTHONPATH="C:/rackfocus/services/ingest/src"
$env:SNAPSHOT_S3="s3://aic-bucket-2026/snapshots/v2"
python -m ingest.build_transcript_index --from-s3 --db C:/rackfocus/.tmp/transcript_full.sqlite
```

`--from-s3` tự nạp `.env` (AWS_*). Các cách khác:
- `--payload <parquet> --transcripts <dir>`: dựng từ file cục bộ (không tải S3).
- `--out-root <dir>`: [legacy] từ output ingest `<name>/scene_*.json` (cần Postgres cấp video_id).

Tokenizer FTS5 `unicode61 remove_diacritics 0` → **giữ dấu tiếng Việt** (`khi` ≠ `khí`).

### Builder fail-closed ở chỗ nào

Lệnh trên **ghi atomic**: dựng vào `<db>.tmp` rồi mới `os.replace` sang chỗ thật, nên mọi
lỗi dưới đây để **nguyên bản index cũ đang dùng**:

| Ca | Hành vi |
|---|---|
| Không thấy file nào dưới `Transcripts_*` (prefix/bucket/credential sai) | `RuntimeError`, dừng |
| Ghép ra 0 row (tên video payload ≠ tên file transcript) | `ValueError`, không ghi file |
| Transcript JSON hỏng / thiếu field `segments` | in `[BAD] <file>` + đếm riêng ở dòng tổng kết |
| Ctrl-C giữa lúc build | chỉ mất `<db>.tmp` |

Cụ thể là **không** còn ca "index 0 row nhưng đúng schema" — BE cũng từ chối mở file như vậy
(xem `transcript.py::TranscriptIndex.__init__`), vì nó làm endpoint trả `200 {"items": []}`
cho mọi keyword, y hệt "không ai nói câu đó".

⚠️ **`transcript.sqlite` nằm ngoài checksum của snapshot.** `manifest.json` chỉ ký các file do
`build_index.py` sinh, mà `build_index.py` hiện là stub và **không** gọi builder này. Nghĩa là
ghép DB dựng theo `snapshots/v1` với core đang chạy `snapshots/v2` sẽ **không bị phát hiện ở
tầng nào** — kết quả là click gợi ý mở sai clip. Người vận hành phải tự đảm bảo `SNAPSHOT_S3`
lúc dựng DB trùng bản snapshot core đang chạy.

## Chạy / test cục bộ

Set `TRANSCRIPT_DB_PATH` trong `.env` trỏ file sqlite vừa dựng, rồi:

1. **BE local** (phục vụ transcript; **không cần core** — endpoint này không đi qua gRPC):

   ```bash
   export PYTHONPATH="$PWD/services/be/src"
   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```

   Chờ log `transcript index mở từ … (N scene có thoại)`. Thấy
   `không mở được transcript index (…)` thay vì dòng đó nghĩa là file thiếu/hỏng/rỗng —
   endpoint sẽ trả 503 kèm lý do, và `/readyz` báo `"transcript_index": "failed"`.
2. **FE**: nếu search vector vẫn đi RunPod, set `VITE_TRANSCRIPT_TARGET=http://localhost:8000`
   để **chỉ** `/api/transcript` về BE local, phần còn lại giữ RunPod (opt-in ở
   `vite.config.ts`). Sau đó `npm run dev`.
3. UI: KIS mode → bật switch **"Tìm transcript"** → gõ ≥2 ký tự (đúng dấu).

Kiểm tra nhanh không cần FE: `http://localhost:8000/api/transcript/suggest?q=<keyword>`.

Trạng thái index xem ở `/readyz` → field `transcript_index`:

| Giá trị | Nghĩa |
|---|---|
| `loaded` | mở được, endpoint chạy |
| `disabled` | `TRANSCRIPT_DB_PATH` rỗng — tắt có ý thức, endpoint 503 |
| `failed` | có cấu hình nhưng file thiếu/hỏng/0 row — **sự cố**, endpoint 503 |

Cả `disabled` và `failed` đều cho 503; field này là cách duy nhất phân biệt "chưa bật" với
"bật mà hỏng".

## Giới hạn triển khai: hiện chỉ chạy LOCAL

Tính năng này **chưa chạy trên endpoint RunPod**. `ops/runpod/*` không được sửa trong PR
thêm tính năng: image không có bước tải `transcript.sqlite` từ S3, và `TRANSCRIPT_DB_PATH`
không được set trong `start.sh`. Nên trên RunPod `/api/transcript/suggest` luôn trả 503
(`transcript_index: disabled`) — đúng thiết kế opt-in, nhưng nghĩa là muốn dùng lúc thi thì
phải chạy BE local (bước 1 ở trên) và trỏ `VITE_TRANSCRIPT_TARGET` về đó.

Để chạy được trên RunPod cần thêm, sau này: tải `transcript.sqlite` vào `$MODEL_CACHE_DIR`
(hoặc network volume) trong `ops/runpod/start.sh`, rồi export `TRANSCRIPT_DB_PATH` trỏ vào đó.

## Reload khi thêm video mới trên S3

Transcript index ghép payload + transcript nên video mới phải có ở **cả hai**:

1. Video mới đã nằm trong 1 snapshot mới (`snapshots/vN+1/payload.parquet`).
2. ASR của nó đã ở `Transcripts_*/` trên S3.
3. Dựng lại sqlite bằng lệnh `--from-s3` ở trên (trỏ `SNAPSHOT_S3` sang bản mới).
4. **Restart BE** — index chỉ nạp một lần lúc khởi động.

⚠️ Để **vector search** cũng thấy video mới, RunPod phải được swap sang snapshot mới; hai
nhánh (vector/transcript) sẽ lệch tập video tới khi cả hai cùng bản.

## Giới hạn

- Chỉ tìm được nội dung **nói ra** — câu hỏi kiểu "trên màn hình có bảng/biểu đồ/đề thi Y"
  (visual/OCR) nên dùng vector/image search, không phải transcript.
- Prefix match nối AND có thể ra nhiễu (`đô*` khớp "đông"); dùng **phrase** `"dồi trường"`
  cho chính xác.

## File liên quan

- `services/ingest/src/ingest/build_transcript_index.py` — builder + CLI.
- `services/be/src/app/services/transcript.py` — mở sqlite, truy vấn FTS5 + bm25 + snippet.
- `services/be/src/app/api/transcript.py` — `GET /api/transcript/suggest`.
- `services/fe/src/components/TranscriptSuggest.tsx` — dropdown; toggle ở `pages.tsx` (KIS).
