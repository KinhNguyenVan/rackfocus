# Ingest — pipeline offline (ASR + shot → scene)

Xử lý từng video thành **scene ngữ nghĩa** kèm **transcript** và **keyframe**, chạy offline
trên GPU thuê/Kaggle. Với mỗi video sinh ra 2 file JSON dùng cho bước build index và search.

## Đã làm

Các stage trong `src/ingest/stages/` (chuyển thể & dọn sạch từ bản nháp `bassl.ipynb`):

| Stage | File | Việc |
|---|---|---|
| Probe | `stages/probe.py` | ffprobe lấy `width, height, fps, n_frames` |
| Shot detect | `stages/shot_detect.py` | TransNetV2 → danh sách shot `(start_frame, end_frame)` |
| Keyframe | `stages/media.py` → `extract_keyframes` | 1 pass ffmpeg, ghi 3 keyframe/shot `{frame:06d}.webp` + `shots.csv` |
| Scene group | `stages/scene_group.py` | BaSSL CRN (ResNet50 + Transformer) gộp shot → scene |
| ASR | `stages/asr.py` | chunkformer (`khanhld/chunkformer-ctc-large-vie`) transcribe cả video + gán script cho từng scene |
| Cắt scene | `stages/media.py` → `cut_scenes` | ffmpeg cắt mỗi scene ra `scene_XXX.mp4` |
| Orchestrator | `pipeline.py` | `process_video()` ghi 2 JSON; `main()` CLI duyệt cả dataset, nạp model 1 lần |

Điểm thiết kế: keyframe chỉ **ghi 1 lần** theo frame index toàn cục; BaSSL đọc lại 3
keyframe/shot qua cột `kf0/kf1/kf2` trong `shots.csv` (không nhân đôi I/O).

## Output

Mỗi video `K01_V001.mp4` → thư mục phẳng `<out-root>/K01_V001/`:

```
out/K01_V001/
  scene_K01_V001.json      # danh sách scene
  keyframes.json           # danh sách keyframe
  shots.csv                # mapping shot → frame index (BaSSL dùng)
  keyframes/000012.webp …  # keyframe, tên = frame index {:06d}
  scenes/scene_000.mp4 …   # clip từng scene
```

`scene_<name>.json`:
```json
[{"scene_id": 0, "script": "…", "start_frame": 0, "end_frame": 128,
  "start_time": 0.0, "end_time": 5.12, "scene_url": "scenes/scene_000.mp4"}]
```

`keyframes.json` (đường dẫn relative so với thư mục video):
```json
[{"keyframe_url": "keyframes/000012.webp", "timestamp": 0.48}]
```

## Cấu trúc input mong đợi

~30 folder lớn `Videos_K{xx}/`, mỗi folder có subfolder `video/` chứa mp4 phẳng:

```
Videos_K01/video/K01_V001.mp4, K01_V002.mp4, …
Videos_K02/video/K02_V001.mp4, …
```

CLI glob đệ quy `**/video/*.mp4` (fallback: mọi `*.mp4`).

## Cần thêm trước khi chạy

- **BaSSL checkpoint `.ckpt`** — bắt buộc (không có sẽ báo lỗi, không chạy random weights).
- **GPU** + các package cài từ git (không có trên PyPI):
  - `pip install git+https://github.com/khanld/chunkformer.git`
  - TransNetV2: thêm thư mục `inference/` của repo
    [TransNetV2](https://github.com/soCzech/TransNetV2) vào `PYTHONPATH`.
- Còn lại: `pip install -r requirements.txt`.

## Chạy

```bash
# từ services/ingest/
export PYTHONPATH="src:/path/to/TransNetV2-master/inference"

# thử 1 video trước
python -m ingest.pipeline \
  --root /path/to/datasets \
  --out-root ./out \
  --bassl-ckpt /path/to/bassl.ckpt \
  --limit 1

# chạy toàn bộ
python -m ingest.pipeline --root /path/to/datasets --out-root ./out --bassl-ckpt /path/to/bassl.ckpt
```

Tuỳ chọn: `--shot-threshold 0.5` (hạ 0.3–0.4 nếu sót gradual cut), `--scene-threshold 0.55`,
`--overwrite` (ghi đè khi đã có output). Mặc định **idempotent** — video đã có
`scene_<name>.json` sẽ bỏ qua, một video lỗi không chặn cả batch.

## Kiểm thử

Logic thuần (gom scene + gán ASR theo thời gian), **không cần model/GPU**:

```bash
pytest tests/test_scene_group.py
```

## Lưu ý

Format timestamp của chunkformer đang **giả định** `"HH:MM:SS:ms"` trong
`asr._to_seconds`. Chạy thử 1 video, mở `scene_*.json` xem `script` có khớp thời gian
scene không; nếu lệch, chỉnh lại parser cho đúng format thực tế.
