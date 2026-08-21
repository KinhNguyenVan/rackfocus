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

## Domain enrichment bằng Cerebras hoặc Gemini

Đây là job CPU độc lập chạy sau pipeline video. Job đọc từng
`Keyscence_*/keyscence/*/scenes.json`, dùng transcript để phân đoạn các chủ đề liên
tục và materialize kết quả thành MongoDB read model. Cerebras và Gemini dùng chung
prompt, Pydantic schema và deterministic validator nên output nghiệp vụ giống nhau.
Job không phụ thuộc vào các stage GPU và không sao chép keyframe manifest.

```text
S3 scenes.json
    -> provider structured output (Cerebras | Gemini)
    -> deterministic validation (đủ scene, đúng thứ tự, không gap/overlap)
    -> immutable analysis + scene interval mappings
    -> atomic promotion domain_jobs.active
```

### Chạy job

Thiết lập MongoDB và API key của provider cần dùng trong `.env`:

```env
MONGO_URI=mongodb+srv://<user>:<password>@<cluster>.mongodb.net/rackfocus?retryWrites=true&w=majority

# Chọn một provider; CLI --provider có thể override
DOMAIN_PROVIDER=gemini
GEMINI_API_KEY=...
GEMINI_MODEL=gemini-3.7-flash
GEMINI_MAX_OUTPUT_TOKENS=256000
GEMINI_THINKING_LEVEL=medium

# Hoặc Cerebras
CEREBRAS_API_KEY=...
CEREBRAS_MODEL=gpt-oss-120b
CEREBRAS_MAX_COMPLETION_TOKENS=256000
CEREBRAS_REASONING_EFFORT=medium
```

Nếu URI không chứa database, job dùng `rackfocus`. `MONGO_DB` là override tùy chọn.
Với Atlas, database user phải có quyền ghi/tạo index và IP của worker phải nằm trong
Network Access allowlist.

```bash
# từ root repo: chỉ kiểm tra các source tìm được
PYTHONPATH=services/ingest/src uv run python -m ingest.domain \
  --prefix Keyscence_L21_a/ --dry-run

# smoke test một video bằng Gemini
PYTHONPATH=services/ingest/src uv run python -m ingest.domain \
  --provider gemini \
  --prefix Keyscence_L21_a/keyscence/L21_V002/scenes.json \
  --workers 1

# cùng input và contract output nhưng dùng Cerebras
PYTHONPATH=services/ingest/src uv run python -m ingest.domain \
  --provider cerebras \
  --prefix Keyscence_L21_a/keyscence/L21_V002/scenes.json \
  --workers 1

# chạy một group
PYTHONPATH=services/ingest/src uv run python -m ingest.domain \
  --prefix Keyscence_L21_a/ --workers 4
```

`--provider` override `DOMAIN_PROVIDER`; `--model` override model mặc định của
provider đã chọn. `--dry-run` chỉ gọi S3, không kết nối MongoDB và không gọi LLM.
Mặc định job idempotent theo content hash và inference fingerprint. `--force` tạo
một lần phân tích mới nhưng chỉ promote khi toàn bộ document và mapping đã được ghi
thành công.

Hai provider mặc định dùng mức suy luận `medium` và chỉ trả JSON theo schema;
Cerebras đặt `reasoning_format=hidden`, còn Gemini không yêu cầu thought summary.
Giới hạn 256000 của hai provider là trần bảo vệ, không phải số token bắt buộc phải
sinh. Cả Gemini và Cerebras đều tính thinking trong ngân sách generation; Gemini
3.6/3.7 hỗ trợ tối đa 65536 output tokens, còn Cerebras `gpt-oss-120b` hỗ trợ tối
đa 40960 completion tokens. Cả hai provider đều dừng job nếu output chạm giới hạn
để không lưu JSON bị cắt; validator ứng dụng tiếp tục kiểm tra coverage, taxonomy
và keyword sau khi SDK đã kiểm tra cấu trúc. Với model override không hỗ trợ
reasoning level, adapter bỏ cấu hình reasoning thay vì gửi tham số không hợp lệ.

### Domain, topic và keywords

Mỗi segment là một câu chuyện biên tập liên tục và chỉ nhận một `domain_id` cùng
một `topic_id`. `topic_id` là taxonomy canonical để filter/group xuyên video;
`sub_domain` là tên câu chuyện dễ đọc; `keywords` là 3–4 cụm từ tự do phục vụ tìm
kiếm. Hai câu chuyện liền nhau có cùng `topic_id` vẫn là hai segment riêng.

Group của dataset và taxonomy ngữ nghĩa là hai lớp độc lập. `source.group` giữ nhóm
S3 gốc; model không gán cả video vào một domain chỉ vì nó thuộc nhóm “tin tức”. Với
các nhóm dữ liệu hiện tại, mapping canonical là:

| Nhóm nguồn | Domain/topic mặc định |
|---|---|
| Tin tức | Domain/topic cụ thể của từng câu chuyện; `general_news` chỉ là fallback |
| Nấu ăn | `food_lifestyle / food_cuisine` |
| Múa lân | `culture_travel_heritage / traditional_performing_arts` |
| Đua xe đạp | `sports / cycling` |

Múa lân chỉ chuyển sang `sports / other_sports` khi transcript nhấn mạnh một giải
đấu có chấm điểm hoặc xếp hạng. Chi tiết “múa lân” vẫn được giữ trong `keywords`,
tránh tạo topic quá hẹp chỉ dành cho một tập dữ liệu.

| Domain | Topic hợp lệ |
|---|---|
| `politics_society` | `public_policy`, `public_administration`, `labor_social_welfare`, `diplomacy`, `community_social_issues` |
| `economy_finance` | `banking_interest_rates`, `real_estate`, `trade_exports`, `business_markets`, `consumer_prices` |
| `agriculture` | `crop_farming`, `livestock_aquaculture`, `agricultural_exports`, `rural_development` |
| `culture_travel_heritage` | `heritage_historical_sites`, `tourism_destinations`, `arts_entertainment`, `festivals_traditions`, `traditional_performing_arts` |
| `science_technology` | `artificial_intelligence`, `robotics_automation`, `biotechnology`, `space_research`, `digital_technology` |
| `health` | `disease_prevention`, `healthcare_services`, `pharmaceuticals_medical_devices`, `public_health`, `nutrition_wellness` |
| `transport_urban` | `road_accident`, `traffic_violation`, `public_transport`, `transport_infrastructure`, `traffic_congestion` |
| `environment_nature` | `extreme_weather`, `natural_disaster`, `conservation_wildlife`, `pollution_waste`, `climate_environment` |
| `sports` | `football`, `cycling`, `combat_sports`, `athletics`, `other_sports` |
| `food_lifestyle` | `food_cuisine`, `consumer_lifestyle`, `family_daily_life`, `fashion_beauty` |
| `law_security` | `crime_investigation`, `court_justice`, `law_enforcement`, `public_security` |
| `education` | `schools_education`, `exams_admissions`, `skills_training`, `education_policy` |
| `general_news` | `breaking_news`, `mixed_news_digest`, `human_interest` |

Mọi domain đều cho phép `other` như fallback cuối cùng. Validator từ chối
`topic_id` không thuộc domain. Các cờ có nghĩa rõ ràng:

- `is_multi_domain`: có nhiều domain khác nhau.
- `is_multi_topic`: có nhiều cặp `(domain_id, topic_id)` khác nhau.
- `is_multi_segment`: có nhiều câu chuyện/segment, kể cả khi cùng topic.

### MongoDB data model

Ba collection dùng chung hai khóa:

| Collection | Vai trò | Khóa chính |
|---|---|---|
| `domain_jobs` | Trạng thái hiện tại của một `scenes.json` và con trỏ tới kết quả active | S3 URI (`source_id`) |
| `domain_analyses` | Kết quả LLM bất biến, versioned theo input/provider/prompt/model | `analysis_id` |
| `scene_domain_map` | Read model một document/scene để lookup frame/topic/keyword | `<analysis_id>:scene:<scene_id>` |

`source_id` là định danh canonical của video trong pipeline này, ví dụ
`s3://aic-bucket-2026/Keyscence_L21_a/keyscence/L21_V002/scenes.json`.
`external_video_id` (`L21_V002`) chỉ phục vụ hiển thị/lọc và không thay thế
`source_id`.

#### `domain_jobs`

Một document cho mỗi source. `active` và `last_attempt` độc lập nên một lần rerun
lỗi không làm mất kết quả đang phục vụ.

```json
{
  "_id": "s3://aic-bucket-2026/Keyscence_L21_a/keyscence/L21_V002/scenes.json",
  "external_video_id": "L21_V002",
  "source": {
    "bucket": "aic-bucket-2026",
    "key": "Keyscence_L21_a/keyscence/L21_V002/scenes.json",
    "group": "Keyscence_L21_a",
    "etag": "...",
    "version_id": null,
    "last_modified": "<BSON datetime>"
  },
  "active": {
    "analysis_id": "<sha256>",
    "content_hash": "<sha256>",
    "inference_fingerprint": "<sha256>",
    "provider": "gemini",
    "model": "gemini-3.7-flash",
    "activated_at": "<BSON datetime>"
  },
  "last_attempt": {
    "status": "completed",
    "content_hash": "<sha256>",
    "inference_fingerprint": "<sha256>",
    "provider": "gemini",
    "model": "gemini-3.7-flash",
    "started_at": "<BSON datetime>",
    "finished_at": "<BSON datetime>",
    "error": null
  },
  "attempt_count": 1,
  "is_multi_domain": true,
  "is_multi_topic": true,
  "is_multi_segment": true,
  "primary_domain": "Thể thao",
  "primary_domain_id": "sports",
  "num_scenes": 58,
  "num_segments": 3,
  "created_at": "<BSON datetime>",
  "updated_at": "<BSON datetime>"
}
```

`last_attempt.status` có thể là `processing`, `completed` hoặc `failed`. Khi failed,
`last_attempt.error` chứa lỗi rút gọn nhưng `active.analysis_id` cũ vẫn giữ nguyên.

#### `domain_analyses`

Một document là toàn bộ kết quả versioned của một lần phân tích. Segment chỉ chứa
khoảng scene liên tục; cùng domain xuất hiện lại sau chủ đề khác vẫn là segment mới.

```json
{
  "_id": "<analysis_id>",
  "source_id": "s3://aic-bucket-2026/.../L21_V002/scenes.json",
  "external_video_id": "L21_V002",
  "source": {
    "bucket": "aic-bucket-2026",
    "key": "Keyscence_L21_a/keyscence/L21_V002/scenes.json",
    "group": "Keyscence_L21_a",
    "etag": "...",
    "version_id": null,
    "last_modified": "<BSON datetime>"
  },
  "schema_version": 3,
  "taxonomy_version": "vi-video-v3",
  "prompt_version": "domain-segmentation-v5",
  "provider": "gemini",
  "model": "gemini-3.7-flash",
  "status": "completed",
  "content_hash": "<sha256>",
  "inference_fingerprint": "<sha256>",
  "is_multi_domain": true,
  "is_multi_topic": true,
  "is_multi_segment": true,
  "primary_domain": "Thể thao",
  "primary_domain_id": "sports",
  "segments": [
    {
      "segment_idx": 0,
      "start_scene_id": 0,
      "end_scene_id": 4,
      "domain": "Thể thao",
      "domain_id": "sports",
      "topic_id": "cycling",
      "sub_domain": "Đua xe đạp",
      "keywords": ["đua xe đạp", "vận động viên", "giải đấu"],
      "summary": "Các vận động viên thi đấu trong chặng đua xe đạp."
    }
  ],
  "inference": {
    "request_id": "...",
    "system_fingerprint": null,
    "model_version": "gemini-3.7-flash-...",
    "usage": {"prompt_token_count": 1000, "candidates_token_count": 200},
    "semantic_attempts": 1
  },
  "num_scenes": 58,
  "created_at": "<BSON datetime>",
  "updated_at": "<BSON datetime>"
}
```

Fingerprint bao gồm provider, model, prompt thực tế, taxonomy và JSON Schema. Đổi
Cerebras sang Gemini hoặc thay đổi bất kỳ thành phần nào sẽ tạo analysis mới thay
vì reuse kết quả không còn tương thích. Với Cerebras, `system_fingerprint` có giá
trị và `model_version` thường là `null`; Gemini thì ngược lại.

#### `scene_domain_map`

Mỗi scene được materialize thành một document interval. Domain/topic/keywords được
lặp có chủ đích để lookup không phải tải và duyệt mảng `segments` của analysis.

```json
{
  "_id": "<analysis_id>:scene:2",
  "analysis_id": "<analysis_id>",
  "source_id": "s3://aic-bucket-2026/.../L21_V002/scenes.json",
  "external_video_id": "L21_V002",
  "scene_id": 2,
  "segment_idx": 0,
  "domain": "Thể thao",
  "domain_id": "sports",
  "topic_id": "cycling",
  "sub_domain": "Đua xe đạp",
  "keywords": ["đua xe đạp", "vận động viên", "giải đấu"],
  "start_frame": 840,
  "end_frame": 1260,
  "start_time": 33.6,
  "end_time": 50.4,
  "scene_url": "s3://.../scene_002.mp4",
  "updated_at": "<BSON datetime>"
}
```

### Mapping và lookup

Frame lookup luôn đi qua active analysis:

```text
source_id + frame_idx
    -> domain_jobs._id
    -> domain_jobs.active.analysis_id
    -> scene_domain_map có start_frame <= frame_idx <= end_frame
    -> scene_id, segment_idx, domain_id, topic_id, sub_domain, keywords
```

```python
scene = repository.find_scene_by_frame(source_id, frame_idx=1000)
```

Topic và keyword lookup bắt đầu từ index tương ứng, sau đó join `domain_jobs` và chỉ
giữ document có `scene_domain_map.analysis_id == domain_jobs.active.analysis_id`.
Vì vậy các analysis cũ vẫn được lưu để audit nhưng không xuất hiện trong kết quả.

```python
cycling = repository.find_scenes_by_topic("cycling", limit=50)
bmx = repository.find_scenes_by_keyword("xe đạp BMX", limit=50)
```

Keyframe không được lưu lại trong MongoDB. Khi consumer cần ảnh keyframe, nó dùng
`source_id`/`external_video_id` để đọc manifest sẵn có trên S3; domain mapping chỉ
chịu trách nhiệm ánh xạ interval frame/time sang scene, topic và keywords.

## Lưu ý

Format timestamp của chunkformer đang **giả định** `"HH:MM:SS:ms"` trong
`asr._to_seconds`. Chạy thử 1 video, mở `scene_*.json` xem `script` có khớp thời gian
scene không; nếu lệch, chỉnh lại parser cho đúng format thực tế.
