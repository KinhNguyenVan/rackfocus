# Deploy rackfocus lên RunPod Pod

Dành cho kiểu dùng "bật khi dev/thi, tắt khi không dùng": **network volume giữ cache
4GB vĩnh viễn (~$1.4/tháng), pod thì terminate hẳn để compute về $0.**

> Muốn dùng **Serverless** (URL cố định, bung nhiều worker) thay vì Pod:
> xem [SERVERLESS.md](SERVERLESS.md). Cùng một image, `start.sh` tự nhận chế độ.
>
> Đây là hướng dẫn **cài đặt lần đầu**. Việc bật/tắt hằng ngày ở
> **[RUNBOOK.md](RUNBOOK.md)**.

## 0. Vì sao là 1 container chứ không phải docker-compose

RunPod Pod chạy **đúng một image** và **không có docker daemon** bên trong, nên
`docker compose up` không tồn tại ở đó. [Dockerfile](Dockerfile) gộp cả 4 thứ:

```
Cloudflare (RunPod proxy, cắt TLS, giới hạn 100s/request)
        │  https://<POD_ID>-8080.proxy.runpod.net
        ▼
   Caddy :8080                      ← trong container
   ├─ /api/*, /healthz, /readyz ──► BE :8000 ──► core qua Unix socket
   └─ /*                        ──► /srv (FE tĩnh, có SPA fallback)

   core  ──mmap──► /workspace/cache/searchcore/...  ← network volume
```

FE gọi đường dẫn **tương đối** (`/api/search`), nên URL pod đổi mỗi lần tạo lại pod mà
không phải build lại FE. Đây là lý do terminate rồi tạo pod mới không tốn công gì.

---

## 1. Làm một lần

### 1.1 Build ảnh (trên GitHub, KHÔNG build trên Mac)

RunPod là **x86_64**. Mac M-series build ra arm64 → pod báo `exec format error`. Build
amd64 trên Mac qua QEMU mất 15–25 phút; runner GitHub là amd64 native nên nhanh và miễn phí.

```bash
git push                                    # đẩy nhánh lên trước
gh workflow run runpod-image                # hoặc: git tag pod-v1 && git push origin pod-v1
gh run watch                                # xem tiến độ
```

Xong sẽ có `ghcr.io/kinhnguyenvan/rackfocus-pod:latest`. **Đặt package thành Public**
(GitHub → Packages → package settings → Change visibility), nếu không phải khai
*Container Registry Credentials* trong RunPod.

Muốn thử tại chỗ trước khi đẩy (arm64, chỉ để test logic — không dùng để deploy):

```bash
docker build -f ops/runpod/Dockerfile -t rackfocus-pod:test .
docker run -d --name rfpod --env-file .env -p 18080:8080 --memory 10g \
  -e SC_STUB_MODE=0 -e MODEL_CACHE_DIR=/workspace/cache/searchcore \
  -e SEARCHCORE_TARGET=unix:///var/run/searchcore/sc.sock \
  -v "$PWD/.cache/searchcore:/workspace/cache/searchcore" \
  rackfocus-pod:test
```

### 1.2 Network volume

RunPod → **Storage** → *New Network Volume*.

| Trường | Giá trị | Lý do |
|---|---|---|
| Size | **20 GB** | cache thật đang 3.8GB (snapshot 1.9 + encoder 1.8); 20GB để đủ chỗ cho 2 bản snapshot lúc swap |
| Datacenter | chọn DC **gần Singapore** và **có CPU pod** | bucket S3 ở `ap-southeast-1`; lần tải đầu là 3.7GB |

> Network volume **khoá theo datacenter**. Pod phải tạo trong đúng DC đó, nên ghi lại DC
> đã chọn. Giá: **$0.07/GB/tháng** dưới 1TB → 20GB ≈ **$1.4/tháng**, trả cả khi pod đã
> terminate. Đây chính là phần bạn cố tình trả để không phải tải lại 3.7GB mỗi lần.

### 1.3 Secrets

RunPod → **Settings** → *Secrets*. Tạo 3 cái, rồi tham chiếu trong env bằng
`{{ RUNPOD_SECRET_<tên> }}`:

| Secret | Lấy từ `.env` |
|---|---|
| `aws_access_key` | `AWS_ACCESS_KEY` |
| `aws_secret_key` | `AWS_SECRET_KEY` |
| `cerebras_key` | `CEREBRAS_API_KEY` |

Đừng dán key thẳng vào Environment Variables: giá trị env hiện nguyên văn trong UI và
trong template dùng lại.

### 1.4 Hash mật khẩu cho basic auth

URL proxy của RunPod là **công khai** — ai có pod id là gọi được `/api/search`, tức đốt
quota Cerebras của bạn. Sinh hash:

```bash
docker run --rm caddy:2-alpine caddy hash-password --plaintext 'mật-khẩu-của-bạn'
```

---

## 2. Tạo pod

RunPod → **Deploy** → tab **CPU** (xem §6 về việc vì sao chưa cần GPU).

| Trường | Giá trị |
|---|---|
| Instance | compute-optimized, **≥8 vCPU / 16 GB RAM** (16 vCPU / 32 GB thì thoải mái) |
| Datacenter | **đúng DC của network volume** |
| Network Volume | volume vừa tạo → mount tại `/workspace` |
| Container Image | `ghcr.io/kinhnguyenvan/rackfocus-pod:latest` |
| Container Disk | 15 GB (chỉ chứa ảnh 1.4GB + chỗ tạm) |
| Expose HTTP Ports | **8080** |
| Expose TCP Ports | `22` nếu muốn SSH vào xem log |
| Container Start Command | *để trống* — Dockerfile đã có `CMD` |

RAM tối thiểu 16GB vì: `visual.f16` 1.4GB phải nằm trong page cache, index HNSW+SQ8
~1GB, session ONNX fp32 ~2GB, cộng BE. Bản compose ở local đặt hạn mức core 12G.

### Environment Variables

Copy nguyên khối này (đã bỏ mọi biến Postgres/Redis vì stack online không dùng):

```
AWS_ACCESS_KEY={{ RUNPOD_SECRET_aws_access_key }}
AWS_SECRET_KEY={{ RUNPOD_SECRET_aws_secret_key }}
AWS_REGION=ap-southeast-1
AWS_BUCKET_NAME=aic-bucket-2026
SNAPSHOT_S3=s3://aic-bucket-2026/snapshots/v1
ENCODER_S3=s3://aic-bucket-2026/models/siglip-so400m-patch14-384-text-onnx
ENCODER_NAME=siglip-so400m-patch14-384
VECTOR_DIM=1152

CEREBRAS_API_KEY={{ RUNPOD_SECRET_cerebras_key }}
CEREBRAS_BASE_URL=https://api.cerebras.ai
LLM_API_KEY={{ RUNPOD_SECRET_cerebras_key }}
LLM_MODEL=cerebras/gpt-oss-120b
LLM_MAX_TOKENS=2000
LLM_REASONING_EFFORT=low

MODEL_CACHE_DIR=/workspace/cache/searchcore
SC_STUB_MODE=0
SC_WARMUP_QUERIES=5
OMP_NUM_THREADS=4
SC_MAX_CONCURRENT_ENCODES=2
FAISS_EF_SEARCH=4000
RERANK_CANDIDATES=800
EXACT_SUBSET_MAX=100000

POD_USER=team
POD_PASS_HASH=<hash từ §1.4>
POD_PORT=8080
```

`MODEL_CACHE_DIR` **phải** nằm trong `/workspace`, nếu không đặt ngoài thì
[start.sh](start.sh) sẽ cảnh báo và bạn mất 3.7GB tải lại mỗi lần tạo pod.

> ⚠️ Sửa env sau khi pod đã chạy sẽ **restart pod và xoá mọi thứ ngoài mount point**.
> Cache trong `/workspace` an toàn; những thứ bạn tự tạo ở `/root` hay `/tmp` thì mất.

---

## 3. Kiểm tra sau khi pod lên

Lấy URL: `https://<POD_ID>-8080.proxy.runpod.net` (pod id ở đầu trang pod).

```bash
POD=https://<POD_ID>-8080.proxy.runpod.net
AUTH='-u team:mật-khẩu-của-bạn'

curl -s -o /dev/null -w '%{http_code}\n' $POD/readyz          # 401 = basic auth đang chạy
curl -s $AUTH $POD/readyz                                      # đợi "ready":true
curl -s $AUTH $POD/api/tags | head -c 200                      # phải thấy count:13
curl -s $AUTH -X POST $POD/api/search -H 'content-type: application/json' \
  -d '{"text":"cầu thủ bóng đá ăn mừng","top_k":3}'
```

Mốc thời gian đo thật (cache đã có sẵn trên volume):

| Mốc | Thời gian |
|---|---|
| Caddy phục vụ FE, BE nhận request (search trả 503) | ~5s |
| encoder nạp xong | ~80s |
| snapshot nạp xong, `/readyz` trả `ready:true` | **~200s** |
| `vmtouch` ghim xong 1.4GB → search đạt tốc độ thật | +80s |

Lần đầu tiên (volume trống) cộng thêm thời gian tải 3.7GB từ S3 — im lặng không log
tiến độ, theo bằng `du -sh /workspace/cache/searchcore` qua SSH.

Trong lúc chờ, mở URL vẫn thấy FE, search trả 503 chứ **không** trả kết quả sai — BE tự
nạp lại tag vocab ở mỗi request nên không cần restart theo thứ tự.

---

## 4. Chu trình dev / thi → tắt

| Việc | Cách | Compute | Storage |
|---|---|---|---|
| Đang dùng | pod Running | tính theo phút | volume disk $0.10/GB/mo |
| Nghỉ vài giờ, muốn giữ URL | **Stop** | $0 | volume disk **$0.20**/GB/mo (gấp đôi!) + network volume $0.07 |
| Nghỉ dài (giữa các đợt dev, sau khi thi) | **Terminate** | $0 | chỉ network volume $0.07/GB/mo ≈ $1.4/tháng |

**Terminate là lựa chọn đúng cho bạn.** Mất pod id → URL đổi, nhưng FE dùng đường dẫn
tương đối nên không phải build lại gì; tạo pod mới với cùng image + cùng env + cùng
network volume là lên lại sau ~200s.

Để không phải dán lại 25 biến env mỗi lần: sau khi pod chạy ổn, vào
**hamburger menu → Save as Template**. Lần sau deploy chọn template đó.

Bật/tắt bằng CLI cho nhanh:

```bash
pip install runpod
runpod config                                  # dán API key
runpod pod list
runpod pod stop <pod_id>
runpod pod terminate <pod_id>
```

---

## 5. Chi phí thực tế

Cố định: **~$1.4/tháng** (network volume 20GB) — trả liên tục, kể cả khi không có pod nào.

Biến động: `giá pod/giờ × số giờ chạy`. Đọc giá trong console (giá CPU pod thay đổi theo
DC và tier). Tham chiếu đã kiểm: **pod GPU L4 là $0.49/giờ** ở Secure Cloud — CPU pod
rẻ hơn đáng kể.

Ví dụ dev 4 giờ/ngày × 20 ngày + thi 8 giờ = 88 giờ. Với pod $0.25/giờ → **$22 + $1.4**.

Cách tiết kiệm lớn nhất là **terminate ngay khi rời máy**, vì stop vẫn trả volume disk ở
mức gấp đôi.

---

## 6. CPU hay GPU

Encoder tự chọn: `SC_ENCODER_PROVIDERS=auto` (mặc định) → có GPU thì dùng CUDA/ROCm,
không có thì CPU. **File `model.onnx` không đổi** (fp32 chạy được cả hai) nên bật GPU
không phải export lại bundle.

| `SC_ENCODER_PROVIDERS` | Hành vi |
|---|---|
| *(không đặt)* / `auto` | có GPU dùng GPU, không có thì CPU, im lặng |
| `cpu` | ép CPU dù có GPU |
| `cuda` / `rocm` / `coreml` | ép EP đó; **không có thì cảnh báo to rồi chạy CPU** (không raise — lúc thi thà chậm còn hơn pod không lên) |

Ảnh CPU (mặc định) cài `onnxruntime` nên `auto` **luôn** ra CPU — GPU chỉ có tác dụng khi
build bản GPU:

```bash
gh workflow run runpod-image -f gpu=true     # -> ...-pod:<sha>-gpu và ...-pod:gpu
```

Bản GPU đổi `onnxruntime` → `onnxruntime-gpu[cuda,cudnn]`, **ghim 1.26.*** vì:

| onnxruntime-gpu | CUDA | Driver host cần |
|---|---|---|
| 1.26.x ← đang ghim | 12 | ≥ 570 |
| 1.29.x (mới nhất) | 13 | ≥ 580 |

Driver là của **host**, container không cài được. Pod dùng driver < 580 mà ảnh đòi CUDA 13
thì CUDA EP im lặng không nạp → chạy CPU chậm 20x. Xem `nvidia-smi` trên pod trước khi
nâng: `--build-arg ORT_GPU_SPEC="onnxruntime-gpu[cuda,cudnn]"`.

Ảnh GPU nặng ~4–5GB (wheel CUDA/cuDNN) so với 1.4GB bản CPU → container disk ≥ 25GB.

### Kiểm tra sau khi bật GPU

```bash
# 1. EP thật đang dùng — tab Logs của pod trong console RunPod, tìm "text encoder"
#   ... EP=CUDAExecutionProvider ...      <- đúng
#   ... EP=CPUExecutionProvider ...       <- GPU không nạp được, xem dòng WARNING ngay trên
nvidia-smi                                # xác nhận pod thật có GPU + phiên bản driver

# 2. Vector GPU có khớp CPU không (đây là rủi ro thật, không phải formality)
python /app/ops/gpu_parity.py

# 3. Nhanh hơn bao nhiêu
for t in 1 2 4; do OMP_NUM_THREADS=$t python /app/ops/encbench.py $t 20; done
```

`gpu_parity.py` so cosine giữa vector CPU và GPU của cùng một câu. Cần vì TF32 (bật mặc
định trên Ampere+) dùng mantissa 10 bit thay vì 23; qua 27 layer sai số đủ để đảo thứ tự
các hit gần bằng điểm nhau. Code **tắt TF32 mặc định** để GPU và CPU cho cùng kết quả;
`SC_CUDA_TF32=1` bật lại nếu đo thấy chấp nhận được.

Nhớ nâng `SC_MAX_CONCURRENT_ENCODES` lên ~8 khi chạy GPU — mặc định 2 là con số tính cho
CPU (8 gRPC worker × 4 BLAS thread trên 4 vCPU), trên GPU nó chỉ làm card nằm chờ. Core
tự log nhắc việc này.

### Khi nào GPU đáng tiền

Số đo trên máy local (M2, corpus 612.975 vector), chi phí CPU mỗi query:

| Tầng | CPU-sec/query |
|---|---|
| encode (1 thread) | 0.94 |
| search top_k=100 không tag | 0.108 |
| search top_k=100 có tag (exact_subset) | 0.013 |
| BE | 0.006 |

encode chiếm ~93% và là 53 GFLOP/query. Batch **không** giúp trên CPU (batch 1/4/8 đều
~0.95 CPU-sec/query) vì đã compute-bound, nên GPU là đòn bẩy duy nhất có ý nghĩa về
throughput.

| Tình huống | Nên dùng |
|---|---|
| Dev một mình, 1 đội dùng đồng thời | **CPU pod 8–16 vCPU.** Latency encode ~520ms ở 2–4 thread, chấp nhận được, và rẻ hơn hẳn |
| Cần chục request/giây trở lên | **GPU pod.** 100 rps = 5.3 TFLOPS sustained; T4 (65 TFLOPS fp16) chỉ cần ~8% peak |
| Toàn CPU cho 100 rps | ~101 core @100% util → ~160 vCPU. Đắt gấp ~6 lần GPU, không nên |

Đo lại trên pod:

```bash
for t in 1 2 4 8; do OMP_NUM_THREADS=$t python /app/ops/encbench.py $t 20; done
```

Lưu ý số thread: trên M2, 2 thread và 4 thread cho latency **bằng nhau** (517 vs 520ms)
nhưng 4 thread tốn 70% CPU nhiều hơn, và 8 thread còn tệ hơn cả 1 thread (1006ms). Trên
x86 nhiều core thì khác — đo rồi mới chỉnh `OMP_NUM_THREADS`.

---

## 7. Bẫy đã gặp

| Hiện tượng | Nguyên nhân |
|---|---|
| `exec format error` khi pod start | ảnh build trên Mac (arm64). Build qua workflow `runpod-image` |
| Pod không tạo được ở DC đã chọn | network volume khoá theo DC, mà DC đó hết CPU pod. Phải tạo volume mới ở DC khác + tải lại 3.7GB |
| Tải lại 3.7GB mỗi lần tạo pod | `MODEL_CACHE_DIR` không nằm trong `/workspace`. `start.sh` có cảnh báo trong log |
| `524` từ Cloudflare | request quá 100s. Worst case của BE là 51s (llm 6 + encode 15 + search 30) nên nếu gặp 524 là có thứ khác treo |
| Search chậm bất thường sau khi pod vừa lên | `visual.f16` chưa vào page cache. Chờ log `vmtouch: xong` (đo local: 477ms → 24ms) |
| Ai cũng gọi được API | quên `POD_USER`/`POD_PASS_HASH`. `start.sh` log rõ `basic_auth: TẮT` |
| Pull ảnh lỗi 403 | GHCR package còn Private mà chưa khai Container Registry Credentials |
| `/api/neighbors` trả 502 | thiếu `AWS_ACCESS_KEY`/`AWS_SECRET_KEY` |
| Thuê pod GPU mà log vẫn `EP=CPUExecutionProvider` | dùng ảnh CPU (`:latest`) thay vì `:gpu`; hoặc driver host cũ hơn CUDA mà ảnh đòi. Tìm dòng `WARNING` ngay trên nó |
| Bật GPU xong kết quả search hơi khác | TF32. Chạy `python /app/ops/gpu_parity.py` để xem lệch bao nhiêu; code đã tắt TF32 mặc định nên nếu vẫn lệch là có nguyên nhân khác |
