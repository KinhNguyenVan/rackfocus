# RUNBOOK — bật / tắt server rackfocus trên RunPod

Đọc file này **mỗi lần mở và tắt server**. Nó chỉ chứa việc phải làm và số phải biết.

Cài đặt lần đầu (tạo volume, secrets, endpoint): [SERVERLESS.md](SERVERLESS.md) hoặc
[README.md](README.md) cho Pod. Không cần đọc lại chúng khi chỉ bật/tắt.

---

## ▶ BẬT

### Serverless (đang dùng)

1. Console → endpoint `rackfocus` → **Edit Endpoint**
2. **Active Workers: 0 → 1**
3. Kiểm 2 ô này còn đúng — chúng hay bị reset khi sửa endpoint:
   - **Idle Timeout ≥ 600s** (mặc định 5s: nghỉ 5 giây là worker chết, query sau chờ 170s)
   - **Max Workers 2**
4. Save. Worker bắt đầu boot.
5. Chờ theo bảng ở [§ Chờ bao lâu](#-chờ-bao-lâu), rồi chạy [§ Kiểm tra](#-kiểm-tra).

### Pod (nếu dùng bản Pod)

1. Console → Pods → **Deploy** từ template đã lưu (hoặc Start nếu pod chỉ Stop)
2. Kiểm: đúng datacenter của network volume, image `ghcr.io/kinhnguyenvan/rackfocus-pod:latest`
3. Lấy pod id mới → URL là `https://<POD_ID>-8080.proxy.runpod.net`
   (pod id **đổi** mỗi lần tạo lại; FE dùng đường dẫn tương đối nên không phải sửa gì)

---

## ⏱ Chờ bao lâu

| Giai đoạn | Lần đầu tiên | Các lần sau |
|---|---|---|
| RunPod build image (chỉ khi vừa tạo Release) | 3–6 phút | — |
| Tải 3.8GB từ S3 → volume | vài phút, **không log tiến độ** | 0 (đã có cache) |
| Nạp encoder + snapshot + warmup | ~170s | **~170s** |
| `vmtouch` ghim 1.4GB vào page cache | +80s | +80s |

**Tổng thực tế: ~170s mỗi lần bật** (lần đầu tiên 5–10 phút).

Trong lúc đó `/api/search` trả **503**, không trả kết quả sai. `/ping` trả **204**.

---

## ✅ Kiểm tra

```bash
# Serverless
EP=https://<endpoint id>.api.runpod.ai
AUTH=(-H "Authorization: Bearer <runpod api key>")

# Pod — chỉ khác hai dòng này, phần dưới giữ nguyên
# EP=https://<POD_ID>-8080.proxy.runpod.net
# AUTH=(-u team:<mật khẩu>)

curl -s -o /dev/null -w '%{http_code}\n' "${AUTH[@]}" $EP/ping
curl -s "${AUTH[@]}" $EP/readyz
curl -s -X POST "${AUTH[@]}" $EP/api/search \
  -H 'content-type: application/json' \
  -d '{"text":"cầu thủ bóng đá ăn mừng","top_k":3}'
```

Đúng thì thấy:

| Lệnh | Kết quả đúng |
|---|---|
| `/ping` | `200` (còn `204` = đang nạp, chờ tiếp) |
| `/readyz` | `ready:true`, `point_count:612975`, **`stub_mode:false`** |
| `/api/search` | 3 hit, `tags_used:[8]`, `strategy:"exact_subset"`, `enrichment.encoded_text:"football player celebrating"` |

**`stub_mode` phải là `false`.** `true` nghĩa là core trả **kết quả giả** mà vẫn báo
`ready` — thiếu `SC_STUB_MODE=0` trong env.

### Mốc log bình thường

```
[start] chế độ serverless — cổng 80, cache /runpod-volume/cache/searchcore
[start] basic_auth: TẮT (RunPod Serverless lo phần xác thực)
searchcore encoder (s3) ...: dim=1152, nạp 77s
searchcore.snapshot snapshot v1: 612975 point, dim=1152, 13 tag, 0 chưa gán
text encoder ...: dim=1152, tokenizer=transformers, EP=CPUExecutionProvider, max_concurrent=2
[start] vmtouch: đang ghim 1.4G vào page cache
[start] vmtouch: xong
```

Dòng `khởi động: 0 tag từ snapshot ?` của BE ở đầu log là **bình thường** — BE lên trước
core và tự nạp lại tag vocab ở request sau, không cần restart.

---

## ⏹ TẮT

### Serverless

1. Edit Endpoint → **Active Workers: 1 → 0** → Save
2. Xong. Compute về **$0**. Còn trả $1.4/tháng cho network volume 20GB.

### Pod

**Terminate**, đừng Stop:

| | Compute | Storage |
|---|---|---|
| Stop | $0 | volume disk **$0.20**/GB/mo — **gấp đôi** |
| **Terminate** | $0 | chỉ network volume $0.07/GB/mo ≈ $1.4/tháng |

Terminate làm mất pod id (URL đổi) nhưng không mất cache — nó nằm trên network volume.
Lưu **Save as Template** để lần sau không phải dán lại env.

### Kiểm tra đã tắt thật

Console → endpoint/pod không còn worker nào Running. Billing → không còn dòng compute
đang chạy. **Network volume vẫn còn** (đó là chủ ý).

---

## 🔧 Sự cố → làm gì

| Hiện tượng | Nguyên nhân | Làm gì |
|---|---|---|
| Request đầu tiên lỗi / "no worker available" | Active Workers = 0, cold start 170s > hạn 120s của LB | Active = 1, chờ 170s |
| Query thứ hai lại chờ 170s | Idle Timeout còn 5s | đặt ≥ 600s |
| Endpoint không bao giờ "healthy" | Caddy nghe sai cổng | **đừng đặt `POD_PORT`** trên serverless; xem log dòng `cổng N` |
| Worker cứ bị thay, tải lại 3.8GB vô tận | `/ping` trả 4xx/5xx lúc đang nạp | đừng đặt auth trước `/ping`; bản gốc trả 204 đúng chuẩn |
| Kết quả search vô nghĩa nhưng `ready:true` | `stub_mode:true` | thêm `SC_STUB_MODE=0` |
| Mỗi lần bật lại tải 3.8GB | `MODEL_CACHE_DIR` không nằm trên volume | serverless: `/runpod-volume/...`, pod: `/workspace/...`. Log có cảnh báo |
| Search chậm bất thường ngay sau khi lên | `visual.f16` chưa vào page cache | chờ log `vmtouch: xong` (477ms → 24ms) |
| Mở URL endpoint ra 401 | trình duyệt không gắn Bearer khi điều hướng | chạy FE ở máy (xem § FE) |
| Lọc topic không có tác dụng | xem `enrichment.tag_source` | `llm_empty` = LLM cố ý không chọn; `guard_low_confidence` = LLM không chắc và guard cũng không nhận ra |
| `warnings:["tag_fallback"]` | `FAISS_EF_SEARCH` quá thấp | đặt `4000` |
| `/api/neighbors` trả 502 | thiếu key S3 | kiểm `AWS_ACCESS_KEY`/`AWS_SECRET_KEY` |
| `524` (pod) | request > 100s của Cloudflare | worst case BE là 51s → có thứ khác treo |
| Sửa code mà worker chạy bản cũ | push không trigger rebuild | `gh release create <tag>` |

---

## 🖥 FE

Không gõ URL endpoint vào trình duyệt (401). Chạy ở máy mình:

```bash
cd services/fe
RUNPOD_ENDPOINT_ID=<endpoint id> RUNPOD_API_KEY=<key> npm run dev
# [vite] proxy /api -> https://<id>.api.runpod.ai (kèm Bearer token)
```

→ `http://localhost:5173`

Dùng key **Restricted** chỉ cho endpoint này. Đừng đặt tên biến là `VITE_RUNPOD_API_KEY`
— tiền tố `VITE_` làm key bị nhúng thẳng vào bundle JS.

Với **Pod** thì mở URL pod trực tiếp được (có basic_auth), không cần chạy FE ở máy.

---

## 📊 Số cần nhớ

| | |
|---|---|
| Corpus | 612.975 vector, 785 video, 13 tag |
| Cold start (cache đã có) | **~170s** |
| Hạn "no worker" của LB | 120s ← nhỏ hơn cold start, đó là lý do cần Active=1 |
| Idle timeout mặc định | **5s** (bẫy) |
| Worst case một request | 51s (llm 6 + encode 15 + search 30) |
| Chi phí khi tắt | **$1.4/tháng** (volume 20GB) |
| Image | 1.4GB (CPU) / 2.4GB (GPU) |
| Payload tối đa | 30MB (serverless) |

---

## 🔁 Khi sửa code

```bash
git push
gh release create sless-vN --notes "..."     # push KHÔNG tự rebuild
```

Rồi Edit Endpoint → chọn build mới → Save. Worker mới cold start lại ~170s.

Sửa **Environment Variables** cũng làm worker restart và **xoá mọi thứ ngoài mount point**
— cache trong `/runpod-volume` an toàn.

---

## 📋 Env để dán lại

Cần khi tạo endpoint mới. Nguồn: `.env` ở máy (file này nằm trong `.gitignore`).

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

MODEL_CACHE_DIR=/runpod-volume/cache/searchcore
SC_STUB_MODE=0
SC_WARMUP_QUERIES=5
OMP_NUM_THREADS=4
SC_MAX_CONCURRENT_ENCODES=2
FAISS_EF_SEARCH=4000
RERANK_CANDIDATES=800
EXACT_SUBSET_MAX=100000
```

Pod thì đổi `MODEL_CACHE_DIR=/workspace/cache/searchcore` và thêm `POD_USER` +
`POD_PASS_HASH` (sinh hash: `docker run --rm caddy:2-alpine caddy hash-password --plaintext '<pass>'`).
