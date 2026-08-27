# Deploy rackfocus lên RunPod Serverless (Load balancer + GitHub)

Bản này cho **Serverless**. Bản Pod ở [README.md](README.md). Cùng một image, khác cách
khởi động — [start.sh](start.sh) tự nhận chế độ qua biến `RUNPOD_ENDPOINT_ID`.

> Đây là hướng dẫn **cài đặt lần đầu**. Việc bật/tắt hằng ngày ở
> **[RUNBOOK.md](RUNBOOK.md)** — đọc file đó, không phải file này.

## 0. Đọc trước khi làm: hai điều sẽ cắn bạn

**Pod THỰC SỰ có endpoint.** Tôi đã chạy thử: `https://<POD_ID>-8080.proxy.runpod.net`
phục vụ cả FE lẫn `/api/search`. Nếu bạn bỏ Pod vì tưởng nó không làm được endpoint thì
tiền đề đó sai. Nhưng Serverless có hai thứ Pod không có, và chúng là lý do hợp lý để
chọn nó: **URL cố định** (pod id đổi mỗi lần tạo lại pod) và **max workers > 1** để bung
khi nhiều người dùng cùng lúc.

**Cold start ~170–200s, mà LB chỉ chờ 120s.** Đo thật trên máy tôi với cache đã nằm sẵn
trên volume: `/ping` trả 200 sau **168s**. Hạn "no worker available" của load balancer là
**2 phút**. Nghĩa là:

> Với Active Workers = 0, request đầu tiên sau khi endpoint ngủ **sẽ lỗi**, và phải retry
> khoảng 3–4 phút mới có worker. Đây không phải cấu hình sai — đó là hệ quả của việc phải
> nạp 3.8GB snapshot + encoder vào RAM trước khi phục vụ được.

Nên: **Active Workers = 1 khi dev/thi, về 0 khi xong.** Đúng chu trình bạn muốn, chỉ là
cái công tắc nằm ở "Active Workers" thay vì "terminate pod".

| | Pod | Serverless LB |
|---|---|---|
| URL | đổi khi tạo lại pod | **cố định** `https://<ID>.api.runpod.ai` |
| Bung nhiều worker | không | **có** (max workers) |
| Tắt để tiết kiệm | terminate | Active Workers → 0 |
| Lần đầu gọi sau khi ngủ | — (pod luôn chạy) | **lỗi ~3–4 phút** |
| Mở FE bằng trình duyệt | được | **không** (xem §4) |
| Auth | basic_auth của mình | Bearer token của RunPod |

---

## 1. Trước khi vào console

```bash
git add -A && git commit -m "..." && git push
```

Build của RunPod đọc từ **GitHub**, không đọc máy bạn. Chưa push thì không có gì để build.

Và **rebuild không tự chạy khi push** — RunPod yêu cầu **tạo một GitHub Release** mới để
worker cập nhật:

```bash
gh release create sless-v1 --notes "deploy đầu tiên"
```

---

## 2. Điền form Deploy (đúng những trường trong ảnh bạn gửi)

| Trường | Điền | Vì sao |
|---|---|---|
| Repository | `KinhNguyenVan/rackfocus` | |
| Branch | **nhánh bạn vừa push** (đang là `main` trong form) | file `ops/runpod/` phải có trên nhánh này |
| **Dockerfile path** | **`/ops/runpod/Dockerfile`** | `/Dockerfile` không tồn tại — đó là lỗi đỏ trong ảnh |
| Advanced → Endpoint Type | **Load balancer** | Queue cần `runpod.serverless.start()`, mình không có và không cần |

Hai cảnh báo vàng trong ảnh sẽ hết:

* *"No Dockerfile found at specified path"* → hết khi sửa đường dẫn.
* *"Could not find runpod.serverless.start() in your repo"* → **bỏ qua**. Đó là yêu cầu
  của chế độ Queue. Load balancer chạy HTTP server thật (ở đây là Caddy + FastAPI), không
  cần handler của RunPod.

Build context là **gốc repo** (RunPod xác nhận trong tài liệu), nên Dockerfile nằm ở
`ops/runpod/` vẫn `COPY services/...` được. Context chỉ 1.3MB vì mọi thứ nặng đã nằm
trong `.gitignore` + `.dockerignore`.

Giới hạn build: image ≤ 80GB, bước `docker build` ≤ 30 phút. Image này ~1.4GB, build ~2–5
phút trên runner amd64.

---

## 3. Cấu hình endpoint

### Worker & scaling

| Thiết lập | Giá trị | Vì sao |
|---|---|---|
| **Active Workers** | **1** khi dev/thi, **0** khi nghỉ | 0 nghĩa là request đầu lỗi 3–4 phút. Đây là công tắc tiết kiệm của bạn |
| Max Workers | 2–3 | mỗi worker nạp RIÊNG 3.8GB → thêm worker = thêm 170s cold start. Đặt cao không giúp gì cho tải nhỏ |
| **Idle Timeout** | **≥ 600s** | mặc định **5s**: nghỉ 5 giây là worker chết, query sau chờ 170s. Bạn bị tính tiền trong lúc idle nhưng đó là giá của việc không phải chờ |
| Execution Timeout | 120s | worst case của BE là 51s (llm 6 + encode 15 + search 30) |
| FlashBoot | bật | RunPod giữ trạng thái worker sau khi tắt để "revive" nhanh hơn boot mới |
| GPU/CPU | **CPU** | encoder mặc định là CPU; xem [README §6](README.md) về khi nào GPU đáng tiền |

### Network volume

Gắn network volume (20GB) → serverless mount ở **`/runpod-volume`**, KHÁC pod (`/workspace`).
[start.sh](start.sh) tự chọn đúng đường dẫn theo chế độ, nhưng nếu bạn đặt
`MODEL_CACHE_DIR` tay thì phải đặt đúng — script sẽ cảnh báo nếu sai:

```
[start] CẢNH BÁO: MODEL_CACHE_DIR=/tmp/searchcore_cache KHÔNG nằm trong /runpod-volume
[start]           -> mất khi container biến mất, lần sau tải lại ~3.7GB
```

Volume khoá endpoint vào một datacenter. Tốc độ 200–400 MB/s nên 3.8GB mất ~10–20s I/O
thuần; phần còn lại của 170s là validate checksum + dựng index + warmup.

### Environment Variables

Giống bản Pod ([README §2](README.md)) nhưng **bỏ `POD_USER`/`POD_PASS_HASH`** (RunPod đã
bắt Bearer token; start.sh cũng tự bỏ qua chúng khi thấy chế độ serverless) và **đổi
`MODEL_CACHE_DIR`**:

```
MODEL_CACHE_DIR=/runpod-volume/cache/searchcore
```

**KHÔNG đặt `POD_PORT`.** Serverless bắt buộc nghe trên `$PORT` của RunPod, và start.sh
gán thẳng `POD_PORT="$PORT"` ở chế độ này. Đặt `POD_PORT` tay chỉ gây nhầm.

`HEALTH_CHECK_PATH` để mặc định `/ping` — BE có sẵn route đó.

---

## 4. FE không mở được bằng trình duyệt — và cách làm

Mọi request tới endpoint phải có `Authorization: Bearer <RUNPOD_API_KEY>`. Trình duyệt
**không gắn được header vào một lần điều hướng thường** (gõ URL vào thanh địa chỉ), nên
`https://<ID>.api.runpod.ai/` sẽ trả 401 chứ không ra FE. Đây là hạn chế của RunPod, không
phải của image.

Cách gọn nhất khi thi — **chạy FE ở máy mình, vite proxy chèn header**:

```bash
cd services/fe
RUNPOD_ENDPOINT_ID=<endpoint id> RUNPOD_API_KEY=<key> npm run dev
# [vite] proxy /api -> https://<id>.api.runpod.ai (kèm Bearer token)
```

Mở `http://localhost:5173`. Key nằm trên máy bạn, **không** vào bundle JS.
[vite.config.ts](../../services/fe/vite.config.ts) cố ý đọc `RUNPOD_API_KEY` chứ không
phải `VITE_RUNPOD_API_KEY`: biến có tiền tố `VITE_` bị nhúng thẳng vào bundle và ai mở
devtools cũng đọc được.

Tạo key **Restricted** (Settings → API Keys) chỉ cho đúng endpoint này, đừng dùng key
`All`: key rò ra thì kẻ khác chỉ đốt được compute của một endpoint, không xoá được pod hay
đọc được thứ khác trong tài khoản.

Nếu cần mở cho người ngoài không có key, đó là lúc **Pod hợp hơn Serverless** — pod phục
vụ FE qua trình duyệt được, có basic_auth của mình.

---

## 5. Kiểm tra

```bash
EP=https://<endpoint id>.api.runpod.ai
K=<runpod api key>

# 204 = đang nạp (bình thường, chờ tiếp), 200 = sẵn sàng
curl -s -o /dev/null -w '%{http_code}\n' -H "Authorization: Bearer $K" $EP/ping

curl -s -H "Authorization: Bearer $K" $EP/readyz          # ready + point_count + stub_mode
curl -s -H "Authorization: Bearer $K" $EP/api/tags | head -c 200

curl -s -X POST $EP/api/search -H "Authorization: Bearer $K" \
  -H 'content-type: application/json' \
  -d '{"text":"cầu thủ bóng đá ăn mừng","top_k":3}'
```

Đã đo trên container chạy đúng chế độ serverless ở máy local (`RUNPOD_ENDPOINT_ID` +
`PORT=8081` + volume ở `/runpod-volume`):

| Mốc | Kết quả |
|---|---|
| Caddy nghe đúng `$PORT` | 8081 ✓ |
| basic_auth tự tắt | `bỏ qua basic_auth: chế độ serverless...` ✓ |
| `/ping` lúc core đang nạp | **204** ✓ |
| `/api/search` lúc core đang nạp | 503 (không phải kết quả sai) ✓ |
| `/ping` khi xong | **200 sau 168s** |
| `/api/search` sau khi ready | 3 hit, `tags [8]`, `exact_subset`, enrich → `football player celebrating` ✓ |

---

## 6. Chi phí

| Khoản | Khi dev/thi (Active=1) | Khi nghỉ (Active=0) |
|---|---|---|
| Worker | tính liên tục theo giá worker/giờ | **$0** |
| Idle timeout | tính tiền trong lúc idle | — |
| Network volume 20GB | $0.07/GB/mo ≈ **$1.4/tháng** | **$1.4/tháng** (vẫn trả) |

So với Pod: Pod terminate cũng về $0 compute và cũng chỉ còn $1.4 volume. **Serverless
không rẻ hơn Pod cho tải nhỏ** — nó mua cho bạn URL cố định và khả năng bung worker. Chọn
theo nhu cầu đó, không phải theo giá.

---

## 7. Bẫy riêng của Serverless

| Hiện tượng | Nguyên nhân |
|---|---|
| Endpoint không bao giờ "healthy" | Caddy nghe cổng khác `$PORT`. Đừng đặt `POD_PORT`; kiểm dòng log `chế độ serverless — cổng N` |
| Worker cứ bị thay, tải lại 3.8GB vô tận | `/ping` trả 4xx/5xx trong lúc đang nạp. Bản này trả **204** đúng chuẩn — nếu bạn thêm auth trước `/ping` thì hỏng ngay ([Caddyfile](Caddyfile) cố ý đặt `/ping` ngoài block auth) |
| Request đầu luôn lỗi | Active Workers = 0 + cold start 170s > hạn 120s của LB. Đặt Active = 1 |
| Query thứ hai lại chờ 170s | Idle Timeout còn mặc định 5s |
| Đổi code mà worker vẫn chạy bản cũ | push KHÔNG trigger rebuild — phải `gh release create` |
| Mở URL endpoint ra 401 | trình duyệt không gắn được Bearer khi điều hướng. Xem §4 |
| Muốn bản GPU | GitHub integration **không hỗ trợ build args**, nên `--build-arg GPU=1` không dùng được ở đường này. Build qua workflow `runpod-image -f gpu=true` rồi deploy endpoint từ image `ghcr.io/...-pod:gpu` |
