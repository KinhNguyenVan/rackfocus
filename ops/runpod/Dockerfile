# Ảnh GỘP cho RunPod: core + BE + FE tĩnh + Caddy trong MỘT container.
#
# Vì sao không dùng docker-compose như ở local: RunPod Pod chạy đúng một image và
# KHÔNG có docker daemon bên trong, nên `docker compose up` không tồn tại ở đó. Ba
# service phải nằm cùng container, nói chuyện qua localhost + Unix socket.
#
# Build từ GỐC repo, không phải từ thư mục này:
#   docker build -f ops/runpod/Dockerfile -t <user>/rackfocus-pod:v1 .

# ── FE: build tĩnh ───────────────────────────────────────────────────
FROM node:20-alpine AS fe
WORKDIR /fe
COPY services/fe/package.json services/fe/package-lock.json ./
RUN npm ci
COPY services/fe/ ./
# Không cần VITE_BE_TARGET: FE gọi đường dẫn TƯƠNG ĐỐI (/api/search,
# /api/neighbors) và Caddy proxy sang BE. Nhờ vậy URL pod đổi mỗi lần tạo lại pod
# mà không phải build lại FE.
RUN npm run build

# ── Runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim

# libgomp1/libopenblas0: ONNX Runtime + faiss. vmtouch: kéo visual.f16 vào page
# cache (xem start.sh) — bắt buộc khi cache nằm trên network volume, vì mmap từ
# network storage chậm hơn NVMe nhiều lần.
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgomp1 libopenblas0 vmtouch curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

COPY --from=caddy:2-alpine /usr/bin/caddy /usr/bin/caddy

WORKDIR /app

# GPU=1 -> đổi onnxruntime (CPU) sang onnxruntime-gpu. Build:
#   docker build --build-arg GPU=1 -f ops/runpod/Dockerfile -t ...-pod:gpu .
#
# KHÔNG cần base image nvidia/cuda: extras [cuda,cudnn] kéo về các wheel nvidia-* nên
# CUDA runtime + cuDNN nằm trong site-packages, chỉ cần *driver* NVIDIA ở host (RunPod
# lo phần đó). Bù lại ảnh từ 1.4GB lên 2.4GB (đo thật).
#
# Phải uninstall onnxruntime trước: hai gói cung cấp CÙNG module `onnxruntime`, cài chồng
# thì import ra bản nào là chuyện của thứ tự file — không phải điều muốn đoán lúc thi.
ARG GPU=0

# GHIM 1.26.* CÓ CHỦ Ý, không dùng bản mới nhất:
#   * onnxruntime-gpu 1.26.x  -> nvidia-cuda-runtime-cu12 (CUDA 12), cần driver >= 570
#   * onnxruntime-gpu 1.29.x  -> nvidia-cuda-runtime ~13.0 (CUDA 13), cần driver >= 580
# Driver là của HOST, mình không cài được từ trong container. Pod RunPod dùng driver cũ
# hơn 580 là rất thường, và biểu hiện khi lệch là CUDA EP im lặng không nạp -> chạy CPU
# chậm 20x. Nâng lên CUDA 13 khi đã xác nhận `nvidia-smi` trên pod báo driver >= 580.
ARG ORT_GPU_SPEC="onnxruntime-gpu[cuda,cudnn]==1.26.*"

# Cài requirements TRƯỚC khi copy src để sửa code không phải cài lại ~2GB wheel.
COPY services/core/requirements.txt /tmp/core-req.txt
COPY services/be/requirements.txt   /tmp/be-req.txt
RUN pip install --no-cache-dir -r /tmp/core-req.txt -r /tmp/be-req.txt \
 && if [ "$GPU" = "1" ]; then \
      pip uninstall -y onnxruntime \
   && pip install --no-cache-dir "$ORT_GPU_SPEC" \
   && python -c "import onnxruntime as o; \
print('EP có sẵn:', o.get_available_providers()); \
assert 'CUDAExecutionProvider' in o.get_available_providers(), 'thiếu CUDA EP'"; \
    fi

COPY services/core/src/ /app/core/src/
COPY services/be/src/   /app/be/src/
COPY --from=fe /fe/dist/ /srv/
COPY ops/runpod/Caddyfile  /etc/caddy/Caddyfile
COPY ops/runpod/start.sh   /usr/local/bin/start.sh
COPY ops/runpod/encbench.py   /app/ops/encbench.py
COPY ops/runpod/gpu_parity.py /app/ops/gpu_parity.py
RUN chmod +x /usr/local/bin/start.sh

# Hai package không đụng tên nhau (`searchcore` vs `app`) nên gộp PYTHONPATH được.
ENV PYTHONPATH=/app/core/src:/app/be/src \
    PYTHONUNBUFFERED=1 \
    SC_STUB_MODE=0 \
    SC_SOCKET_PATH=/var/run/searchcore/sc.sock \
    SEARCHCORE_TARGET=unix:///var/run/searchcore/sc.sock \
    MODEL_CACHE_DIR=/workspace/cache/searchcore
# CỐ Ý không nướng POD_PORT vào đây: start.sh phải chọn được cổng theo chế độ (Serverless
# bắt buộc nghe trên $PORT của RunPod), mà một ENV nướng sẵn thì mọi phép gán "chỉ khi
# chưa có" đều thành vô hiệu. Mặc định 8080 nằm ở Caddyfile: `{$POD_PORT:8080}`.

# Chạy bằng root: network volume /workspace do root sở hữu, và pod là single-tenant
# sau proxy của RunPod. Đây là chỗ khác với services/*/Dockerfile (chạy user 10001/10002).
EXPOSE 8080
CMD ["/usr/local/bin/start.sh"]
