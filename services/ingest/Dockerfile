FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# Ubuntu 22.04 mặc định python3 = 3.10, và `python3-pip` cũng trỏ vào 3.10 — cài
# python3.11 mà dùng pip hệ thống thì package rơi vào 3.10, không phải interpreter
# mình định dùng. Nên lấy pip riêng cho 3.11 qua ensurepip và trỏ python3 về nó,
# để khớp một phiên bản duy nhất với CI, core/be Dockerfile và pyproject.
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.11 python3.11-venv python3.11-dev \
      ffmpeg libgl1 libglib2.0-0 \
 && python3.11 -m ensurepip --upgrade \
 && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
# python3 -m pip, KHÔNG phải `pip` — sau ensurepip thì `pip` trên PATH có thể vẫn
# là của interpreter khác.
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# Bake model weights vào image: pod GPU tính tiền theo giờ, đừng tải lại mỗi lần khởi động
# COPY models/ /models/

COPY src/ ./src/
ENV PYTHONPATH=/app/src PYTHONUNBUFFERED=1
# Bỏ --resume: pipeline.main() không khai argparse arg đó nên sẽ lỗi ngay khi start.
CMD ["python3", "-m", "ingest.pipeline"]
