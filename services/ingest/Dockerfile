FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.11 python3-pip ffmpeg libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake model weights vào image: pod GPU tính tiền theo giờ, đừng tải lại mỗi lần khởi động
# COPY models/ /models/

COPY src/ ./src/
ENV PYTHONPATH=/app/src PYTHONUNBUFFERED=1
CMD ["python3", "-m", "ingest.pipeline", "--resume"]
