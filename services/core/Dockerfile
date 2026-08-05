FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
      libgomp1 libopenblas0 vmtouch \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
ENV PYTHONPATH=/app/src PYTHONUNBUFFERED=1

# Không chạy bằng root
RUN useradd -m -u 10001 sc && mkdir -p /var/run/searchcore && chown -R sc /var/run/searchcore
USER sc

CMD ["python", "-m", "searchcore.main"]
