FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY migrations/ ./migrations/
COPY alembic.ini .
ENV PYTHONPATH=/app/src PYTHONUNBUFFERED=1

RUN useradd -m -u 10002 be
USER be

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
