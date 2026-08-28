"""Tạo FastAPI app, mount router, lifespan: mở gRPC channel + nạp trước tag vocab."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from .api.router import api_router, root_router
from .clients import searchcore
from .config import get_settings
from .services import tagvocab, transcript

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    st = get_settings()
    await searchcore.connect(st.searchcore_target)

    # Nạp trước vocab để query đầu tiên không phải trả thêm một round-trip.
    # Core có thể còn đang nạp snapshot (vài phút) -> lỗi ở đây là bình thường,
    # tagvocab.get() sẽ thử lại ở request sau.
    try:
        vocab, ver = await tagvocab.get(st)
        log.info("khởi động: %d tag từ snapshot %s", len(vocab), ver or "?")
    except Exception as ex:  # noqa: BLE001
        log.warning("chưa nạp được tag vocab lúc khởi động (%s) — sẽ thử lại khi có "
                    "request. Core có thể đang tải snapshot.", type(ex).__name__)

    # Transcript keyword index (tuỳ chọn): rỗng = tắt; lỗi mở file KHÔNG chặn khởi động,
    # endpoint /api/transcript/suggest sẽ trả 503 cho tới khi cấu hình đúng.
    if st.transcript_db_path:
        try:
            transcript.load(st.transcript_db_path)
        except Exception as ex:  # noqa: BLE001
            log.warning("không mở được transcript index (%s: %s) — /api/transcript/suggest "
                        "sẽ trả 503.", type(ex).__name__, ex)
    else:
        log.info("transcript_db_path rỗng — bỏ qua transcript keyword search.")

    yield
    await searchcore.close()


app = FastAPI(title="rackfocus BE", lifespan=lifespan)
app.include_router(root_router)
app.include_router(api_router)
