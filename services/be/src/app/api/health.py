"""GET /healthz (BE sống) và /readyz (core đã nạp snapshot + encoder xong).

Hai cái KHÁC nhau và không được gộp: core mất vài phút để tải + validate snapshot rồi
warmup. Trong khoảng đó BE vẫn sống nhưng search sẽ trả 503. FE phải xem /readyz trước khi
mở ô tìm kiếm, nếu không người dùng gõ query và nhận về không gì cả mà không hiểu vì sao.
"""
from __future__ import annotations

from fastapi import APIRouter

from ..clients import searchcore

router = APIRouter()


@router.get("/healthz")
async def healthz() -> dict:
    return {"ok": True}


@router.get("/readyz")
async def readyz() -> dict:
    try:
        h = await searchcore.health(timeout=2.0)
    except Exception as ex:  # noqa: BLE001
        return {"ready": False, "reason": f"{type(ex).__name__}: {ex}"}
    return {
        "ready": h.ready,
        "snapshot_ver": h.snapshot_ver,
        "point_count": h.point_count,
        "encoder": h.encoder_name,
        "stub_mode": h.stub_mode,
        # stub_mode=True nghĩa là core trả kết quả giả — phải thấy được từ ngoài,
        # không thì rất dễ tưởng hệ thống đang chạy thật.
    }
