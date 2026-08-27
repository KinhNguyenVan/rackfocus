"""GET /healthz (BE sống), /readyz (core đã nạp xong), /ping (cho RunPod Serverless).

`/healthz` và `/readyz` KHÁC nhau và không được gộp: core mất vài phút để tải + validate
snapshot rồi warmup. Trong khoảng đó BE vẫn sống nhưng search sẽ trả 503. FE phải xem
/readyz trước khi mở ô tìm kiếm, nếu không người dùng gõ query và nhận về không gì cả mà
không hiểu vì sao.

`/ping` là cùng thông tin với `/readyz` nhưng nói bằng MÃ TRẠNG THÁI thay vì JSON, theo
hợp đồng của RunPod Serverless load-balancing: 200 = sẵn sàng nhận request, 204 = đang
khởi tạo (LB chờ, không gửi request tới), khác = worker hỏng (LB thay worker). Không có
nó thì LB thấy 200 ngay giây thứ 5 và đẩy request vào lúc core còn đang tải snapshot ->
người dùng nhận 503 hàng loạt.

Chỗ dễ sai nhất ở đây KHÔNG phải 200 mà là 204: trả 4xx/5xx trong lúc core đang tải làm
LB huỷ worker rồi dựng worker mới, và vòng tải 3.8GB bắt đầu lại từ đầu — vô tận.
"""
from __future__ import annotations

from fastapi import APIRouter, Response

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


@router.get("/ping")
async def ping() -> Response:
    """200 sẵn sàng / 204 đang khởi tạo / 503 core hỏng. Xem docstring đầu file."""
    try:
        h = await searchcore.health(timeout=2.0)
    except Exception:  # noqa: BLE001
        # Core chưa mở socket (đang tải snapshot) — CHƯA phải hỏng. Trả 204 để LB chờ
        # tiếp; trả lỗi ở đây là tự huỷ worker rồi khởi động lại vòng tải 3.8GB từ đầu.
        return Response(status_code=204)
    if not h.ready:
        return Response(status_code=204)
    # CỐ Ý không xét stub_mode ở đây, dù stub_mode=1 là MẶC ĐỊNH của Config và nghĩa là
    # core trả kết quả giả. Lý do: bật stub là quyết định có ý thức của người vận hành, và
    # đó là cách duy nhất để smoke-test đường dây RunPod (core lên trong vài giây thay vì
    # 200s). Chặn ở đây là làm mất luôn khả năng đó. Cảnh báo đã có ở hai chỗ dành cho
    # người đọc: log WARNING lúc core khởi động, và field `stub_mode` trong /readyz.
    return Response(status_code=200)
