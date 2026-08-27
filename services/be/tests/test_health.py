"""GET /ping — hợp đồng health check của RunPod Serverless load-balancing.

Vì sao đáng test riêng: RunPod đọc MÃ TRẠNG THÁI, không đọc body. Trả sai mã là hai kiểu
hỏng nặng mà không có log nào của mình báo:

  * 200 quá sớm (core còn đang tải 3.8GB) -> LB đẩy request vào, người dùng nhận 503 hàng
    loạt trong ~3 phút.
  * 4xx/5xx trong lúc đang nạp -> LB coi worker là hỏng, thay worker, và vòng tải 3.8GB
    bắt đầu lại từ đầu. Vô tận.

Đúng phải là: 204 khi chưa xong, 200 khi xong, lỗi chỉ khi thật sự hỏng.
"""
from __future__ import annotations

import types

import grpc
import pytest

from app.clients import searchcore


def test_ping_200_khi_core_da_san_sang(client):
    r = client.get("/ping")
    assert r.status_code == 200
    assert r.content == b"", "RunPod chỉ đọc mã trạng thái; body chỉ làm tốn byte"


def test_healthz_khong_phu_thuoc_core(client):
    """/healthz là 'BE sống', /ping là 'sẵn sàng nhận request'. Không được gộp."""
    assert client.get("/healthz").json() == {"ok": True}


def _patch_health(monkeypatch, result):
    async def fake(*a, **kw):
        if isinstance(result, Exception):
            raise result
        return result
    monkeypatch.setattr(searchcore, "health", fake)


def test_ping_204_khi_core_chua_mo_socket(client, monkeypatch):
    """Core đang tải snapshot -> chưa mở Unix socket. CHƯA phải hỏng."""
    _patch_health(monkeypatch, grpc.aio.AioRpcError(
        grpc.StatusCode.UNAVAILABLE, None, None, details="socket not found"))
    r = client.get("/ping")
    assert r.status_code == 204


def test_ping_204_khi_core_song_nhung_chua_ready(client, monkeypatch):
    _patch_health(monkeypatch, types.SimpleNamespace(
        ready=False, snapshot_ver="", point_count=0, encoder_name="", stub_mode=False))
    assert client.get("/ping").status_code == 204


def test_ping_van_200_khi_stub_mode(client, monkeypatch):
    """stub_mode=1 là quyết định CÓ Ý THỨC của người vận hành và là cách duy nhất để
    smoke-test đường dây RunPod (core lên trong vài giây thay vì 200s). Chặn ở /ping là
    làm mất khả năng đó. Cảnh báo nằm ở log core + field stub_mode của /readyz."""
    _patch_health(monkeypatch, types.SimpleNamespace(
        ready=True, snapshot_ver="9", point_count=1, encoder_name="x", stub_mode=True))
    assert client.get("/ping").status_code == 200


@pytest.mark.parametrize("ready", [True, False])
def test_ping_khong_bao_gio_kem_body(client, monkeypatch, ready):
    """204 kèm body là vi phạm HTTP; một số proxy coi đó là lỗi."""
    _patch_health(monkeypatch, types.SimpleNamespace(
        ready=ready, snapshot_ver="9", point_count=1, encoder_name="x", stub_mode=False))
    r = client.get("/ping")
    assert r.status_code == (200 if ready else 204)
    assert r.content == b""
