"""Unit test services/segment.py — chỉ mock litellm, không cần core gRPC.

Khác test_search_temporal.py: ở đây gọi thẳng hàm với settings giả, nên không dùng
fixture `client` (vốn dựng cả một core gRPC thật qua Unix socket).

Gọi qua `asyncio.run` thay vì `@pytest.mark.asyncio`: repo không cài pytest-asyncio, mà
CI cài dependency bằng tay theo ma trận trong ci.yml nên thêm gói là phải sửa cả đó.
"""
import asyncio
import json
import sys
import types

import pytest

from app.services import segment as seg


@pytest.fixture(autouse=True)
def _clean_litellm():
    yield
    sys.modules.pop("litellm", None)


def fake_settings(**over):
    base = dict(llm_enabled=True, llm_model="fake/model", llm_api_key="",
                llm_temperature=0.0, llm_timeout_s=6.0, llm_max_tokens=2000,
                llm_reasoning_effort="")
    base.update(over)
    return types.SimpleNamespace(**base)


def mock_litellm(content=None, raises=None):
    """Đặt litellm giả trả đúng `content`. `raises` để giả lập lỗi mạng/timeout."""
    calls = []

    async def acompletion(**kwargs):
        calls.append(kwargs)
        if raises:
            raise raises
        msg = types.SimpleNamespace(content=content)
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=msg, finish_reason="stop")])

    sys.modules["litellm"] = types.SimpleNamespace(acompletion=acompletion)
    return calls


def test_tach_nhieu_doan():
    mock_litellm(json.dumps([
        {"order": 1, "english_clip_query": "A fish placed on a scale."},
        {"order": 2, "english_clip_query": "A person holding a fish."},
    ]))
    r = asyncio.run(seg.segment("con cá được cân, sau đó người cầm đuôi cá",
                                fake_settings()))
    assert r.ok
    assert [s.english_clip_query for s in r.segments] == [
        "A fish placed on a scale.", "A person holding a fish."]


def test_bo_qua_nhan_nguon_neu_llm_van_tra():
    """Prompt cấm mang nhãn nguồn sang output, nhưng LLM lỡ trả thì cũng không sao.

    Segment không có chỗ chứa `label` -> khoá thừa bị bỏ im lặng, không được raise.
    """
    mock_litellm(json.dumps([
        {"order": 1, "label": "E1", "english_clip_query": "Golden dragons spinning."},
    ]))
    r = asyncio.run(seg.segment("múa lân", fake_settings()))
    assert r.ok
    assert r.segments[0].english_clip_query == "Golden dragons spinning."


def test_order_danh_lai_theo_vi_tri():
    """LLM đánh trùng/nhảy số được; UI dùng order làm khoá nên phải đánh lại."""
    mock_litellm(json.dumps([
        {"order": 5, "english_clip_query": "Golden dragons spinning."},
        {"order": 5, "english_clip_query": "A mallet striking a gong."},
    ]))
    r = asyncio.run(seg.segment("múa lân", fake_settings()))
    assert [s.order for s in r.segments] == [1, 2]


def test_bo_doan_rong():
    mock_litellm(json.dumps([
        {"order": 1, "english_clip_query": "   "},
        {"order": 2, "english_clip_query": "A mallet striking a gong."},
    ]))
    r = asyncio.run(seg.segment("múa lân", fake_settings()))
    assert len(r.segments) == 1
    assert r.segments[0].order == 1


def test_json_rac_thi_lui_ve_cau_goc():
    mock_litellm("xin lỗi, tôi không hiểu")
    r = asyncio.run(seg.segment("câu gốc", fake_settings()))
    assert not r.ok
    assert [s.english_clip_query for s in r.segments] == ["câu gốc"]


def test_llm_loi_thi_lui_ve_cau_goc():
    mock_litellm(raises=TimeoutError("quá thời gian"))
    r = asyncio.run(seg.segment("câu gốc", fake_settings()))
    assert not r.ok
    assert [s.english_clip_query for s in r.segments] == ["câu gốc"]


def test_tat_llm_thi_khong_goi_mang():
    calls = mock_litellm(json.dumps([{"order": 1, "english_clip_query": "x"}]))
    r = asyncio.run(seg.segment("câu gốc", fake_settings(llm_enabled=False)))
    assert calls == []
    assert r.ok
    assert r.segments[0].english_clip_query == "câu gốc"
