"""Test build_prompt/enrich với vocab = TagInfo (tag hiện tại = domain_id).

Xem docs/search-design.md §4, §6. Vocab giờ có cả `name` (slug) và `description` (nhãn
tiếng Việt) thay vì chỉ description tự do — prompt phải dùng cả hai, và vẫn phải chạy
đúng với vocab "phẳng" cũ (name rỗng) vì snapshot.py chấp nhận cả hai dạng.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from app.clients.searchcore import TagInfo
from app.services.enrich import build_prompt, enrich

VOCAB = {
    8: TagInfo(id=8, name="sports", description="Thể thao", point_count=1200),
    11: TagInfo(id=11, name="education", description="Giáo dục", point_count=300),
}


def settings(**overrides):
    # Phải khai ĐỦ field mà enrich() đọc, khớp app/config.py::Settings. Thiếu một field
    # thì enrich() bắt AttributeError vào nhánh "LLM lỗi" và test fail ở tận assert
    # result.ok, không chỉ ra được nguyên nhân — đúng cách nó đã fail khi thêm
    # llm_max_tokens/llm_reasoning_effort.
    base = dict(llm_enabled=True, llm_model="groq/llama-3.3-70b-versatile",
               llm_api_key="", llm_temperature=0.0, llm_timeout_s=6.0, llm_max_tags=5,
               llm_max_tokens=2000, llm_reasoning_effort="",
               llm_tag_confidence_min=0.75)
    base.update(overrides)
    return SimpleNamespace(**base)


def test_build_prompt_lists_both_name_and_description():
    prompt = build_prompt("cầu thủ ăn mừng", VOCAB, max_tags=5)
    listing = prompt[1]["content"]
    assert "8: Thể thao (sports)" in listing
    assert "11: Giáo dục (education)" in listing


def test_build_prompt_omits_empty_parens_for_flat_vocab():
    """tag_vocab.json dạng phẳng {"id": "mô tả"} -> TagInfo.name rỗng."""
    flat_vocab = {0: TagInfo(id=0, name="", description="cảnh bóng đá trên sân",
                             point_count=0)}
    listing = build_prompt("bất kỳ", flat_vocab, max_tags=5)[1]["content"]
    assert "0: cảnh bóng đá trên sân" in listing
    assert "()" not in listing


def test_enrich_selects_valid_tag_and_passes_vocab_into_prompt(llm):
    llm.reply = {"tags": [8], "enriched": "football celebration"}
    result = asyncio.run(enrich("cầu thủ ăn mừng", VOCAB, settings()))

    assert result.ok
    assert result.tags == [8]
    assert "Thể thao" in llm.last_prompt


def test_enrich_drops_tag_id_outside_vocab(llm):
    llm.reply = {"tags": [8, 999], "enriched": ""}
    result = asyncio.run(enrich("truy vấn", VOCAB, settings()))

    assert result.tags == [8]  # 999 không nằm trong VOCAB -> bị loại, không crash
