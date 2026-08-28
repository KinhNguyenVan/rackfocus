"""Tách một câu truy vấn tiếng Việt thành N mô tả tiếng Anh cho CLIP.

Prompt nằm ở `segment_prompt.txt` cạnh file này (trước là `prompt.yaml` ở gốc repo).
Chạy SONG SONG với `enrich.py` trong `api/search_temporal.py::prepare` — hai lời gọi LLM
độc lập trên cùng câu gốc, tổng = max chứ không phải cộng.

Theo đúng hợp đồng của `enrich.py`: KHÔNG BAO GIỜ raise. Lỗi/timeout/JSON rác -> trả đúng
MỘT đoạn = câu gốc nguyên văn, kèm `error`. Hỏng kiểu đó rơi vào nhánh N=1 mà UI đã phải
xử lý sẵn (không tạo được chuỗi, mời chạy KIS), nên không cần đường lỗi riêng.
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger("app.segment")

# Đọc một lần lúc import. Prompt phải nằm trong package để ship được trong container BE.
_SYSTEM = Path(__file__).with_name("segment_prompt.txt").read_text(encoding="utf-8")


@dataclass
class Segment:
    order: int
    english_clip_query: str
    # KHÔNG có `label`. Prompt chốt mỗi đoạn đúng hai trường và nói rõ nhãn nguồn (E1,
    # "Sự kiện 1", ...) chỉ để LLM biết có bao nhiêu đoạn — không mang sang output.


@dataclass
class Segmentation:
    segments: list[Segment] = field(default_factory=list)
    model: str = ""
    latency_ms: float = 0.0
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error


def _fallback(query: str) -> list[Segment]:
    """Một đoạn duy nhất = câu gốc nguyên văn.

    Không đánh dấu gì đặc biệt vào dữ liệu: phân biệt "hỏng" với "đúng là chỉ có 1 đoạn"
    bằng `Segmentation.error` (rồi thành warning `llm_failed_segment`), không bằng một
    nhãn ma thuật mà UI phải so chuỗi để đoán ra.
    """
    return [Segment(order=1, english_clip_query=query)]


def _parse(content: str) -> list[Segment]:
    """Bóc JSON ARRAY khỏi output LLM.

    Khác `enrich._parse` bóc object (`\\{.*\\}`): prompt này bắt trả về một mảng.
    """
    m = re.search(r"\[.*\]", content, re.DOTALL)
    if not m:
        raise ValueError(f"không thấy JSON array trong output: {content[:120]!r}")
    data = json.loads(m.group(0))
    if not isinstance(data, list):
        raise TypeError(f"phải là list, nhận {type(data).__name__}")

    out: list[Segment] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        text = str(item.get("english_clip_query") or "").strip()
        if not text:
            continue
        # `order` đánh lại theo vị trí SAU khi lọc, không tin số LLM trả: nó đánh trùng
        # hoặc nhảy số được, mà UI dùng order làm khoá React lẫn thứ tự hiển thị.
        # Khoá thừa (`label` chẳng hạn, nếu LLM lỡ trả dù prompt đã cấm) bị bỏ im lặng.
        out.append(Segment(order=len(out) + 1, english_clip_query=text))
    if not out:
        raise ValueError("không còn đoạn nào sau khi lọc")
    return out


async def segment(query: str, settings) -> Segmentation:
    """Không bao giờ raise. Lỗi -> Segmentation(1 đoạn = câu gốc, error=...)."""
    if not settings.llm_enabled:
        return Segmentation(segments=_fallback(query))

    t0 = time.perf_counter()
    try:
        import litellm

        # reasoning_effort chỉ gửi khi có khai: model không phải reasoning sẽ lỗi nếu nhận.
        extra = ({"reasoning_effort": settings.llm_reasoning_effort}
                 if settings.llm_reasoning_effort else {})
        resp = await litellm.acompletion(
            model=settings.llm_model,
            messages=[{"role": "system", "content": _SYSTEM},
                      {"role": "user", "content": query}],
            api_key=settings.llm_api_key or None,
            temperature=settings.llm_temperature,
            timeout=settings.llm_timeout_s,
            max_tokens=settings.llm_max_tokens,
            **extra,
        )
        choice = resp.choices[0]
        content = choice.message.content or ""
        if not content.strip():
            raise ValueError(
                f"LLM trả content rỗng (finish_reason={choice.finish_reason}) — "
                "nếu là 'length' thì tăng LLM_MAX_TOKENS")
        return Segmentation(segments=_parse(content), model=settings.llm_model,
                            latency_ms=(time.perf_counter() - t0) * 1000)
    except Exception as ex:  # noqa: BLE001 — LLM lỗi KHÔNG được làm chết prepare
        log.warning("tách đoạn lỗi (%s: %s) -> lùi về 1 đoạn = câu gốc",
                    type(ex).__name__, ex)
        return Segmentation(segments=_fallback(query), model=settings.llm_model,
                            latency_ms=(time.perf_counter() - t0) * 1000,
                            error=f"{type(ex).__name__}: {ex}")
