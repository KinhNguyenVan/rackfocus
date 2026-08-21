"""LLM enrich + chọn tag, qua litellm (đổi provider chỉ bằng đổi tên model).

Vai trò của LLM ở đây là **chọn tag**, KHÔNG phải viết lại query. Query gốc của người dùng
được encode nguyên văn và chạy song song với lời gọi này (docs/search-design.md §3):

  - Tổng latency = max(LLM, encode) thay vì tổng cộng, tiết kiệm 80-300ms.
  - Giữ đúng cách diễn đạt của user. Một competitor biết domain ("Honda Cub xanh, thấy
    biển số") phụ thuộc vào chính chữ họ gõ; để LLM viết lại là lấy đi thứ đó.

`enriched_text` vẫn được trả về để log và để BE có thể dùng nếu muốn, nhưng đường mặc
định KHÔNG dùng nó cho embedding.

LUÔN có đường thoát: LLM lỗi/timeout/trả rác -> trả tags rỗng, tức search toàn bộ corpus.
Thà chậm hơn và đúng, còn hơn lọc sai rồi trả một trang tự tin từ vùng corpus không chứa
đáp án — lọc tag là CỨNG, frame ngoài tag đã chọn là không thể với tới.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

log = logging.getLogger("app.enrich")

_SYSTEM = """Bạn giúp chọn tag để thu hẹp phạm vi tìm kiếm trong kho video.

Cho một câu truy vấn và danh sách tag (id: mô tả), hãy chọn các tag mà cảnh cần tìm CÓ THỂ
thuộc vào.

Quy tắc:
- Mỗi khung hình chỉ mang ĐÚNG MỘT tag. Chọn thiếu tag đúng thì cảnh cần tìm sẽ không bao
  giờ xuất hiện trong kết quả, nên khi không chắc hãy chọn RỘNG hơn.
- Chọn tối đa {max_tags} tag.
- Không chắc chắn thì trả danh sách rỗng — hệ thống sẽ tìm toàn bộ kho.
- Chỉ trả JSON, không giải thích:
  {{"tags": [<id>, ...], "enriched": "<câu truy vấn viết rõ hơn bằng tiếng Anh>"}}"""


@dataclass
class Enrichment:
    tags: list[int] = field(default_factory=list)
    enriched_text: str = ""
    model: str = ""
    latency_ms: float = 0.0
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error


def _parse(content: str, valid: set[int], max_tags: int) -> tuple[list[int], str]:
    """Bóc JSON khỏi output LLM. Chịu được ```json fence và chữ thừa quanh."""
    m = re.search(r"\{.*\}", content, re.DOTALL)
    if not m:
        raise ValueError(f"không thấy JSON trong output: {content[:120]!r}")
    data = json.loads(m.group(0))

    raw = data.get("tags") or []
    if not isinstance(raw, list):
        raise TypeError(f"'tags' phải là list, nhận {type(raw).__name__}")

    tags: list[int] = []
    for t in raw:
        try:
            tid = int(t)
        except (TypeError, ValueError):
            continue
        # Bỏ tag LLM tự bịa. Gửi sang core sẽ bị INVALID_ARGUMENT, nhưng lọc ở đây thì
        # query vẫn chạy được với các tag hợp lệ còn lại.
        if tid in valid and tid not in tags:
            tags.append(tid)
    dropped = len(raw) - len(tags)
    if dropped:
        log.warning("bỏ %d tag không hợp lệ do LLM trả: %s", dropped, raw)

    return tags[:max_tags], str(data.get("enriched") or "")


def build_prompt(query: str, vocab: dict[int, str], max_tags: int) -> list[dict]:
    """vocab: {tag_id: description} — đúng định dạng file tag_vocab.json.

    Lưu ý chi phí: ~500 tag là 15-25k prompt token MỖI query. Nếu vocab lớn, cân nhắc
    rút ngắn mô tả hoặc lọc trước bằng embedding thay vì đưa cả bộ vào prompt.
    """
    listing = "\n".join(f"{tid}: {desc}" for tid, desc in sorted(vocab.items()))
    return [
        {"role": "system", "content": _SYSTEM.format(max_tags=max_tags)},
        {"role": "user", "content": f"Tag khả dụng:\n{listing}\n\nTruy vấn: {query}"},
    ]


async def enrich(query: str, vocab: dict[int, str], settings) -> Enrichment:
    """Không bao giờ raise. Lỗi -> Enrichment(tags=[], error=...) = search toàn bộ."""
    import time

    if not settings.llm_enabled or not vocab:
        return Enrichment(enriched_text=query,
                          error="" if not settings.llm_enabled else "vocab rỗng")

    t0 = time.perf_counter()
    try:
        import litellm

        resp = await litellm.acompletion(
            model=settings.llm_model,
            messages=build_prompt(query, vocab, settings.llm_max_tags),
            api_key=settings.llm_api_key or None,
            temperature=settings.llm_temperature,
            timeout=settings.llm_timeout_s,
            max_tokens=400,
        )
        content = resp.choices[0].message.content or ""
        tags, enriched = _parse(content, set(vocab), settings.llm_max_tags)
        return Enrichment(tags=tags, enriched_text=enriched or query,
                          model=settings.llm_model,
                          latency_ms=(time.perf_counter() - t0) * 1000)
    except Exception as ex:  # noqa: BLE001 — LLM lỗi KHÔNG được làm chết search
        log.warning("LLM lỗi (%s: %s) -> search toàn bộ corpus", type(ex).__name__, ex)
        return Enrichment(enriched_text=query, model=settings.llm_model,
                          latency_ms=(time.perf_counter() - t0) * 1000,
                          error=f"{type(ex).__name__}: {ex}")
