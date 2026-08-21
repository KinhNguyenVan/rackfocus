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

from ..clients.searchcore import TagInfo

log = logging.getLogger("app.enrich")

# Tag hiện tại = domain_id, tức LĨNH VỰC/CHỦ ĐỀ tin tức của cảnh (13 giá trị cố định,
# xem ingest/build_tags.py) — không phải mô tả cảnh tự do như bản nháp ban đầu. Nói rõ
# "lĩnh vực" thay vì "tag" trong prompt để LLM dùng kiến thức phân loại tin tức có sẵn,
# thay vì đoán một tập tag ngữ nghĩa mở nó không biết trước.
_SYSTEM = """Bạn giúp chọn LĨNH VỰC/CHỦ ĐỀ tin tức để thu hẹp phạm vi tìm kiếm trong kho
video. Mỗi khung hình trong kho đã được gán sẵn đúng một lĩnh vực (ví dụ "Thể thao",
"Giáo dục", "Kinh tế - Tài chính").

Cho một câu truy vấn và danh sách lĩnh vực (id: mô tả tiếng Việt (tên)), hãy chọn (các)
lĩnh vực mà cảnh cần tìm CÓ THỂ thuộc vào.

Quy tắc:
- Chọn thiếu lĩnh vực đúng thì cảnh cần tìm sẽ không bao giờ xuất hiện trong kết quả, nên
  khi không chắc hãy chọn RỘNG hơn (nhiều lĩnh vực liên quan) thay vì chọn hẹp.
- Chọn tối đa {max_tags} lĩnh vực.
- Truy vấn không gắn với lĩnh vực rõ ràng (ví dụ chỉ tả màu sắc, hành động chung) thì trả
  danh sách rỗng — hệ thống sẽ tìm toàn bộ kho.
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


def _listing_line(tag_id: int, info: TagInfo) -> str:
    # name rỗng khi tag_vocab.json ở dạng phẳng {"id": "mô tả"} (snapshot.py chấp nhận cả
    # hai dạng) — không in "()" rỗng trong trường hợp đó.
    return f"{tag_id}: {info.description} ({info.name})" if info.name else \
        f"{tag_id}: {info.description}"


def build_prompt(query: str, vocab: dict[int, TagInfo], max_tags: int) -> list[dict]:
    """vocab: {tag_id: TagInfo(name, description, ...)} lấy qua `tagvocab.get()`.

    Chỉ 13 lĩnh vực (domain_id) nên prompt vài trăm token, không phải 15-25k như giả định
    ban đầu cho vocab ~500 tag tự do — xem ingest/build_tags.py.
    """
    listing = "\n".join(
        _listing_line(tid, info) for tid, info in sorted(vocab.items())
    )
    return [
        {"role": "system", "content": _SYSTEM.format(max_tags=max_tags)},
        {"role": "user", "content": f"Lĩnh vực khả dụng:\n{listing}\n\nTruy vấn: {query}"},
    ]


async def enrich(query: str, vocab: dict[int, TagInfo], settings) -> Enrichment:
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
