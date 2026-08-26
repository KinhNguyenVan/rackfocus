"""LLM enrich + chọn tag, qua litellm (đổi provider chỉ bằng đổi tên model).

LLM làm HAI việc, cả hai đều nằm trên đường chính:

  1. Chọn tag (lĩnh vực) để thu hẹp phạm vi search.
  2. Viết lại query thành `enriched_text` tiếng Anh — và bản này ĐƯỢC dùng để embedding
     (api/search.py). Trước đây encode query gốc để chạy song song với LLM, nhưng
     tokenizer SigLIP tốn 4-7,5x token cho tiếng Việt và giới hạn 64 token là cứng, nên
     query tiếng Việt dài bị cắt mất phần chi tiết ở cuối. Xem comment trong
     api/search.py để biết số đo cụ thể.

Vì `enriched_text` giờ quyết định vector query, prompt bắt nó chỉ giữ HÀNH ĐỘNG và SỰ VẬT
NHÌN THẤY ĐƯỢC — thêm chữ không nhìn thấy được là kéo vector ra khỏi vùng ảnh cần tìm.

Quy tắc chọn tag COPY từ prompt ingest (services/be/src/app/services/taxonomy.py) để hai
bên không phân loại lệch nhau, kèm một lớp guard regex bù tag LLM bỏ sót.

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
from . import taxonomy

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

Mỗi lĩnh vực gồm các chủ đề con sau — dùng chúng để hiểu lĩnh vực đó BAO GỒM những gì
(không cần trả về topic):
{topics}

Quy tắc chọn lĩnh vực — GIỐNG HỆT quy tắc đã dùng lúc gán lĩnh vực cho khung hình, phải
theo đúng để hai bên không lệch nhau:
{rules}
- Chọn thiếu lĩnh vực đúng thì cảnh cần tìm sẽ không bao giờ xuất hiện trong kết quả, nên
  khi không chắc hãy chọn RỘNG hơn (nhiều lĩnh vực liên quan) thay vì chọn hẹp.
- Chọn tối đa {max_tags} lĩnh vực.
- Truy vấn không gắn với lĩnh vực rõ ràng (ví dụ chỉ tả màu sắc, hành động chung) thì trả
  danh sách rỗng — hệ thống sẽ tìm toàn bộ kho.

Về "confidence" — số 0..1, mức chắc chắn của bạn về danh sách "tags":
- Cao (>= 0.8) chỉ khi truy vấn nêu rõ chủ thể/hoạt động thuộc hẳn về lĩnh vực đã chọn.
- Thấp (<= 0.5) khi truy vấn mơ hồ, thiếu ngữ cảnh, chỉ tả hình dáng/màu sắc/động tác
  chung, hoặc bạn phải suy đoán để chọn lĩnh vực.
- ĐỪNG mặc định số cao. Khai thấp không bị phạt gì: hệ thống sẽ tự search rộng hơn, còn
  khai cao mà sai thì cảnh cần tìm bị loại hẳn khỏi kết quả và không cách nào lấy lại.
Về "enriched" — đây là mô tả để KHỚP HÌNH ẢNH, không phải câu văn hay:
- BẮT BUỘC viết bằng tiếng Anh.
- Chỉ giữ những HÀNH ĐỘNG và SỰ VẬT THẬT SỰ NHÌN THẤY ĐƯỢC mà truy vấn nói tới. Bỏ hết
  phần không nhìn thấy được: kiến thức nền, lịch sử, đánh giá, cảm xúc, mục đích, tên
  quốc gia/thời kỳ nếu truy vấn không nêu.
- KHÔNG thêm chi tiết truy vấn không có. Không suy diễn bối cảnh, không đoán thêm vật thể,
  không tô vẽ. Truy vấn ngắn thì "enriched" cũng ngắn — chỉ dịch, không nối thêm.
  Ví dụ SAI  : "chùa một cột" -> "Chùa Một Cột, a historic pagoda in Vietnam"
               ("historic", "in Vietnam" là kiến thức thêm, không nhìn thấy được)
  Ví dụ ĐÚNG : "chùa một cột" -> "Chùa Một Cột"
  Ví dụ ĐÚNG : "cầu thủ bóng đá ăn mừng" -> "football player celebrating"
- BỎ từ chỉ đánh giá, cảm xúc, cảm nhận KỂ CẢ KHI truy vấn có chúng — chúng không phải
  vật thể hay hành động nhìn thấy được ("hùng vĩ", "tuyệt đẹp", "choáng ngợp", "nổi
  tiếng", "quan trọng", "đáng chú ý"...).
  Ví dụ SAI  : "cảnh hùng vĩ của vịnh Hạ Long khiến du khách choáng ngợp"
               -> "majestic view of Ha Long Bay, tourists amazed"
  Ví dụ ĐÚNG : "cảnh hùng vĩ của vịnh Hạ Long khiến du khách choáng ngợp"
               -> "Vịnh Hạ Long, tourists"
- Tên riêng tiếng Việt (địa danh, di tích, tổ chức, sự kiện, nhân vật) giữ NGUYÊN dạng
  tiếng Việt, ĐỦ DẤU, KHÔNG dịch và KHÔNG đổi trật tự từ:
    "Chùa Một Cột" -> "Chùa Một Cột"      (không phải "One Pillar Pagoda")
    "vịnh Hạ Long" -> "Vịnh Hạ Long"      (không phải "Ha Long Bay" hay "Ha Long")
    "phố cổ Hội An" -> "Phố Cổ Hội An"    (không phải "Hoi An ancient town")
- Chỉ trả JSON, không giải thích:
  {{"tags": [<id>, ...], "confidence": <0..1>, "enriched": "<hành động/sự vật nhìn thấy được, bằng tiếng Anh, tên riêng tiếng Việt giữ nguyên>"}}"""


@dataclass
class Enrichment:
    tags: list[int] = field(default_factory=list)
    enriched_text: str = ""
    model: str = ""
    latency_ms: float = 0.0
    error: str = ""
    # Tag guard regex đóng góp. Trả về để thấy được từ ngoài: nếu guard phải bù thường
    # xuyên thì prompt đang lệch so với taxonomy ingest.
    guard_added: list[int] = field(default_factory=list)
    # LLM tự khai (0..1). Thiếu -> 0.0, tức không tin, xem `_confidence`.
    confidence: float = 0.0
    # "llm" | "guard_low_confidence" | "llm_empty" — ai đã quyết định tập tag.
    tag_source: str = ""

    @property
    def ok(self) -> bool:
        return not self.error


def guard_tags(query: str, vocab: dict[int, TagInfo], enriched: str = "") -> list[int]:
    """Tag id guard regex nhận ra (xem services/taxonomy.py).

    `enriched` là bản viết lại tiếng Anh — nguồn CHÍNH vì từ vựng tiếng Anh ít nhập
    nhằng hơn tiếng Việt đơn âm ghép. `query` gốc giữ vai trò dự phòng.
    """
    slug_to_id = {info.name: tid for tid, info in vocab.items() if info.name}
    return sorted(slug_to_id[s] for s in taxonomy.domains_from_text(query, enriched)
                  if s in slug_to_id)


def decide_tags(query: str, vocab: dict[int, TagInfo], llm_tags: list[int],
                confidence: float, max_tags: int, min_confidence: float,
                enriched: str = "") -> tuple[list[int], list[int], str]:
    """(tags dùng thật, tag do guard đóng góp, nguồn quyết định).

    Ba đường, theo đúng thứ tự:

    1. `confidence < min_confidence` -> BỎ tag của LLM, dùng tag của guard. Guard không
       nhận ra gì -> rỗng (search toàn kho). Lọc tag là CỨNG nên một phán đoán không
       chắc mà vẫn áp vào là cách nhanh nhất làm đáp án biến mất khỏi kết quả; guard
       khớp từ khoá rời nên hẹp hơn nhưng không bịa.
    2. LLM đủ tự tin mà trả rỗng -> giữ rỗng. Rỗng là tín hiệu CÓ CHỦ Ý "không lĩnh vực
       nào rõ ràng"; bù guard vào đây sẽ biến nó thành lọc hẹp người dùng không yêu cầu.
    3. LLM đủ tự tin và có tag -> lấy tag LLM, guard chỉ BÙ THÊM phần LLM bỏ sót. Thừa
       tag chỉ làm tập ứng viên rộng hơn (chậm hơn chút); thiếu tag là mất đáp án.
    """
    guard = guard_tags(query, vocab, enriched)

    if confidence < min_confidence:
        chosen = guard[:max_tags]
        return chosen, chosen, "guard_low_confidence"

    if not llm_tags:
        return [], [], "llm_empty"

    added = [t for t in guard if t not in llm_tags]
    merged = (llm_tags + added)[:max_tags]
    # Giữ trọn tag LLM (nó có ngữ cảnh cả câu), cắt phần guard bù nếu vượt max_tags.
    return merged, [t for t in added if t in merged], "llm"


def _confidence(data: dict) -> float:
    """`confidence` LLM khai, kẹp về [0,1]. Thiếu/không parse được -> 0.0.

    Mặc định 0.0 (không phải 1.0) là CÓ CHỦ Ý: model không khai confidence thì coi như
    không tự tin, và quyết định rơi về guard. Mặc định 1.0 sẽ làm ngưỡng vô hiệu ngay
    khi đổi sang model không tuân thủ schema — kiểu hỏng âm thầm tệ nhất.
    """
    try:
        return max(0.0, min(1.0, float(data.get("confidence"))))
    except (TypeError, ValueError):
        return 0.0


def _parse(content: str, valid: set[int], max_tags: int) -> tuple[list[int], str, float]:
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

    return tags[:max_tags], str(data.get("enriched") or ""), _confidence(data)


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
    slugs = [info.name for _, info in sorted(vocab.items()) if info.name]
    return [
        {"role": "system", "content": _SYSTEM.format(
            max_tags=max_tags,
            topics=taxonomy.topics_listing(slugs) or "  (không có dữ liệu topic)",
            rules=taxonomy.DISAMBIGUATION_RULES)},
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

        # reasoning_effort chỉ gửi khi có khai: model không phải reasoning sẽ lỗi nếu nhận.
        extra = ({"reasoning_effort": settings.llm_reasoning_effort}
                 if settings.llm_reasoning_effort else {})
        resp = await litellm.acompletion(
            model=settings.llm_model,
            messages=build_prompt(query, vocab, settings.llm_max_tags),
            api_key=settings.llm_api_key or None,
            temperature=settings.llm_temperature,
            timeout=settings.llm_timeout_s,
            max_tokens=settings.llm_max_tokens,
            **extra,
        )
        choice = resp.choices[0]
        content = choice.message.content or ""
        # finish_reason vào thông báo lỗi: khi model reasoning tiêu hết max_tokens thì
        # content rỗng/JSON cụt, và nếu không thấy "length" ở đây thì rất khó đoán ra.
        if not content.strip():
            raise ValueError(
                f"LLM trả content rỗng (finish_reason={choice.finish_reason}, "
                f"completion_tokens={getattr(resp.usage, 'completion_tokens', '?')}) — "
                "nếu là 'length' thì tăng LLM_MAX_TOKENS hoặc hạ LLM_REASONING_EFFORT")
        llm_tags, enriched, conf = _parse(content, set(vocab), settings.llm_max_tags)
        tags, added, source = decide_tags(
            query, vocab, llm_tags, conf,
            settings.llm_max_tags, settings.llm_tag_confidence_min,
            enriched=enriched)
        if source == "guard_low_confidence":
            log.info("confidence %.2f < %.2f -> bỏ tag LLM %s, dùng guard %s cho %r",
                     conf, settings.llm_tag_confidence_min, llm_tags, tags, query[:60])
        elif added:
            log.info("guard bù tag %s (LLM chọn %s, conf %.2f) cho %r",
                     added, llm_tags, conf, query[:60])
        return Enrichment(tags=tags, enriched_text=enriched or query,
                          model=settings.llm_model, guard_added=added,
                          confidence=conf, tag_source=source,
                          latency_ms=(time.perf_counter() - t0) * 1000)
    except Exception as ex:  # noqa: BLE001 — LLM lỗi KHÔNG được làm chết search
        log.warning("LLM lỗi (%s: %s) -> search toàn bộ corpus", type(ex).__name__, ex)
        return Enrichment(enriched_text=query, model=settings.llm_model,
                          latency_ms=(time.perf_counter() - t0) * 1000,
                          error=f"{type(ex).__name__}: {ex}")
