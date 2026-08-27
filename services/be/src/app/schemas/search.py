"""Pydantic: SearchRequest/Response cho POST /api/search và GET /api/tags.

Tách khỏi `api/search.py` để router chỉ còn luồng xử lý, còn hợp đồng HTTP nằm một chỗ
— FE đọc file này là đủ, không phải lần trong code handler.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

# Khớp FilterStrategy trong proto — trả tên ra ngoài cho dễ đọc.
STRATEGY_NAMES: dict[int, str] = {
    0: "unspecified", 1: "pre", 2: "post", 3: "exact_subset",
}


class SearchRequest(BaseModel):
    text: str = Field(min_length=1)
    top_k: int | None = None
    # Tắt LLM để tìm toàn bộ corpus. Đây là đường lùi khi LLM chọn sai tag.
    use_llm: bool = True
    # Chỉ định tag thẳng, bỏ qua LLM (dùng để debug hoặc khi user tự chọn).
    tags: list[int] | None = None
    min_score: float | None = None
    # False (mặc định) = "rerank": core tự chọn — có tag hẹp thì EXACT_SUBSET trên
    # subset, không thì HNSW+SQ8 coarse rồi rerank exact trên top rerank_candidates.
    # True = "exact": ép brute-force TOÀN candidate (hoặc toàn corpus nếu không tag),
    # bỏ qua HNSW hoàn toàn — chậm hơn nhưng không xấp xỉ.
    exact: bool = False


class Hit(BaseModel):
    point_id: int
    row: int
    score: float
    rank: int
    video_name: str = ""
    frame: int = 0
    keyframe_time: float = 0.0
    start_sec: float = 0.0
    end_sec: float = 0.0
    keyframe_url: str = ""
    clip_url: str = ""
    scene_idx: int = 0
    has_speech: bool = False


class EnrichmentInfo(BaseModel):
    """Trước là `dict` không kiểu. Thành model vì số field cứ tăng dần mà không có chỗ
    nào ghi hợp đồng, nên FE phải đoán field nào tồn tại."""

    model: str = ""
    tags: list[int] = Field(default_factory=list)
    enriched_text: str = ""
    error: str = ""
    used_llm: bool = False

    # Tag guard regex đóng góp (services/taxonomy.py). Bù thường xuyên = prompt đang
    # lệch so với taxonomy ingest.
    guard_added: list[int] = Field(default_factory=list)

    # LLM tự khai (0..1) và AI đã quyết định tập tag. Không phơi ra thì không phân biệt
    # được "LLM chọn thế" với "guard chọn thay vì LLM không chắc".
    confidence: float = 0.0
    tag_source: str = ""   # "llm" | "guard_low_confidence" | "llm_empty"

    # Text THỰC SỰ được encode — khác query gốc khi có enrich. Không expose thì không
    # cách nào biết vector được dựng từ chữ nào.
    encoded_text: str = ""

    # True = lấy từ cache, KHÔNG gọi LLM ở request này. Cần cờ này vì `timings_ms.llm`
    # là latency của lần gọi GỐC (nằm trong object đã cache), nên nó báo hàng nghìn ms
    # cho một request chỉ mất vài ms — không có cờ thì số liệu nói sai.
    cached: bool = False


class SearchResponse(BaseModel):
    hits: list[Hit]
    # Phơi ra để user biết mình vừa tìm trong bao nhiêu phần của kho.
    tags_used: list[int]
    candidate_count: int
    corpus_count: int
    strategy: str
    warnings: list[str]
    snapshot_ver: str
    timings_ms: dict[str, float]
    enrichment: EnrichmentInfo


class TagItem(BaseModel):
    id: int
    name: str
    description: str
    point_count: int


class TagsResponse(BaseModel):
    snapshot_ver: str
    count: int
    tags: list[TagItem]
