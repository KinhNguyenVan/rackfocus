"""GET /api/transcript/suggest — gõ keyword, gợi ý scene có lời thoại chứa keyword.

Dropdown kiểu Google dưới thanh search (FE bật bằng checkbox). Click 1 gợi ý -> FE mở scene
clip bằng VideoPlayer sẵn có. Xem services/transcript.py cho phần khớp FTS5.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from ..config import get_settings
from ..schemas.transcript import TranscriptSuggestItem, TranscriptSuggestResponse
from ..services import transcript

router = APIRouter()

# Dưới 2 ký tự thì prefix match quá rộng (mọi scene) và vô nghĩa cho autocomplete -> bỏ qua,
# không đụng DB.
_MIN_QUERY_LEN = 2


@router.get("/transcript/suggest", response_model=TranscriptSuggestResponse)
def suggest(
    q: str = Query("", description="keyword trong lời thoại"),
    limit: int | None = Query(None, ge=1, le=50),
) -> TranscriptSuggestResponse:
    st = get_settings()
    query = q.strip()
    if len(query) < _MIN_QUERY_LEN:
        return TranscriptSuggestResponse(query=query, items=[])

    index = transcript.get()
    if index is None:
        # transcript_db_path chưa cấu hình hoặc mở lỗi lúc khởi động.
        raise HTTPException(503, "transcript index chưa sẵn sàng")

    top = min(limit or st.transcript_suggest_limit, 50)
    items = [TranscriptSuggestItem(**row) for row in index.suggest(query, top)]
    return TranscriptSuggestResponse(query=query, items=items)
