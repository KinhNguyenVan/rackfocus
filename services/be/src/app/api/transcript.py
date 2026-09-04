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

# Trần độ dài query. Endpoint này chạy MỖI KEYSTROKE nên là hot path: không có trần thì một
# client (hoặc paste nhầm) đẩy được 100KB/lần gõ vào `_TOKEN_RE.findall` + biểu thức MATCH.
# Regex `\w+` đã kín về mặt injection (fuzz đủ input độc không phá được cú pháp FTS5), nhưng
# "không phá được cú pháp" khác với "không tốn CPU". 200 ký tự dư sức cho autocomplete.
_MAX_QUERY_LEN = 200


@router.get("/transcript/suggest", response_model=TranscriptSuggestResponse)
def suggest(
    q: str = Query("", max_length=_MAX_QUERY_LEN, description="keyword trong lời thoại"),
    limit: int | None = Query(None, ge=1, le=50),
) -> TranscriptSuggestResponse:
    st = get_settings()
    query = q.strip()
    if len(query) < _MIN_QUERY_LEN:
        return TranscriptSuggestResponse(query=query, items=[])

    index = transcript.get()
    if index is None:
        # transcript_db_path chưa cấu hình ("disabled") hoặc mở lỗi lúc khởi động ("failed").
        # /readyz phân biệt hai ca đó qua field `transcript_index`.
        raise HTTPException(503, f"transcript index chưa sẵn sàng ({transcript.status()})")

    top = min(limit or st.transcript_suggest_limit, 50)
    try:
        rows = index.suggest(query, top)
    except transcript.TranscriptIndexError as ex:
        # Index mở được lúc khởi động nhưng giờ hỏng (file bị swap/xoá). 503, không 500:
        # đây là artifact vận hành thiếu, không phải bug xử lý request.
        raise HTTPException(503, f"transcript index lỗi: {ex}") from ex
    items = [TranscriptSuggestItem(**row) for row in rows]
    return TranscriptSuggestResponse(query=query, items=items)
