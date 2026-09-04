"""Keyword search transcript: mở SQLite FTS5 (ship kèm snapshot) và gợi ý scene.

Nguồn dữ liệu build ở `services/ingest/src/ingest/build_transcript_index.py`. Online KHÔNG
có Postgres nên transcript nằm ở file bất biến này; BE mở read-only lúc khởi động
(`main.py` lifespan) và giữ 1 instance module-level.

Khớp keyword = FTS5 prefix MATCH + xếp hạng bm25() + snippet() highlight. KHÔNG gọi LLM:
dropdown as-you-type phải rẻ và nhanh, không kẹt round-trip mỗi keystroke.
"""
from __future__ import annotations

import logging
import re
import sqlite3
from collections.abc import Callable

log = logging.getLogger("app.services.transcript")

# Tên bảng phải khớp `ingest/build_transcript_index.py` (META_TABLE/FTS_TABLE).
META_TABLE = "scenes_meta"
FTS_TABLE = "transcript_fts"

# Token = cụm chữ/số liên tiếp (giữ Unicode để không cắt vụn tiếng Việt có dấu). Mọi ký tự
# khác (kể cả toán tử FTS5 " * : ( )) bị loại -> query người dùng không thể phá cú pháp MATCH.
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)

_QUERY_SQL = (
    "SELECT m.video_name, m.scene_idx, m.start_sec, m.end_sec, m.clip_key, m.keyframe_key,"
    f"       snippet({FTS_TABLE}, 0, '[', ']', '…', 12) AS snippet "
    f"FROM {FTS_TABLE} JOIN {META_TABLE} m ON m.rowid = {FTS_TABLE}.rowid "
    f"WHERE {FTS_TABLE} MATCH ? ORDER BY bm25({FTS_TABLE}) LIMIT ?"
)


class TranscriptIndexError(RuntimeError):
    """Index không dùng được (thiếu/hỏng/rỗng, hoặc truy vấn lỗi ở tầng SQLite).

    Tách riêng khỏi `Exception` chung để endpoint map được sang 503 ("chưa sẵn sàng")
    thay vì để lọt lên FastAPI thành 500 ("lỗi server không rõ") — 503 nói đúng bản
    chất (thiếu artifact vận hành) và khớp tài liệu + comment ở `main.py`.
    """


def build_match(q: str) -> str:
    """Query người dùng -> biểu thức FTS5 MATCH prefix an toàn.

    Mỗi token thành `token*` (prefix, hợp autocomplete), nối bằng khoảng trắng = AND ngầm.
    Không token hợp lệ -> "" (caller phải coi là "không tìm").

    Lowercase token: FTS5 nhận diện AND/OR/NOT là toán tử KHI viết hoa; hạ về thường khiến
    "OR" người dùng gõ thành prefix "or*" vô hại (matching vốn không phân biệt hoa/thường)."""
    tokens = _TOKEN_RE.findall((q or "").lower())
    return " ".join(f"{t}*" for t in tokens)


def _default_resolver() -> Callable[[str], str]:
    """Dựng resolver key->URL công khai bằng S3 helper (giống browse.py). Tách ra để test
    inject được resolver thuần, không cần AWS."""
    from ..clients.s3 import AWSStorageHelper

    helper = AWSStorageHelper()

    def resolve(key: str) -> str:
        if not key:
            return ""
        if key.startswith("http://") or key.startswith("https://"):
            return key  # ingest đôi khi đã ghi URL tuyệt đối
        return helper.get_s3_public_url(key)

    return resolve


class TranscriptIndex:
    """SQLite FTS5 read-only. File bất biến; mở connection MỚI mỗi truy vấn.

    FastAPI chạy endpoint sync trong threadpool -> nhiều thread có thể suggest() cùng lúc.
    Dùng chung 1 connection sqlite qua các thread là nguồn lỗi khó lần; mở read-only mỗi
    lần (file đã ở page cache) rẻ hơn nhiều so với chi phí đó.
    """

    def __init__(self, db_path: str, url_resolver: Callable[[str], str] | None = None):
        self._uri = f"file:{db_path}?mode=ro"
        # FAIL-CLOSED lúc khởi tạo. Phải TRUY VẤN THẬT, không chỉ connect+close:
        # `sqlite3.connect(...)` mở lazy, nên file rác (không phải sqlite) và file sqlite
        # hợp lệ nhưng THIẾU BẢNG `scenes_meta` đều KHÔNG raise ở connect — chỉ thiếu hẳn
        # file mới raise. Cả hai ca đó lọt qua thì mọi suggest() sau này vỡ ở tầng HTTP
        # (500) thay vì chặn ngay lúc khởi động.
        #
        # `count(*) == 0` cũng là lỗi: build bị kill giữa `CREATE TABLE` (tự commit) và
        # `finalize()` (chỗ commit INSERT duy nhất) để lại file ĐÚNG SCHEMA, 0 ROW. Khi đó
        # endpoint trả 200 {"items": []} cho mọi keyword — không phân biệt được với
        # "không có câu thoại nào khớp", nên sai kiểu im lặng, đúng lúc thi.
        # `connect` nằm TRONG try: thiếu file thì chính nó raise, còn file rác/thiếu bảng
        # thì `execute` mới raise — gom cả hai về một loại lỗi để caller không phải phân
        # biệt `OperationalError` với `DatabaseError`.
        try:
            conn = sqlite3.connect(self._uri, uri=True)
            try:
                (n,) = conn.execute(f"SELECT count(*) FROM {META_TABLE}").fetchone()
            finally:
                conn.close()
        except sqlite3.Error as ex:
            raise TranscriptIndexError(
                f"transcript index không đọc được ({type(ex).__name__}: {ex}): {db_path}"
            ) from ex
        if not n:
            raise TranscriptIndexError(
                f"transcript index RỖNG (0 row trong {META_TABLE}) — build bị dở dang: "
                f"{db_path}"
            )
        self.row_count = int(n)
        self._resolve = url_resolver or _default_resolver()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._uri, uri=True)

    def suggest(self, q: str, limit: int) -> list[dict]:
        match = build_match(q)
        if not match:
            return []
        conn = self._connect()
        try:
            rows = conn.execute(_QUERY_SQL, (match, limit)).fetchall()
        except sqlite3.Error as ex:
            # File bị đổi/hỏng SAU khi khởi động xong (swap snapshot, xoá file, FTS lệch
            # bảng meta). Không bọc thì `sqlite3.DatabaseError` chạy thẳng lên FastAPI =
            # 500 cho MỌI keystroke, mãi mãi. 503 nói đúng chuyện: artifact vận hành hỏng.
            log.error("truy vấn transcript index lỗi: %s: %s", type(ex).__name__, ex)
            raise TranscriptIndexError(f"transcript index lỗi: {ex}") from ex
        finally:
            conn.close()
        out = []
        for video_name, scene_idx, start_sec, end_sec, clip_key, keyframe_key, snippet in rows:
            out.append({
                "video_name": video_name,
                "scene_idx": scene_idx,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "clip_url": self._resolve(clip_key or ""),
                "keyframe_url": self._resolve(keyframe_key or ""),
                "snippet": snippet,
            })
        return out

    def close(self) -> None:
        # Không giữ connection sống (mở theo từng truy vấn) -> không có gì để đóng.
        pass


# ── instance module-level (nạp ở lifespan) ──────────────────────────
_index: TranscriptIndex | None = None

# "disabled" = TRANSCRIPT_DB_PATH rỗng (tính năng tắt có ý thức) | "loaded" = mở được |
# "failed" = có cấu hình nhưng mở lỗi. Ba trạng thái này phải PHÂN BIỆT ĐƯỢC từ ngoài:
# cả "disabled" và "failed" đều làm endpoint trả 503, nhưng một cái là cố ý còn cái kia
# là sự cố cần sửa. Không có field này thì người vận hành nhìn 503 không biết đường nào.
_status: str = "disabled"


def load(db_path: str) -> None:
    """Mở index. Lỗi (thiếu/hỏng/rỗng file) ném ra để lifespan log — endpoint trả 503."""
    global _index, _status
    try:
        index = TranscriptIndex(db_path)
    except Exception:
        _index, _status = None, "failed"
        raise
    _index, _status = index, "loaded"
    log.info("transcript index mở từ %s (%d scene có thoại)", db_path, index.row_count)


def get() -> TranscriptIndex | None:
    return _index


def status() -> str:
    """"loaded" | "failed" | "disabled" — cho /readyz. Xem `_status`."""
    return _status
