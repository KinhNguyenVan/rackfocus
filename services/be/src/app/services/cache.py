"""Cache query embedding + kết quả enrich theo text đã normalize.

Vì sao đáng làm trước các stub khác: LLM enrich tốn 400-3000ms và trước giờ KHÔNG cache
gì cả, trong khi encode tốn ~170ms. Hai thứ này chiếm gần hết e2e (core search chỉ
12-48ms). Mà lúc thi thì người dùng gõ lại/tinh chỉnh cùng một câu liên tục.

Cache nằm trong RAM của TIẾN TRÌNH BE (một `OrderedDict`), KHÔNG dùng Redis:
`clients/redis.py` vẫn là stub và Redis không chạy trong lúc dev. Ba hệ quả:

* Restart BE là mất cache. Chấp nhận được — đây là tối ưu độ trễ, không phải nguồn sự
  thật; và cũng chính là cách "invalidate" khi sửa prompt (xem đoạn dưới).
* KHÔNG chia sẻ giữa các worker. Hiện chạy `uvicorn` 1 worker nên không sao, nhưng bật
  `--workers N` thì mỗi worker có cache riêng -> hit rate chia cho N. Không sai kết quả
  (temperature=0 nên các worker tính ra cùng đáp án), chỉ là chậm hơn kỳ vọng. Cần chia
  sẻ thật thì phải làm `clients/redis.py`.
* Bộ nhớ có chặn: xem `maxsize` ở cuối file, kèm số đo thật.

An toàn về mặt đúng đắn:

* `temperature=0` nên cùng một query cho cùng một tag — chính lý do đã ghi ở
  `config.py`. Không có nó thì cache enrich sẽ khoá cứng một kết quả ngẫu nhiên.
* Khoá CÓ `snapshot_ver`: tag id chỉ có nghĩa trong đúng bản snapshot đó (xem
  `services/tagvocab.py`). Thiếu nó thì sau khi core swap snapshot, BE sẽ trả tag id đã
  đổi nghĩa — sai âm thầm, không lỗi nào báo. Vector cũng phụ thuộc encoder đi kèm
  snapshot nên dùng chung quy ước.

Cái khoá KHÔNG bắt được: sửa prompt trong `services/enrich.py`. Prompt không có version
nên entry cũ thành lệch. Chấp nhận vì sửa prompt luôn kèm restart tiến trình, mà restart
là xoá cache.
"""
from __future__ import annotations

import logging
import re
import time
import unicodedata
from collections import OrderedDict
from typing import Any

log = logging.getLogger("app.cache")


def normalize(text: str) -> str:
    """Gộp khoảng trắng + NFC. CỐ Ý không lowercase.

    Lowercase sẽ gộp "Chùa Một Cột" với "chùa một cột" thành một entry. Với embedding thì
    vô hại (SigLIP `canonicalize` tự lowercase nên vector y hệt), nhưng LLM thì nhìn
    thấy đúng chữ user gõ, và hoa/thường ở tên riêng có thể đổi output. Bỏ lowercase là
    mất vài lần hit, giữ lại là có nguy cơ trả sai — chọn cái đầu.
    """
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip()


class TTLCache:
    """LRU + TTL, chặn cả số entry lẫn tuổi entry.

    Chặn số entry vì `searchcore.encode` trả `list[float]` chứ không phải numpy: một
    vector 1152 chiều tốn **36.8 KB** đo thật (1152 con trỏ + 1152 object float của
    Python), gấp 8 lần 4.5 KB của numpy float32 cùng kích thước. 256 entry = ~9 MB;
    không giới hạn thì một phiên thi dài đủ làm phình bộ nhớ.

    Chặn tuổi vì entry cũ tương ứng prompt/model cũ.
    """

    def __init__(self, maxsize: int = 512, ttl_s: float = 3600.0, name: str = "") -> None:
        self._data: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self._maxsize = max(1, maxsize)
        self._ttl = ttl_s
        self._name = name
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Any | None:
        item = self._data.get(key)
        if item is None:
            self.misses += 1
            return None
        expires_at, value = item
        if expires_at < time.monotonic():
            # Hết hạn thì xoá luôn, không để rác chiếm chỗ của entry còn dùng được.
            del self._data[key]
            self.misses += 1
            return None
        self._data.move_to_end(key)
        self.hits += 1
        return value

    def set(self, key: str, value: Any) -> None:
        self._data[key] = (time.monotonic() + self._ttl, value)
        self._data.move_to_end(key)
        while len(self._data) > self._maxsize:
            self._data.popitem(last=False)   # bỏ entry cũ nhất theo lần dùng

    def clear(self) -> None:
        self._data.clear()
        self.hits = self.misses = 0

    def stats(self) -> dict[str, Any]:
        total = self.hits + self.misses
        return {
            "name": self._name,
            "size": len(self._data),
            "maxsize": self._maxsize,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 4) if total else 0.0,
        }


# Hai cache riêng, không gộp: encode dùng được cả khi enrich lỗi/bị tắt, và vòng đời
# của chúng khác nhau (đổi prompt chỉ làm hỏng cache enrich, không hỏng cache vector).
#
# maxsize lệch nhau vì kích thước entry lệch nhau ~100 lần: một vector là 36.8 KB, còn
# một `Enrichment` chỉ là vài chuỗi ngắn + list int. 256 vector = ~9 MB, đủ cho một phiên
# thi mà vẫn nhỏ so với RSS ~32 MB của BE.
embedding = TTLCache(maxsize=256, ttl_s=3600.0, name="embedding")
enrichment = TTLCache(maxsize=2048, ttl_s=3600.0, name="enrichment")


def embedding_key(text: str, snapshot_ver: str) -> str:
    return f"{snapshot_ver}\x00{normalize(text)}"


def enrichment_key(text: str, snapshot_ver: str, model: str, max_tags: int,
                   confidence_min: float) -> str:
    """Mọi tham số đổi được mà có thể đổi output đều phải nằm trong khoá.

    `max_tags` cắt danh sách tag; `confidence_min` quyết định lấy tag của LLM hay của
    guard. Thiếu chúng thì sửa config xong vẫn nhận lại kết quả tính theo config cũ.
    """
    return f"{snapshot_ver}\x00{model}\x00{max_tags}\x00{confidence_min}\x00{normalize(text)}"


def all_stats() -> list[dict[str, Any]]:
    return [embedding.stats(), enrichment.stats()]
