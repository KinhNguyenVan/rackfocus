"""Cache embedding + enrich: LRU/TTL và quy ước khoá.

Khoá là chỗ dễ sai âm thầm nhất: thiếu một tham số trong khoá thì đổi config/snapshot
xong vẫn nhận lại kết quả tính theo cấu hình cũ, không có lỗi nào báo.
"""
from __future__ import annotations

import time

from app.services import cache


def test_normalize_gop_khoang_trang_nhung_giu_hoa_thuong():
    """Giữ hoa/thường có chủ ý: LLM nhìn thấy đúng chữ user gõ, và hoa/thường ở tên
    riêng có thể đổi output. Gộp lại là mất vài lần hit, không phải trả sai."""
    assert cache.normalize("  cầu   thủ \n bóng đá  ") == "cầu thủ bóng đá"
    assert cache.normalize("Chùa Một Cột") != cache.normalize("chùa một cột")


def test_hit_va_miss():
    c = cache.TTLCache(maxsize=4, ttl_s=60, name="t")
    assert c.get("a") is None
    c.set("a", 1)
    assert c.get("a") == 1
    assert c.stats()["hits"] == 1 and c.stats()["misses"] == 1
    assert c.stats()["hit_rate"] == 0.5


def test_lru_bo_entry_it_dung_nhat():
    c = cache.TTLCache(maxsize=2, ttl_s=60)
    c.set("a", 1)
    c.set("b", 2)
    c.get("a")            # "a" vừa dùng -> "b" thành cũ nhất
    c.set("c", 3)
    assert c.get("a") == 1
    assert c.get("b") is None
    assert c.get("c") == 3


def test_ttl_het_han_thi_coi_nhu_miss_va_xoa_luon():
    c = cache.TTLCache(maxsize=4, ttl_s=0.01)
    c.set("a", 1)
    time.sleep(0.02)
    assert c.get("a") is None
    # Phải xoá hẳn, không để rác chiếm chỗ của entry còn dùng được.
    assert c.stats()["size"] == 0


def test_khoa_embedding_doi_theo_snapshot():
    """Vector phụ thuộc encoder đi kèm snapshot -> khác snapshot phải khác khoá."""
    a = cache.embedding_key("cầu thủ", "1")
    b = cache.embedding_key("cầu thủ", "2")
    assert a != b
    assert a == cache.embedding_key("  cầu   thủ ", "1")   # normalize rồi mới làm khoá


def test_khoa_enrichment_gom_moi_tham_so_doi_duoc_output():
    """Thiếu bất kỳ tham số nào ở đây là sửa config xong vẫn nhận kết quả cũ."""
    base = dict(text="cầu thủ", snapshot_ver="1", model="m", max_tags=5,
                confidence_min=0.75)
    k = cache.enrichment_key(**base)
    assert k != cache.enrichment_key(**{**base, "snapshot_ver": "2"})
    assert k != cache.enrichment_key(**{**base, "model": "khác"})
    assert k != cache.enrichment_key(**{**base, "max_tags": 3})
    assert k != cache.enrichment_key(**{**base, "confidence_min": 0.5})
    assert k == cache.enrichment_key(**{**base, "text": " cầu  thủ "})


def test_search_lan_hai_khong_goi_lai_llm(client, llm):
    """Cùng một query -> lần hai lấy từ cache, không tốn thêm lượt LLM.

    Đây là lý do cache tồn tại: LLM enrich tốn 400-3000ms và trước giờ không cache gì,
    trong khi core search chỉ 12-48ms.
    """
    llm.reply = {"tags": [8], "enriched": "football player celebrating", "confidence": 0.9}
    body = {"text": "cầu thủ bóng đá ăn mừng", "top_k": 3}

    first = client.post("/api/search", json=body).json()
    assert llm.calls == 1

    second = client.post("/api/search", json=body).json()
    assert llm.calls == 1, "lần hai phải lấy enrich từ cache"
    assert second["tags_used"] == first["tags_used"]
    assert second["enrichment"]["encoded_text"] == first["enrichment"]["encoded_text"]
    assert cache.enrichment.stats()["hits"] >= 1
    assert cache.embedding.stats()["hits"] >= 1


def test_enrich_loi_thi_khong_cache(client, llm):
    """Lỗi LLM thường là tạm thời (timeout, rate limit) mà TTL tới 1 giờ. Cache lại là
    khoá cứng trạng thái "không lọc tag" cho cả phiên."""
    llm.raises = RuntimeError("rate limit")
    body = {"text": "cầu thủ bóng đá ăn mừng", "top_k": 3}
    client.post("/api/search", json=body)
    client.post("/api/search", json=body)
    assert llm.calls == 2, "không được cache kết quả lỗi"
    assert cache.enrichment.stats()["size"] == 0


def test_tat_llm_thi_van_cache_embedding(client, llm):
    """use_llm=false không gọi LLM nhưng vẫn encode -> vector phải được cache."""
    body = {"text": "cầu thủ bóng đá ăn mừng", "top_k": 3, "use_llm": False}
    client.post("/api/search", json=body)
    client.post("/api/search", json=body)
    assert llm.calls == 0
    assert cache.embedding.stats()["hits"] >= 1
