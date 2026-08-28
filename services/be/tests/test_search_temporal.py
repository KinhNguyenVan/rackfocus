"""Test POST /api/search/temporal: 2 event, union tag khi bật LLM, chain 2 hit.

Xem docs/superpowers/specs/2026-08-28-temporal-llm-segmentation-design.md. conftest.py's snap
fixture: 200 row, video luân phiên theo i%4, keyframe_time=i*0.2s/row.
"""


def search_temporal(client, **body):
    body.setdefault("event1", "người đàn ông cầm micro")
    body.setdefault("event2", "khán giả vỗ tay")
    r = client.post("/api/search/temporal", json=body)
    assert r.status_code == 200, r.text
    return r.json()


def test_temporal_search_returns_chains_with_two_hits_each(client, llm):
    d = search_temporal(client)
    assert isinstance(d["chains"], list)
    if d["chains"]:
        c = d["chains"][0]
        assert len(c["hits"]) == 2
        assert c["hits"][0]["video_name"] == c["video_name"]
        assert c["span_sec"] >= 0


def test_temporal_use_llm_false_skips_llm(client, llm):
    d = search_temporal(client, use_llm=False)
    assert llm.calls == 0
    assert "warnings" in d


def test_temporal_enrich_mot_lan_tren_hai_event_ghep(client, llm):
    """MỘT lời gọi enrich, và nó phải thấy CẢ HAI event.

    Trước đây enrich riêng từng event rồi hợp tag -- tốn 2 lời gọi LLM mà mỗi lời gọi chỉ
    thấy nửa câu chuyện, trong khi core dù sao cũng chỉ nhận một Filter.
    """
    calls = []

    async def reply(**kwargs):
        calls.append(kwargs["messages"][1]["content"])
        import json
        import types
        msg = types.SimpleNamespace(
            content=json.dumps({"tags": [0], "enriched": ""}))
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])

    import sys
    import types as _types
    sys.modules["litellm"] = _types.SimpleNamespace(acompletion=reply)

    search_temporal(client, use_llm=True,
                    event1="người đàn ông cầm micro", event2="khán giả vỗ tay")
    assert len(calls) == 1
    assert "người đàn ông cầm micro" in calls[0]
    assert "khán giả vỗ tay" in calls[0]


def test_temporal_empty_event_rejected(client):
    assert client.post("/api/search/temporal",
                       json={"event1": "", "event2": "x"}).status_code == 422


# ── tags tường minh ─────────────────────────────────────────────────────────

# KHÔNG khẳng định gì về d["tags_used"] trong ba test dưới. `search_with_fallback` xoá
# tags_used về () mỗi khi tag_fallback nổ (services/core/src/searchcore/search.py:166),
# mà snapshot test chỉ có 200 row / 5 tag với DEFAULT_TOP_K=10 -- fallback nổ hay không
# phụ thuộc vector ngẫu nhiên của FakeEncoder. Khẳng định vào đó là test đỏ vì lý do
# không liên quan. Hợp đồng đang test ở đây là "có tags thì KHÔNG gọi LLM", và
# `llm.calls` đo đúng cái đó.


def test_tags_tuong_minh_thi_bo_qua_llm(client, llm):
    search_temporal(client, use_llm=True, tags=[0])
    assert llm.calls == 0          # tags có sẵn -> không cần LLM chọn tag nữa


def test_tags_rong_nghia_la_khong_loc(client, llm):
    """[] = "user bỏ tick hết, search toàn kho" — KHÁC None, và cũng không gọi LLM.

    Nếu [] bị gộp nhầm với None thì use_llm=True sẽ kéo LLM chạy -> llm.calls == 1.
    """
    search_temporal(client, use_llm=True, tags=[])
    assert llm.calls == 0


def test_tags_vang_mat_giu_nguyen_hanh_vi_cu(client, llm):
    """None = "không qua prepare" -> vẫn quyết định theo use_llm như trước."""
    llm.reply = {"tags": [2], "enriched": "", "confidence": 1.0}
    search_temporal(client, use_llm=True)
    assert llm.calls == 1          # một enrich cho cả hai event đã ghép


# ── /api/search/temporal/prepare ────────────────────────────────────────────


def dual_llm(segment_content, tag_payload=None):
    """Mock litellm phân biệt HAI lời gọi bằng system prompt.

    Prompt tách đoạn là tiếng Anh ("You are a preprocessing assistant..."), prompt chọn
    tag là tiếng Việt ("Bạn giúp chọn LĨNH VỰC..."). Không dùng lại được fixture `llm`
    của conftest vì nó luôn trả JSON object, còn bước tách đoạn cần JSON array.

    Trả về dict đếm để test khẳng định "đúng 2 lời gọi, mỗi bên một cái".
    """
    import json
    import sys
    import types

    calls = {"segment": 0, "tags": 0}

    async def acompletion(**kwargs):
        if "preprocessing assistant" in kwargs["messages"][0]["content"]:
            calls["segment"] += 1
            content = segment_content
        else:
            calls["tags"] += 1
            content = json.dumps({"tags": [], "enriched": "", "confidence": 1.0,
                                  **(tag_payload or {})})
        msg = types.SimpleNamespace(content=content)
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=msg, finish_reason="stop")])

    sys.modules["litellm"] = types.SimpleNamespace(acompletion=acompletion)
    return calls


SEG_TWO = """[
  {"order": 1, "english_clip_query": "A fish placed on a scale."},
  {"order": 2, "english_clip_query": "A person holding a fish."}
]"""


def test_prepare_tra_ve_doan_va_tag(client, llm):
    dual_llm(SEG_TWO, {"tags": [0, 1], "confidence": 0.9})
    r = client.post("/api/search/temporal/prepare",
                    json={"query": "cá được cân, sau đó người cầm đuôi cá"})
    assert r.status_code == 200, r.text
    d = r.json()
    assert [s["order"] for s in d["segments"]] == [1, 2]
    assert d["segments"][0]["english_clip_query"] == "A fish placed on a scale."
    # `>=` chứ không `==`: enrich.decide_tags còn BÙ THÊM tag mà guard regex nhận ra
    # (services/be/src/app/services/enrich.py::decide_tags nhánh 3), nên danh sách cuối
    # có thể rộng hơn tag LLM trả. Cái cần chứng minh là tag của LLM có đi qua được.
    assert set(d["tags"]) >= {0, 1}
    assert d["confidence"] == 0.9
    assert d["warnings"] == []


def test_prepare_goi_dung_hai_llm(client, llm):
    calls = dual_llm(SEG_TWO)
    client.post("/api/search/temporal/prepare", json={"query": "x sau đó y"})
    assert calls == {"segment": 1, "tags": 1}


def test_prepare_tat_loc_tag_thi_chi_goi_segment(client, llm):
    """Tách event và lọc tag là HAI công tắc độc lập trên UI.

    use_llm=False ở đây = "tách event nhưng không lọc" -> phải BỎ HẲN lời gọi enrich, chứ
    không phải gọi rồi vứt kết quả (vẫn tốn tiền + latency của một lời gọi LLM).
    """
    calls = dual_llm(SEG_TWO, {"tags": [0, 1]})
    r = client.post("/api/search/temporal/prepare",
                    json={"query": "x sau đó y", "use_llm": False})
    assert r.status_code == 200, r.text
    assert calls == {"segment": 1, "tags": 0}

    d = r.json()
    assert [s["order"] for s in d["segments"]] == [1, 2]   # tách đoạn vẫn chạy
    assert d["tags"] == []                                 # không lọc
    assert d["tag_names"] == {}                            # không có gì để tick
    assert d["tag_source"] == "disabled"
    assert "llm_failed_tags" not in d["warnings"]          # bỏ qua khác với hỏng


def test_prepare_mac_dinh_van_loc_tag(client, llm):
    """Thiếu `use_llm` -> mặc định True, giữ nguyên hành vi client cũ."""
    calls = dual_llm(SEG_TWO)
    client.post("/api/search/temporal/prepare", json={"query": "x sau đó y"})
    assert calls == {"segment": 1, "tags": 1}


# ── cache ───────────────────────────────────────────────────────────────────


def test_bat_loc_sau_khi_da_phan_tich_chi_ton_them_enrich(client, llm):
    """Kịch bản chính của hai cache riêng biệt.

    User Phân tích lúc TẮT lọc (chỉ segment chạy), rồi đổi ý bật lọc lên và Phân tích
    lại. Lần hai chỉ được phép tốn lời gọi ENRICH — segment phải lấy từ cache, vì nó là
    lời gọi đắt hơn (`segment_prompt.txt` dài hơn hẳn prompt enrich).
    """
    calls = dual_llm(SEG_TWO, {"tags": [0, 1]})
    q = {"query": "cá được cân rồi người cầm đuôi cá"}

    client.post("/api/search/temporal/prepare", json={**q, "use_llm": False})
    assert calls == {"segment": 1, "tags": 0}

    d = client.post("/api/search/temporal/prepare", json={**q, "use_llm": True}).json()
    assert calls == {"segment": 1, "tags": 1}   # segment KHÔNG chạy lại
    assert set(d["tags"]) >= {0, 1}             # mà vẫn có tag
    assert [s["order"] for s in d["segments"]] == [1, 2]   # đoạn vẫn trả về đủ


def test_gat_tat_roi_bat_lai_loc_khong_goi_lai_llm(client, llm):
    """Gạt tắt/bật công tắc lọc nhiều lần chỉ tốn ĐÚNG một lời gọi enrich.

    Lần bật thứ hai phải lấy lại đúng tập tag cũ từ cache chứ không hỏi LLM lại.
    """
    llm.reply = {"tags": [2], "enriched": "", "confidence": 1.0}

    search_temporal(client, use_llm=True)          # bật -> enrich
    assert llm.calls == 1
    search_temporal(client, use_llm=False)         # tắt -> không đụng LLM
    assert llm.calls == 1
    d = search_temporal(client, use_llm=True)      # bật lại -> cache hit
    assert llm.calls == 1
    assert d  # vẫn trả về hợp lệ


def test_sua_mot_event_van_dung_lai_vector_event_kia(client, llm):
    """Cache embedding khoá theo TỪNG event, không theo chuỗi ghép.

    Sửa sự kiện 2 rồi tìm lại thì vector của sự kiện 1 phải được dùng lại — encode tốn
    170-420ms/lần nên đây là chỗ ăn tiền nhất ở luồng nhập tay.
    """
    from app.services import cache

    search_temporal(client, use_llm=False, event1="người cầm micro", event2="khán giả vỗ tay")
    sau_lan_dau = cache.embedding.stats()["misses"]

    search_temporal(client, use_llm=False, event1="người cầm micro", event2="khán giả đứng dậy")
    # Chỉ event2 là mới -> đúng 1 miss nữa, event1 lấy từ cache.
    assert cache.embedding.stats()["misses"] == sau_lan_dau + 1


def test_prepare_tra_ve_ca_vocab_de_user_tick_them(client, llm):
    """tag_names phải là CẢ vocab, không chỉ tag đã chọn — UI cần tick THÊM vào."""
    dual_llm(SEG_TWO, {"tags": [0]})
    d = client.post("/api/search/temporal/prepare", json={"query": "x sau đó y"}).json()
    # conftest VOCAB có 5 tag; khoá JSON luôn là chuỗi.
    assert len(d["tag_names"]) == 5
    assert set(d["tag_names"]) == {"0", "1", "2", "3", "4"}


def test_prepare_loi_tach_doan_lui_ve_cau_goc(client, llm):
    """Hỏng -> 1 đoạn = câu gốc, và BÁO ra bằng warning chứ không bằng nhãn trong dữ liệu."""
    dual_llm("xin lỗi, tôi không hiểu")
    d = client.post("/api/search/temporal/prepare",
                    json={"query": "câu gốc"}).json()
    assert len(d["segments"]) == 1
    assert d["segments"][0]["english_clip_query"] == "câu gốc"
    assert "llm_failed_segment" in d["warnings"]


def test_prepare_mot_doan_hop_le_khong_bao_loi(client, llm):
    """N=1 do LLM trả ĐÚNG (câu không có mốc thời gian) khác N=1 do hỏng.

    Cả hai rẽ vào cùng một nhánh UI (mời chạy KIS), nhưng chỉ nhánh hỏng mới hiện cảnh
    báo. Không có `warnings` để phân biệt thì UI phải đoán bằng cách so chuỗi.
    """
    dual_llm('[{"order": 1, "english_clip_query": "A group of people exercising."}]')
    d = client.post("/api/search/temporal/prepare",
                    json={"query": "nhóm người tập thể dục"}).json()
    assert len(d["segments"]) == 1
    assert "llm_failed_segment" not in d["warnings"]


def test_prepare_query_rong_bi_tu_choi(client):
    assert client.post("/api/search/temporal/prepare",
                       json={"query": ""}).status_code == 422
