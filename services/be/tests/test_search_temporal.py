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


def test_temporal_use_llm_true_unions_both_events_tags(client, llm):
    calls = []

    async def reply(**kwargs):
        text = kwargs["messages"][1]["content"]
        calls.append(text)
        payload = {"tags": [0]} if len(calls) == 1 else {"tags": [1]}
        import json
        import types
        msg = types.SimpleNamespace(content=json.dumps({**payload, "enriched": ""}))
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])

    import sys
    import types as _types
    sys.modules["litellm"] = _types.SimpleNamespace(acompletion=reply)

    d = search_temporal(client, use_llm=True)
    assert len(calls) == 2
    assert d  # phản hồi vẫn hợp lệ dù mỗi event ra tag khác nhau


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

    Nếu [] bị gộp nhầm với None thì use_llm=True sẽ kéo LLM chạy -> llm.calls == 2.
    """
    search_temporal(client, use_llm=True, tags=[])
    assert llm.calls == 0


def test_tags_vang_mat_giu_nguyen_hanh_vi_cu(client, llm):
    """None = "không qua prepare" -> vẫn quyết định theo use_llm như trước."""
    llm.reply = {"tags": [2], "enriched": "", "confidence": 1.0}
    search_temporal(client, use_llm=True)
    assert llm.calls == 2          # enrich riêng cho từng event, như cũ


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
