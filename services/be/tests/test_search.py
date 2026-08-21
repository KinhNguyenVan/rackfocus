"""Test POST /api/search: LLM chọn tag, encode song song, đường lùi khi LLM sai.

Xem docs/search-design.md §3, §4. Điểm quan trọng nhất được kiểm ở đây: **lọc tag là
CỨNG** — frame mang tag không được chọn là không thể với tới. Nên BE phải (a) phơi ra
`tags_used`/`candidate_count`, (b) có đường tắt LLM, (c) không bao giờ chết vì LLM.
"""

from collections import Counter

from conftest import NTAG, N


def search(client, **body):
    body.setdefault("text", "cầu thủ bóng đá ăn mừng")
    r = client.post("/api/search", json=body)
    assert r.status_code == 200, r.text
    return r.json()


# --------------------------- readiness ---------------------------
def test_readyz_reports_core_state(client):
    d = client.get("/readyz").json()
    assert d["ready"] is True
    assert d["snapshot_ver"] == "9"
    assert d["point_count"] == N


def test_tags_endpoint_exposes_vocab(client):
    d = client.get("/api/tags").json()
    assert d["count"] == NTAG
    assert d["tags"][0]["description"] == "cảnh bóng đá trên sân"


# --------------------------- đường chính ---------------------------
def test_llm_tags_narrow_the_search(client, llm):
    llm.reply = {"tags": [0], "enriched": "football celebration"}
    d = search(client, top_k=5)

    assert d["tags_used"] == [0]
    assert d["candidate_count"] == N // NTAG
    assert d["corpus_count"] == N // NTAG
    assert d["strategy"] == "exact_subset"
    assert d["enrichment"]["tags"] == [0]
    assert llm.calls == 1


def test_hits_carry_fields_needed_to_submit_and_seek(client, llm):
    """`video_name` + `frame` là thứ phải nộp; `keyframe_time` là thứ để seek đúng chỗ.

    `start_sec` là thời gian của SCENE (nhiều shot) nên seek theo nó sẽ lệch.
    """
    llm.reply = {"tags": [], "enriched": ""}
    hit = search(client, top_k=3)["hits"][0]

    assert hit["video_name"].startswith("L26_V")
    assert isinstance(hit["frame"], int)
    assert hit["keyframe_url"].startswith("https://")
    assert hit["clip_url"].startswith("https://")
    assert hit["keyframe_time"] >= 0.0
    assert hit["point_id"] > 0


def test_llm_and_encode_run_in_parallel(client, llm):
    """Tổng phải là max(LLM, encode), không phải tổng cộng — tiết kiệm 80-300ms/query."""
    llm.reply = {"tags": [], "enriched": ""}
    t = search(client)["timings_ms"]
    assert t["llm_and_encode_parallel"] <= t["llm"] + t["core_encode"] + 50


# --------------------------- đường lùi ---------------------------
def test_use_llm_false_skips_llm_and_searches_everything(client, llm):
    """Đường lùi duy nhất của user khi LLM chọn sai tag."""
    d = search(client, use_llm=False)
    assert llm.calls == 0
    assert d["tags_used"] == []
    assert d["candidate_count"] == N


def test_explicit_tags_bypass_llm(client, llm):
    d = search(client, tags=[2, 4])
    assert llm.calls == 0
    assert d["tags_used"] == [2, 4]
    assert d["candidate_count"] == 2 * (N // NTAG)


def test_llm_failure_falls_back_to_full_corpus(client, llm):
    """LLM chết KHÔNG được làm chết search: thà chậm mà đúng, còn hơn không trả gì."""
    llm.raises = RuntimeError("groq 503")
    d = search(client)

    assert "llm_failed" in d["warnings"]
    assert d["tags_used"] == []
    assert d["candidate_count"] == N
    assert d["hits"], "phải vẫn có kết quả"
    assert "groq 503" in d["enrichment"]["error"]


def test_llm_returning_garbage_falls_back(client, llm):
    """Output không phải JSON -> coi như không chọn được tag, không phải 500."""
    llm.reply = None
    llm.raises = None
    import json as _json
    import sys
    import types

    async def bad(**kwargs):
        llm.calls += 1
        msg = types.SimpleNamespace(content="tôi nghĩ là tag 0 và 3")
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])

    sys.modules["litellm"] = types.SimpleNamespace(acompletion=bad)
    d = search(client)
    assert d["tags_used"] == []
    assert d["candidate_count"] == N
    assert _json  # giữ import cho rõ ý: parse JSON là chỗ hỏng


def test_llm_hallucinated_tag_is_dropped_not_sent(client, llm):
    """Tag không tồn tại bị lọc ở BE -> query vẫn chạy với các tag hợp lệ còn lại,
    thay vì để core trả 400 và người dùng mất trắng lượt tìm."""
    llm.reply = {"tags": [1, 9999], "enriched": ""}
    d = search(client)
    assert d["tags_used"] == [1]


# --------------------------- lỗi ---------------------------
def test_explicit_unknown_tag_returns_400(client):
    """Tag do CALLER chỉ định thẳng thì không đoán ý — trả 400 kèm lý do."""
    r = client.post("/api/search", json={"text": "x", "tags": [9999]})
    assert r.status_code == 400
    assert "9999" in r.json()["detail"]


def test_empty_text_rejected(client):
    assert client.post("/api/search", json={"text": ""}).status_code == 422


# --------------------------- diversity ---------------------------
def test_diversity_caps_hits_per_video(client, llm, monkeypatch):
    llm.reply = {"tags": [], "enriched": ""}
    d = search(client, use_llm=False, top_k=20)
    names = Counter(h["video_name"] for h in d["hits"])
    # DIVERSITY_MAX_PER_VIDEO=0 trong conftest -> không giới hạn; chỉ kiểm là có nhiều video
    assert len(names) > 1


def test_timings_exposed_for_debugging(client, llm):
    llm.reply = {"tags": [0], "enriched": ""}
    t = search(client)["timings_ms"]
    for key in ("llm", "core_encode", "core_filter", "core_rerank", "core_total", "total"):
        assert key in t
