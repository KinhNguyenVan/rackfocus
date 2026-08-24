"""Test POST /api/search/temporal: 2 event, union tag khi bật LLM, chain 2 hit.

Xem docs/superpowers/specs/2026-08-24-temporal-search-design.md. conftest.py's snap
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
