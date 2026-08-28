# Tách sự kiện bằng LLM cho Temporal Search — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Người dùng gõ một câu tiếng Việt, LLM tách thành N mô tả tiếng Anh cho CLIP, in ra UI cho sửa và chọn 2 sự kiện theo thứ tự, rồi mới search temporal.

**Architecture:** Thêm một service BE (`segment.py`) chạy `prompt.yaml` qua litellm, và một endpoint `POST /api/search/temporal/prepare` gọi nó **song song** với `enrich.py` (`asyncio.gather`, tổng = max chứ không phải cộng). Endpoint search sẵn có nhận thêm `tags` tường minh để bỏ qua LLM. FE thêm một component chọn/sửa đoạn. **Không đụng `proto/` và không đụng logic `services/core`.**

**Tech Stack:** Python 3.11 / FastAPI / pydantic / litellm / pytest (BE); React 18 + TypeScript + Vite + Bootstrap 5 (FE).

**Spec:** [docs/superpowers/specs/2026-08-28-temporal-llm-segmentation-design.md](../specs/2026-08-28-temporal-llm-segmentation-design.md)

## Global Constraints

- Code, comment và docstring trong `services/be` + `services/core` viết **tiếng Việt** (quy ước repo). Định danh code giữ tiếng Anh.
- **Không sửa `proto/`.** Không cần chạy `make proto` cho plan này. `SearchTemporal` vẫn nhận đúng 2 vector.
- `tags: list[int] | None` — `[]` và `None` **không được** gộp. `[]` = "user bỏ tick hết, search toàn kho"; `None` = "không qua prepare, quyết định theo `use_llm`".
- Service tầng `services/` **không bao giờ raise** ra ngoài: lỗi LLM trả dataclass có `error`, không ném exception (theo `enrich.py`).
- Bốn test temporal hiện có trong `services/be/tests/test_search_temporal.py` phải **pass nguyên trạng, không sửa nội dung test** — bằng chứng luồng nhập tay không bị đụng.
- Chạy test BE: `pytest services/be/tests -q` (từ gốc repo). Cần stub proto trước: `bash scripts/gen_proto.sh`.
- Gate FE: `cd services/fe && npm run build` (chạy `tsc --noEmit` trước khi build). **FE không có test runner** — không thêm vitest trong plan này (ngoài phạm vi spec); logic chọn đoạn tách thành hàm thuần để kiểm bằng mắt và kiểm được về sau.
- Dùng lại `settings.llm_*` sẵn có. **Không thêm biến môi trường mới.**

## File Structure

| File | Trách nhiệm | Task |
|---|---|---|
| `services/be/src/app/services/segment_prompt.txt` | **Tạo** — `prompt.yaml` chuyển vào đây, ship trong container BE | 1 |
| `services/be/src/app/services/segment.py` | **Tạo** — gọi LLM tách đoạn, parse JSON array, đường lùi 1 đoạn = câu gốc | 1 |
| `services/be/tests/test_segment.py` | **Tạo** — unit test thuần, không cần core gRPC | 1 |
| `run_prompt.py` | **Sửa** — trỏ sang đường dẫn prompt mới | 1 |
| `services/be/src/app/api/search_temporal.py` | **Sửa** — thêm endpoint `prepare`; thêm `tags` vào request search | 2, 3 |
| `services/be/tests/test_search_temporal.py` | **Sửa** — thêm test mới, giữ nguyên 4 test cũ | 2, 3 |
| `services/fe/src/api/types.ts` | **Sửa** — `TemporalSegment`, `TemporalPrepareResponse`, `tags` | 4 |
| `services/fe/src/api/client.ts` | **Sửa** — `prepareTemporal()` | 4 |
| `services/fe/src/components/TemporalPrepare.tsx` | **Tạo** — ô query, danh sách đoạn sửa được, chọn 2, tick tag | 5 |
| `services/fe/src/hooks/useTemporalSearch.ts` | **Sửa** — nhận thêm `tags` | 6 |
| `services/fe/src/App.tsx` | **Sửa** — bỏ ép tắt LLM, rẽ nhánh hai luồng | 6 |

`TemporalQueryBuilder.tsx` **không đổi một dòng nào** — nó là luồng nhập tay khi tắt LLM.

---

### Task 1: Service tách đoạn (`segment.py`)

Thuần logic + một lời gọi LLM. Không HTTP, không core gRPC — test chạy trong mili giây.

**Files:**
- Create: `services/be/src/app/services/segment_prompt.txt`
- Create: `services/be/src/app/services/segment.py`
- Create: `services/be/tests/test_segment.py`
- Modify: `run_prompt.py:19`
- Delete: `prompt.yaml` (chuyển chỗ, không phải xoá nội dung)

**Interfaces:**
- Consumes: `settings` có các thuộc tính `llm_enabled`, `llm_model`, `llm_api_key`, `llm_temperature`, `llm_timeout_s`, `llm_max_tokens`, `llm_reasoning_effort` (từ `app.config.Settings`).
- Produces:
  - `Segment(order: int, english_clip_query: str)` — dataclass. **Không có `label`**: prompt chốt mỗi đoạn đúng hai trường và cấm mang nhãn nguồn (`E1`, `Sự kiện 1`) sang output.
  - `Segmentation(segments: list[Segment], model: str, latency_ms: float, error: str)` — dataclass, có `.ok` property
  - `async def segment(query: str, settings) -> Segmentation` — không bao giờ raise

- [ ] **Step 1: Chuyển prompt vào package BE**

`prompt.yaml` hiện chưa được git theo dõi (untracked), nên đây là move file thường, không phải `git mv`:

```bash
mv prompt.yaml services/be/src/app/services/segment_prompt.txt
```

Nội dung **không sửa một chữ nào**. Đổi đuôi `.yaml` → `.txt` vì file thực chất là văn bản thuần, không phải YAML (nó chưa từng được parse bằng YAML — `run_prompt.py` đọc bằng `read_text()`).

- [ ] **Step 2: Trỏ `run_prompt.py` sang đường dẫn mới**

Sửa `run_prompt.py` dòng 19, từ:

```python
system_prompt = (REPO_ROOT / "prompt.yaml").read_text(encoding="utf-8")
```

thành:

```python
# Prompt sống trong package BE (nó phải ship trong container). Harness này đọc CHÍNH file
# đó, không phải bản sao — hai bản trôi lệch nhau là cách âm thầm nhất để "chạy tay thì
# đúng, chạy trong BE thì sai".
system_prompt = (REPO_ROOT / "services" / "be" / "src" / "app" / "services"
                 / "segment_prompt.txt").read_text(encoding="utf-8")
```

- [ ] **Step 3: Viết test thất bại**

Tạo `services/be/tests/test_segment.py`:

```python
"""Unit test services/segment.py — chỉ mock litellm, không cần core gRPC.

Khác test_search_temporal.py: ở đây gọi thẳng hàm với settings giả, nên không dùng
fixture `client` (vốn dựng cả một core gRPC thật qua Unix socket).
"""
import json
import sys
import types

import pytest

from app.services import segment as seg


@pytest.fixture(autouse=True)
def _clean_litellm():
    yield
    sys.modules.pop("litellm", None)


def fake_settings(**over):
    base = dict(llm_enabled=True, llm_model="fake/model", llm_api_key="",
                llm_temperature=0.0, llm_timeout_s=6.0, llm_max_tokens=2000,
                llm_reasoning_effort="")
    base.update(over)
    return types.SimpleNamespace(**base)


def mock_litellm(content=None, raises=None):
    """Đặt litellm giả trả đúng `content`. `raises` để giả lập lỗi mạng/timeout."""
    calls = []

    async def acompletion(**kwargs):
        calls.append(kwargs)
        if raises:
            raise raises
        msg = types.SimpleNamespace(content=content)
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=msg, finish_reason="stop")])

    sys.modules["litellm"] = types.SimpleNamespace(acompletion=acompletion)
    return calls


@pytest.mark.asyncio
async def test_tach_nhieu_doan():
    mock_litellm(json.dumps([
        {"order": 1, "english_clip_query": "A fish placed on a scale."},
        {"order": 2, "english_clip_query": "A person holding a fish."},
    ]))
    r = await seg.segment("con cá được cân, sau đó người cầm đuôi cá", fake_settings())
    assert r.ok
    assert [s.english_clip_query for s in r.segments] == [
        "A fish placed on a scale.", "A person holding a fish."]


@pytest.mark.asyncio
async def test_bo_qua_nhan_nguon_neu_llm_van_tra():
    """Prompt cấm mang nhãn nguồn sang output, nhưng LLM lỡ trả thì cũng không sao.

    Segment không có chỗ chứa `label` -> khoá thừa bị bỏ im lặng, không được raise.
    """
    mock_litellm(json.dumps([
        {"order": 1, "label": "E1", "english_clip_query": "Golden dragons spinning."},
    ]))
    r = await seg.segment("múa lân", fake_settings())
    assert r.ok
    assert r.segments[0].english_clip_query == "Golden dragons spinning."


@pytest.mark.asyncio
async def test_order_danh_lai_theo_vi_tri():
    """LLM đánh trùng/nhảy số được; UI dùng order làm khoá nên phải đánh lại."""
    mock_litellm(json.dumps([
        {"order": 5, "english_clip_query": "Golden dragons spinning."},
        {"order": 5, "english_clip_query": "A mallet striking a gong."},
    ]))
    r = await seg.segment("múa lân", fake_settings())
    assert [s.order for s in r.segments] == [1, 2]


@pytest.mark.asyncio
async def test_bo_doan_rong():
    mock_litellm(json.dumps([
        {"order": 1, "english_clip_query": "   "},
        {"order": 2, "english_clip_query": "A mallet striking a gong."},
    ]))
    r = await seg.segment("múa lân", fake_settings())
    assert len(r.segments) == 1
    assert r.segments[0].order == 1


@pytest.mark.asyncio
async def test_json_rac_thi_lui_ve_cau_goc():
    mock_litellm("xin lỗi, tôi không hiểu")
    r = await seg.segment("câu gốc", fake_settings())
    assert not r.ok
    assert [s.english_clip_query for s in r.segments] == ["câu gốc"]


@pytest.mark.asyncio
async def test_llm_loi_thi_lui_ve_cau_goc():
    mock_litellm(raises=TimeoutError("quá thời gian"))
    r = await seg.segment("câu gốc", fake_settings())
    assert not r.ok
    assert [s.english_clip_query for s in r.segments] == ["câu gốc"]


@pytest.mark.asyncio
async def test_tat_llm_thi_khong_goi_mang():
    calls = mock_litellm(json.dumps([{"order": 1, "english_clip_query": "x"}]))
    r = await seg.segment("câu gốc", fake_settings(llm_enabled=False))
    assert calls == []
    assert r.ok
    assert r.segments[0].english_clip_query == "câu gốc"
```

- [ ] **Step 4: Chạy test, xác nhận nó đỏ**

```bash
pytest services/be/tests/test_segment.py -q
```

Expected: FAIL — `ModuleNotFoundError: No module named 'app.services.segment'` (collection error, 0 test chạy).

Nếu thay vào đó lỗi là `'async def' functions are not natively supported`, nghĩa là thiếu `pytest-asyncio`. Kiểm tra:

```bash
grep -rn "asyncio" services/be/requirements.txt services/be/tests/*.py | head
```

Nếu repo chưa có `pytest-asyncio`, bỏ hết `@pytest.mark.asyncio` và gọi qua `asyncio.run(...)` trong test đồng bộ, ví dụ `r = asyncio.run(seg.segment("câu gốc", fake_settings()))`. **Không** thêm dependency mới chỉ để chạy test — CI cài deps bằng tay theo ma trận trong `ci.yml`, thêm gói là phải sửa cả đó.

- [ ] **Step 5: Viết implementation tối thiểu**

Tạo `services/be/src/app/services/segment.py`:

```python
"""Tách một câu truy vấn tiếng Việt thành N mô tả tiếng Anh cho CLIP.

Prompt nằm ở `segment_prompt.txt` cạnh file này (trước là `prompt.yaml` ở gốc repo).
Chạy SONG SONG với `enrich.py` trong `api/search_temporal.py::prepare` — hai lời gọi LLM
độc lập trên cùng câu gốc, tổng = max chứ không phải cộng.

Theo đúng hợp đồng của `enrich.py`: KHÔNG BAO GIỜ raise. Lỗi/timeout/JSON rác -> trả đúng
MỘT đoạn = câu gốc nguyên văn, kèm `error`. Hỏng kiểu đó rơi vào nhánh N=1 mà UI đã phải
xử lý sẵn (không tạo được chuỗi, mời chạy KIS), nên không cần đường lỗi riêng.
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger("app.segment")

# Đọc một lần lúc import. Prompt phải nằm trong package để ship được trong container BE.
_SYSTEM = Path(__file__).with_name("segment_prompt.txt").read_text(encoding="utf-8")


@dataclass
class Segment:
    order: int
    english_clip_query: str
    # KHÔNG có `label`. Prompt chốt mỗi đoạn đúng hai trường và nói rõ nhãn nguồn (E1,
    # "Sự kiện 1", ...) chỉ để LLM biết có bao nhiêu đoạn — không mang sang output.


@dataclass
class Segmentation:
    segments: list[Segment] = field(default_factory=list)
    model: str = ""
    latency_ms: float = 0.0
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error


def _fallback(query: str) -> list[Segment]:
    """Một đoạn duy nhất = câu gốc nguyên văn.

    Không đánh dấu gì đặc biệt vào dữ liệu: phân biệt "hỏng" với "đúng là chỉ có 1 đoạn"
    bằng `Segmentation.error` (rồi thành warning `llm_failed_segment`), không bằng một
    nhãn ma thuật mà UI phải so chuỗi để đoán ra.
    """
    return [Segment(order=1, english_clip_query=query)]


def _parse(content: str) -> list[Segment]:
    """Bóc JSON ARRAY khỏi output LLM.

    Khác `enrich._parse` bóc object (`\\{.*\\}`): prompt này bắt trả về một mảng.
    """
    m = re.search(r"\[.*\]", content, re.DOTALL)
    if not m:
        raise ValueError(f"không thấy JSON array trong output: {content[:120]!r}")
    data = json.loads(m.group(0))
    if not isinstance(data, list):
        raise TypeError(f"phải là list, nhận {type(data).__name__}")

    out: list[Segment] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        text = str(item.get("english_clip_query") or "").strip()
        if not text:
            continue
        # `order` đánh lại theo vị trí SAU khi lọc, không tin số LLM trả: nó đánh trùng
        # hoặc nhảy số được, mà UI dùng order làm khoá React lẫn thứ tự hiển thị.
        # Khoá thừa (`label` chẳng hạn, nếu LLM lỡ trả dù prompt đã cấm) bị bỏ im lặng.
        out.append(Segment(order=len(out) + 1, english_clip_query=text))
    if not out:
        raise ValueError("không còn đoạn nào sau khi lọc")
    return out


async def segment(query: str, settings) -> Segmentation:
    """Không bao giờ raise. Lỗi -> Segmentation(1 đoạn = câu gốc, error=...)."""
    if not settings.llm_enabled:
        return Segmentation(segments=_fallback(query))

    t0 = time.perf_counter()
    try:
        import litellm

        # reasoning_effort chỉ gửi khi có khai: model không phải reasoning sẽ lỗi nếu nhận.
        extra = ({"reasoning_effort": settings.llm_reasoning_effort}
                 if settings.llm_reasoning_effort else {})
        resp = await litellm.acompletion(
            model=settings.llm_model,
            messages=[{"role": "system", "content": _SYSTEM},
                      {"role": "user", "content": query}],
            api_key=settings.llm_api_key or None,
            temperature=settings.llm_temperature,
            timeout=settings.llm_timeout_s,
            max_tokens=settings.llm_max_tokens,
            **extra,
        )
        choice = resp.choices[0]
        content = choice.message.content or ""
        if not content.strip():
            raise ValueError(
                f"LLM trả content rỗng (finish_reason={choice.finish_reason}) — "
                "nếu là 'length' thì tăng LLM_MAX_TOKENS")
        return Segmentation(segments=_parse(content), model=settings.llm_model,
                            latency_ms=(time.perf_counter() - t0) * 1000)
    except Exception as ex:  # noqa: BLE001 — LLM lỗi KHÔNG được làm chết prepare
        log.warning("tách đoạn lỗi (%s: %s) -> lùi về 1 đoạn = câu gốc",
                    type(ex).__name__, ex)
        return Segmentation(segments=_fallback(query), model=settings.llm_model,
                            latency_ms=(time.perf_counter() - t0) * 1000,
                            error=f"{type(ex).__name__}: {ex}")
```

- [ ] **Step 6: Chạy test, xác nhận nó xanh**

```bash
pytest services/be/tests/test_segment.py -q
```

Expected: PASS, 7 passed.

- [ ] **Step 7: Xác nhận harness chạy tay vẫn dùng được**

```bash
python -c "from pathlib import Path; p = Path('services/be/src/app/services/segment_prompt.txt'); assert p.exists() and 'preprocessing assistant' in p.read_text(encoding='utf-8'); print('prompt ok', len(p.read_text(encoding='utf-8')), 'ký tự')"
```

Expected: in ra `prompt ok <số> ký tự`. (Không chạy `run_prompt.py` thật — nó gọi mạng và cần API key.)

- [ ] **Step 8: Commit**

```bash
git add services/be/src/app/services/segment_prompt.txt \
        services/be/src/app/services/segment.py \
        services/be/tests/test_segment.py \
        run_prompt.py
git rm --cached prompt.yaml 2>/dev/null || true
git commit -m "feat(be): service tách câu truy vấn thành N đoạn tiếng Anh cho CLIP"
```

`prompt.yaml` chưa từng được track nên `git rm --cached` sẽ báo lỗi và bị `|| true` nuốt — đó là chủ ý, để lệnh chạy được cả hai trường hợp.

---

### Task 2: Endpoint `POST /api/search/temporal/prepare`

**Files:**
- Modify: `services/be/src/app/api/search_temporal.py`
- Modify: `services/be/tests/test_search_temporal.py`

**Interfaces:**
- Consumes: `segment.segment(query, settings) -> Segmentation` (Task 1); `enrich.enrich(query, vocab, settings) -> Enrichment` với các field `tags: list[int]`, `confidence: float`, `tag_source: str`, `latency_ms: float`, `error: str`; `tagvocab.get(settings) -> tuple[dict[int, TagInfo], str]` với `TagInfo.name` / `TagInfo.description`.
- Produces: `POST /api/search/temporal/prepare` nhận `{"query": str}`, trả JSON có khoá `segments`, `tags`, `tag_names`, `confidence`, `tag_source`, `warnings`, `snapshot_ver`, `timings_ms`. Task 4 dựng type FE khớp đúng hình dạng này.

- [ ] **Step 1: Viết test thất bại**

Thêm vào cuối `services/be/tests/test_search_temporal.py` (giữ nguyên 4 test đang có):

```python
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
```

- [ ] **Step 2: Chạy test, xác nhận nó đỏ**

```bash
pytest services/be/tests/test_search_temporal.py -q -k prepare
```

Expected: FAIL — 5 test đầu trả `404 Not Found` (assert `status_code == 200` đỏ, hoặc `KeyError` khi đọc khoá của response 404), test cuối cũng đỏ vì 404 ≠ 422.

- [ ] **Step 3: Viết implementation tối thiểu**

Trong `services/be/src/app/api/search_temporal.py`, thêm `segment` vào import:

```python
from ..services import enrich as enrich_svc
from ..services import segment as segment_svc
from ..services import tagvocab
```

Thêm các model và endpoint (đặt **trên** `search_temporal`, sau `TemporalSearchResponse`):

```python
class PrepareRequest(BaseModel):
    query: str = Field(min_length=1)


class SegmentOut(BaseModel):
    order: int
    english_clip_query: str


class PrepareResponse(BaseModel):
    segments: list[SegmentOut]
    tags: list[int]
    # CẢ vocab, không chỉ tag đã chọn: UI phải hiện tag chưa chọn để user tick THÊM vào,
    # không chỉ bỏ bớt. Khoá JSON luôn là chuỗi -> tới FE thành {"3": "..."}.
    tag_names: dict[int, str]
    confidence: float
    tag_source: str
    warnings: list[str]
    snapshot_ver: str
    timings_ms: dict[str, float]


@router.post("/search/temporal/prepare", response_model=PrepareResponse)
async def prepare(req: PrepareRequest) -> PrepareResponse:
    """Bước 1 của temporal: LLM tách đoạn + LLM chọn tag, CHẠY SONG SONG.

    Không nằm trên hot path — sau bước này người dùng còn phải sửa câu và bấm chọn 2 sự
    kiện, nên ngân sách 100-200ms không áp ở đây. Bước tách đoạn nhiều khả năng là bên
    chậm hơn: `segment_prompt.txt` dài hơn hẳn prompt trong `enrich.py`.

    Không có try/except: cả `segment()` lẫn `enrich()` đều cam kết không raise.
    """
    st = get_settings()
    t_all = time.perf_counter()

    vocab, snap_ver = await tagvocab.get(st)

    segmentation, enrichment = await asyncio.gather(
        segment_svc.segment(req.query, st),
        enrich_svc.enrich(req.query, vocab, st),
    )

    warnings: list[str] = []
    if segmentation.error:
        warnings.append("llm_failed_segment")
    if enrichment.error:
        warnings.append("llm_failed_tags")

    return PrepareResponse(
        segments=[SegmentOut(order=s.order, english_clip_query=s.english_clip_query)
                  for s in segmentation.segments],
        tags=enrichment.tags,
        tag_names={tid: (info.name or info.description)
                   for tid, info in vocab.items()},
        confidence=enrichment.confidence,
        tag_source=enrichment.tag_source,
        warnings=warnings,
        snapshot_ver=snap_ver,
        timings_ms={
            "segment": round(segmentation.latency_ms, 2),
            "enrich": round(enrichment.latency_ms, 2),
            "total": round((time.perf_counter() - t_all) * 1000, 2),
        },
    )
```

`asyncio` và `time` đã được import sẵn ở đầu file — không thêm import mới cho chúng.

- [ ] **Step 4: Chạy test, xác nhận nó xanh**

```bash
pytest services/be/tests/test_search_temporal.py -q
```

Expected: PASS, 10 passed (4 test cũ + 6 test mới).

- [ ] **Step 5: Commit**

```bash
git add services/be/src/app/api/search_temporal.py services/be/tests/test_search_temporal.py
git commit -m "feat(be): endpoint prepare — tách đoạn và chọn tag chạy song song"
```

---

### Task 3: Nhận `tags` tường minh ở `/api/search/temporal`

Mirror đúng cách `api/search.py` đã làm (`search.py:87` và `search.py:113`) — không phát minh cách mới.

**Files:**
- Modify: `services/be/src/app/api/search_temporal.py`
- Modify: `services/be/tests/test_search_temporal.py`
- Modify: `services/core/src/searchcore/temporal.py:3`, `services/core/src/searchcore/config.py:83`, `services/core/tests/test_temporal.py:4`, `services/be/src/app/api/search_temporal.py:3`, `services/be/tests/test_search_temporal.py:3` (trỏ lại tham chiếu spec chết)

**Interfaces:**
- Consumes: `PrepareResponse.tags` từ Task 2 (client gửi trả lại sau khi user tick).
- Produces: `POST /api/search/temporal` nhận thêm khoá tuỳ chọn `tags: list[int] | null`. Task 4/6 dựa vào đúng ngữ nghĩa `[]` ≠ `null` này.

- [ ] **Step 1: Viết test thất bại**

Thêm vào `services/be/tests/test_search_temporal.py`:

```python
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
```

- [ ] **Step 2: Chạy test, xác nhận nó đỏ**

```bash
pytest services/be/tests/test_search_temporal.py -q -k tags
```

Expected: FAIL — `test_tags_tuong_minh_thi_bo_qua_llm` và `test_tags_rong_nghia_la_khong_loc` đỏ ở `assert llm.calls == 0` (nhận `2`), vì `tags` hiện bị pydantic bỏ qua và LLM vẫn chạy.

- [ ] **Step 3: Viết implementation tối thiểu**

Trong `services/be/src/app/api/search_temporal.py`, thêm field vào `TemporalSearchRequest` (ngay dưới `use_llm`):

```python
    # Chỉ định tag thẳng, bỏ qua LLM — client gửi lại tag lấy từ /search/temporal/prepare
    # sau khi user tick/bỏ tick. Mirror `SearchRequest.tags` (schemas/search.py).
    #
    # [] và None KHÁC nhau và không được gộp: [] = "user đã bỏ tick hết, search toàn kho",
    # None = "không qua prepare, quyết định theo use_llm". Gộp lại là âm thầm bật lại lọc
    # tag mà user vừa cố ý tắt — mà lọc tag là CỨNG, không cứu được ở tầng nào khác.
    tags: list[int] | None = None
```

Trong hàm `search_temporal`, thay:

```python
    async def _enrich(text: str):
        if not req.use_llm:
            return enrich_svc.Enrichment(enriched_text=text)
        return await enrich_svc.enrich(text, vocab, st)
```

bằng:

```python
    # Tính một lần rồi dùng lại, giống api/search.py:87.
    used_llm = req.tags is None and req.use_llm

    async def _enrich(text: str):
        if not used_llm:
            return enrich_svc.Enrichment(enriched_text=text)
        return await enrich_svc.enrich(text, vocab, st)
```

và thay:

```python
    tags = sorted(set(enr1.tags) | set(enr2.tags))
```

bằng:

```python
    tags = req.tags if req.tags is not None else sorted(set(enr1.tags) | set(enr2.tags))
```

- [ ] **Step 4: Chạy test, xác nhận nó xanh**

```bash
pytest services/be/tests/test_search_temporal.py -q
```

Expected: PASS, 13 passed. Bốn test gốc (`test_temporal_search_returns_chains_with_two_hits_each`, `test_temporal_use_llm_false_skips_llm`, `test_temporal_use_llm_true_unions_both_events_tags`, `test_temporal_empty_event_rejected`) phải xanh **mà không sửa nội dung**.

- [ ] **Step 5: Trỏ lại 5 tham chiếu spec chết**

`docs/superpowers/specs/2026-08-24-temporal-search-design.md` chưa từng tồn tại. Đổi cả 5 chỗ sang `2026-08-28-temporal-llm-segmentation-design.md`:

```bash
grep -rln "2026-08-24-temporal-search-design" services/ \
  | xargs sed -i 's|2026-08-24-temporal-search-design\.md|2026-08-28-temporal-llm-segmentation-design.md|g'
grep -rn "2026-08-24-temporal-search-design" services/ ; echo "còn lại: $? (1 = đã hết)"
```

Expected: lệnh `grep` cuối không in dòng nào, in `còn lại: 1`.

Năm chỗ: `services/core/src/searchcore/temporal.py:3`, `services/core/src/searchcore/config.py:83`, `services/core/tests/test_temporal.py:4`, `services/be/src/app/api/search_temporal.py:3`, `services/be/tests/test_search_temporal.py:3`.

- [ ] **Step 6: Chạy toàn bộ test BE và core**

```bash
pytest services/be/tests -q && pytest services/core/tests -q
```

Expected: cả hai PASS. (Sửa ở Step 5 chỉ đụng docstring/comment nên core không được đỏ.)

- [ ] **Step 7: Commit**

```bash
git add services/be/src/app/api/search_temporal.py services/be/tests/test_search_temporal.py \
        services/core/src/searchcore/temporal.py services/core/src/searchcore/config.py \
        services/core/tests/test_temporal.py
git commit -m "feat(be): temporal nhận tags tường minh, bỏ qua LLM khi client đã chọn"
```

---

### Task 4: Type và client FE

**Files:**
- Modify: `services/fe/src/api/types.ts:68-74`
- Modify: `services/fe/src/api/client.ts`

**Interfaces:**
- Consumes: hình dạng JSON của `PrepareResponse` (Task 2) và khoá `tags` của `TemporalSearchRequest` (Task 3).
- Produces: `TemporalSegment`, `TemporalPrepareResponse`, `prepareTemporal(query, signal) -> Promise<TemporalPrepareResponse>`, và `TemporalSearchRequest.tags?: number[]`. Task 5 và 6 dùng đúng các tên này.

- [ ] **Step 1: Thêm type**

Trong `services/fe/src/api/types.ts`, sửa `TemporalSearchRequest` thành:

```ts
export type TemporalSearchRequest = {
	event1: string;
	event2: string;
	use_llm?: boolean;
	exact?: boolean;
	top_k?: number;
	// Bỏ trống = "không qua prepare, để BE quyết theo use_llm". [] = "user đã bỏ tick
	// hết, search toàn kho". Hai cái này KHÁC nhau — đừng gửi [] thay cho "không biết".
	tags?: number[];
};
```

Thêm vào cuối file:

```ts
// Khớp services/be/src/app/api/search_temporal.py — PrepareResponse.
export type TemporalSegment = {
	// Thứ tự thời gian LLM suy ra, đánh lại từ 1 ở BE. Không có `label`: prompt cấm mang
	// nhãn nguồn (E1, "Sự kiện 1") sang output.
	order: number;
	english_clip_query: string;
};

export type TemporalPrepareResponse = {
	segments: TemporalSegment[];
	tags: number[];
	// Khoá JSON luôn là chuỗi: BE khai dict[int, str] nhưng tới đây thành {"3": "..."}.
	// Tra cứu phải là tag_names[String(id)] — tag_names[id] với id số sẽ ra undefined.
	tag_names: Record<string, string>;
	confidence: number;
	tag_source: string;
	warnings: string[];
	snapshot_ver: string;
	timings_ms: Record<string, number>;
};
```

- [ ] **Step 2: Thêm hàm gọi API**

Trong `services/fe/src/api/client.ts`, sửa dòng import đầu file thành:

```ts
import type { NeighborsResponse, SearchRequest, SearchResponse, TemporalPrepareResponse, TemporalSearchRequest, TemporalSearchResponse } from "./types";
```

Thêm vào cuối file:

```ts
export async function prepareTemporal(
	query: string,
	signal?: AbortSignal,
): Promise<TemporalPrepareResponse> {
	const response = await fetch("/api/search/temporal/prepare", {
		method: "POST",
		headers: { "content-type": "application/json" },
		body: JSON.stringify({ query }),
		signal,
	});

	if (!response.ok) {
		const detail = await response.text();
		throw new Error(detail || `Prepare failed (${response.status})`);
	}

	return response.json() as Promise<TemporalPrepareResponse>;
}
```

- [ ] **Step 3: Kiểm type**

```bash
cd services/fe && npx tsc --noEmit
```

Expected: không in ra lỗi nào, exit 0.

- [ ] **Step 4: Commit**

```bash
git add services/fe/src/api/types.ts services/fe/src/api/client.ts
git commit -m "feat(fe): type và client cho /api/search/temporal/prepare"
```

---

### Task 5: Component `TemporalPrepare`

**Files:**
- Create: `services/fe/src/components/TemporalPrepare.tsx`

**Interfaces:**
- Consumes: `prepareTemporal`, `TemporalPrepareResponse`, `TemporalSegment` (Task 4).
- Produces:
  - `toggleSelection(selected: number[], order: number) -> number[]` — hàm thuần, export để kiểm được
  - `TemporalPrepare` — props `{ onSearch: (event1: string, event2: string, tags: number[]) => void; onRunAsKis: (text: string) => void }`. Task 6 truyền đúng hai callback này.

- [ ] **Step 1: Tạo component**

Tạo `services/fe/src/components/TemporalPrepare.tsx`:

```tsx
import { useState } from "react";
import { prepareTemporal } from "../api/client";
import type { TemporalPrepareResponse } from "../api/types";

// Thứ tự bấm = thứ tự sự kiện. Quy tắc viết đủ để không phải đoán:
//   - chưa chọn gì, bấm A          -> A thành 1
//   - đã có 1, bấm B               -> B thành 2
//   - bấm lại đoạn đang chọn       -> bỏ chọn nó; bỏ 1 thì 2 TỤT LÊN thành 1
//   - đã đủ 2 mà bấm đoạn thứ ba   -> BỎ QUA, không thay thế ngầm
// Không thay thế ngầm vì làm thế là vứt mất lựa chọn user vừa cân nhắc; muốn đổi thì bỏ
// chọn tường minh trước. Tách thành hàm thuần để đọc/kiểm mà không phải dựng React.
export function toggleSelection(selected: number[], order: number): number[] {
	const at = selected.indexOf(order);
	if (at !== -1) return selected.filter((o) => o !== order);
	if (selected.length >= 2) return selected;
	return [...selected, order];
}

type Props = {
	onSearch: (event1: string, event2: string, tags: number[]) => void;
	onRunAsKis: (text: string) => void;
};

// Bước 1 của temporal khi BẬT LLM: gõ một câu tiếng Việt -> LLM tách thành N câu tiếng
// Anh cho CLIP -> user sửa, chọn 2 theo thứ tự, tick tag -> mới search.
// Tắt LLM thì App dùng TemporalQueryBuilder (2 ô nhập tay) thay cho component này.
export function TemporalPrepare({ onSearch, onRunAsKis }: Props) {
	const [query, setQuery] = useState("");
	const [prep, setPrep] = useState<TemporalPrepareResponse | null>(null);
	// Chữ đang hiển thị, theo order. Tách khỏi `prep` vì user sửa được: `prep` giữ bản
	// LLM trả về, `texts` giữ bản thực sự đem đi search.
	const [texts, setTexts] = useState<Record<number, string>>({});
	const [selected, setSelected] = useState<number[]>([]);
	const [tags, setTags] = useState<number[]>([]);
	const [loading, setLoading] = useState(false);
	const [error, setError] = useState<string | null>(null);

	const analyze = () => {
		const q = query.trim();
		if (!q) return;
		setLoading(true);
		setError(null);
		prepareTemporal(q)
			.then((data) => {
				setPrep(data);
				setTexts(Object.fromEntries(data.segments.map((s) => [s.order, s.english_clip_query])));
				setSelected([]);
				setTags(data.tags);
			})
			.catch((cause: unknown) => {
				setPrep(null);
				setError(cause instanceof Error ? cause.message : "Prepare failed");
			})
			.finally(() => setLoading(false));
	};

	const segments = prep?.segments ?? [];
	const ready = selected.length === 2;

	return (
		<div className="mb-3">
			<label className="form-label small fw-semibold">Câu truy vấn (tiếng Việt)</label>
			<div className="input-group mb-2">
				<textarea
					className="form-control"
					rows={2}
					placeholder="VD: Cá được đặt lên cân, sau đó một người cầm đuôi con cá khác."
					value={query}
					onChange={(e) => setQuery(e.target.value)}
				/>
				<button
					className="btn btn-outline-primary"
					type="button"
					disabled={loading || !query.trim()}
					onClick={analyze}
				>
					{loading ? "Đang phân tích..." : "Phân tích"}
				</button>
			</div>

			{error && <div className="alert alert-danger p-2 small">{error}</div>}

			{prep && prep.warnings.includes("llm_failed_segment") && (
				<div className="alert alert-warning p-2 small">
					LLM tách đoạn lỗi — đang hiện nguyên câu gốc.
				</div>
			)}

			{segments.length === 1 && (
				<div className="alert alert-info p-2 small">
					Chỉ tìm thấy 1 sự kiện — không tạo được chuỗi thời gian.
					<button
						className="btn btn-sm btn-outline-primary ms-2"
						type="button"
						onClick={() => onRunAsKis(texts[segments[0].order] ?? "")}
					>
						Chạy như KIS thường
					</button>
				</div>
			)}

			{segments.length > 1 && (
				<>
					<label className="form-label small fw-semibold">
						Câu tiếng Anh cho CLIP — bấm số để chọn 2, theo thứ tự:
					</label>
					{segments.map((s) => {
						const at = selected.indexOf(s.order);
						return (
							<div className="d-flex align-items-start gap-2 mb-2" key={s.order}>
								<button
									type="button"
									className={`btn btn-sm ${at === -1 ? "btn-outline-secondary" : "btn-primary"}`}
									style={{ width: "2.5rem", flexShrink: 0 }}
									onClick={() => setSelected((cur) => toggleSelection(cur, s.order))}
									title={at === -1 ? "Chọn làm sự kiện" : "Bỏ chọn"}
								>
									{at === -1 ? "·" : at + 1}
								</button>
								{/* `order` = thứ tự thời gian LLM suy ra, chỉ để đọc. Badge bên trái mới
								    là thứ tự sự kiện user chọn — hai con số khác nhau, đừng gộp hiển thị. */}
								<span className="text-muted small align-self-center" style={{ minWidth: "1.5rem" }}>
									{s.order}.
								</span>
								<textarea
									className="form-control form-control-sm"
									rows={2}
									value={texts[s.order] ?? ""}
									onChange={(e) =>
										setTexts((cur) => ({ ...cur, [s.order]: e.target.value }))
									}
								/>
							</div>
						);
					})}

					<label className="form-label small fw-semibold mt-2">
						Lọc theo lĩnh vực (LLM tự tin {prep!.confidence.toFixed(2)} · nguồn: {prep!.tag_source || "?"}):
					</label>
					<div className="d-flex flex-wrap gap-2 mb-2">
						{Object.entries(prep!.tag_names).map(([id, name]) => {
							const tid = Number(id);
							return (
								<div className="form-check" key={id}>
									<input
										className="form-check-input"
										type="checkbox"
										checked={tags.includes(tid)}
										onChange={() =>
											setTags((cur) =>
												cur.includes(tid) ? cur.filter((t) => t !== tid) : [...cur, tid],
											)
										}
									/>
									<label className="form-check-label small">
										{id} {name}
									</label>
								</div>
							);
						})}
					</div>
					<div className="text-muted small mb-2">
						Bỏ tick hết = search toàn kho (chậm hơn nhưng không bỏ sót).
					</div>

					<button
						className="btn btn-primary"
						type="button"
						disabled={!ready}
						onClick={() => onSearch(texts[selected[0]], texts[selected[1]], tags)}
					>
						Tìm chuỗi {ready ? "" : `(còn thiếu ${2 - selected.length})`}
					</button>
				</>
			)}
		</div>
	);
}
```

- [ ] **Step 2: Kiểm type**

```bash
cd services/fe && npx tsc --noEmit
```

Expected: exit 0, không lỗi. Component chưa được dùng ở đâu nên đây mới chỉ là kiểm cú pháp/type — nối vào App ở Task 6.

- [ ] **Step 3: Commit**

```bash
git add services/fe/src/components/TemporalPrepare.tsx
git commit -m "feat(fe): component chọn và sửa đoạn cho temporal search"
```

---

### Task 6: Nối vào App

**Files:**
- Modify: `services/fe/src/hooks/useTemporalSearch.ts`
- Modify: `services/fe/src/App.tsx`

**Interfaces:**
- Consumes: `TemporalPrepare` + props của nó (Task 5); `searchTemporal` với `tags` (Task 4).
- Produces: luồng chạy được đầu-cuối.

**Ghi chú sửa lại spec:** spec §5.3 nói "bỏ auto-fire theo debounce" — đọc kỹ code thì **không có debounce nào cả**. `useTemporalSearch` đã chạy theo `submittedEvent1/2`, mà hai state đó chỉ đổi trong `search()` khi user submit form ([App.tsx:288-302](../../services/fe/src/App.tsx#L288-L302)). Kích hoạt đã là tường minh sẵn. Việc thật sự cần làm chỉ là luồn thêm `tags`.

- [ ] **Step 1: Luồn `tags` qua hook**

Sửa `services/fe/src/hooks/useTemporalSearch.ts`:

```ts
export function useTemporalSearch(
	event1: string,
	event2: string,
	useLlm = false,
	exact = false,
	tags: number[] | null = null,
) {
```

Trong lời gọi `searchTemporal`, thêm `tags`:

```ts
		searchTemporal(
			{ event1: e1, event2: e2, use_llm: useLlm, exact, tags: tags ?? undefined },
			controller.signal,
		)
```

Sửa mảng dependency ở cuối `useEffect`:

```ts
	// tags là mảng -> mỗi lần render là một identity mới, đưa thẳng vào deps sẽ chạy lại
	// vô hạn. Serialize thành chuỗi. null (không qua prepare) khác [] (bỏ tick hết) nên
	// hai cái phải cho ra hai khoá khác nhau.
	}, [event1, event2, useLlm, exact, tags === null ? "null" : tags.join(",")]);
```

`tags: tags ?? undefined` là có chủ ý: `JSON.stringify` bỏ hẳn khoá `undefined`, nên BE nhận `tags` vắng mặt → `None` → giữ hành vi cũ. Gửi `null` tường minh cũng ra `None`, nhưng bỏ khoá thì payload sạch hơn và khớp với type `tags?: number[]`.

- [ ] **Step 2: Thêm state và bỏ ép tắt LLM trong App**

Trong `services/fe/src/App.tsx`, thêm import:

```tsx
import { TemporalPrepare } from "./components/TemporalPrepare";
```

Thêm state cạnh `submittedEvent2` (dòng ~105):

```tsx
  // null = "không qua prepare" -> BE quyết theo use_llm (luồng nhập tay).
  // [] = "user đã bỏ tick hết" -> search toàn kho. Hai cái KHÁC nhau.
  const [submittedTags, setSubmittedTags] = useState<number[] | null>(null);
```

Sửa lời gọi hook (dòng ~195):

```tsx
  } = useTemporalSearch(submittedEvent1, submittedEvent2, useLlm, exactMode, submittedTags);
```

Bỏ ép tắt LLM ở [App.tsx:516-519](../../services/fe/src/App.tsx#L516-L519) — sửa `onChange` của radio `modeTemporal` từ:

```tsx
                        onChange={() => {
                          setSearchMode("temporal");
                          setUseLlm(false);
                        }}
```

thành:

```tsx
                        onChange={() => setSearchMode("temporal")}
```

- [ ] **Step 3: Rẽ nhánh hai luồng temporal**

Sửa hàm `search()` (dòng ~288) để nó không nuốt submit của luồng có LLM:

```tsx
  function search(event: FormEvent) {
    event.preventDefault();
    if (searchMode === "temporal") {
      // Bật LLM: TemporalPrepare tự có nút riêng (Phân tích / Tìm chuỗi), form submit
      // không có việc gì để làm.
      if (useLlm) return;
      if (event1.trim() && event2.trim()) {
        setSelected([]);
        setSubmittedTags(null);      // nhập tay -> để BE quyết theo use_llm
        setSubmittedEvent1(event1.trim());
        setSubmittedEvent2(event2.trim());
      }
      return;
    }
    if (query.trim()) {
      setSelected([]);
      setSubmitted(query.trim());
    }
  }
```

Sửa phần render trong form (dòng ~526), từ:

```tsx
                      {searchMode === "temporal" ? (
                        <TemporalQueryBuilder
                          event1={event1}
                          event2={event2}
                          onEvent1Change={setEvent1}
                          onEvent2Change={setEvent2}
                        />
                      ) : (
```

thành:

```tsx
                      {searchMode === "temporal" && useLlm ? (
                        <TemporalPrepare
                          onSearch={(e1, e2, tags) => {
                            setSelected([]);
                            setSubmittedTags(tags);
                            setSubmittedEvent1(e1.trim());
                            setSubmittedEvent2(e2.trim());
                          }}
                          onRunAsKis={(text) => {
                            setSearchMode("kis");
                            setSelected([]);
                            setQuery(text);
                            setSubmitted(text);
                          }}
                        />
                      ) : searchMode === "temporal" ? (
                        <TemporalQueryBuilder
                          event1={event1}
                          event2={event2}
                          onEvent1Change={setEvent1}
                          onEvent2Change={setEvent2}
                        />
                      ) : (
```

- [ ] **Step 4: Kiểm type và build**

```bash
cd services/fe && npm run build
```

Expected: `tsc --noEmit` sạch rồi vite build thành công, không lỗi.

- [ ] **Step 5: Kiểm bằng tay đầu-cuối**

Cần core chạy được (hoặc `SC_STUB_MODE=1`) và `LLM_API_KEY` thật. Ba terminal theo README "Cách A".

Kiểm đúng bốn điều — đây là bốn quyết định của spec mà typecheck không bắt được:

1. Bật LLM + mode Temporal → hiện **một** ô query, không phải hai. Gõ câu có "sau đó", bấm **Phân tích** → hiện danh sách câu tiếng Anh kèm badge nhãn.
2. Bấm đoạn A → badge `1`. Bấm đoạn B → badge `2`. Bấm đoạn C → **không có gì xảy ra**. Bấm lại A → A bỏ chọn và B **tụt lên thành `1`**.
3. Sửa chữ trong một textarea → lựa chọn **không** đổi. Bỏ tick hết tag rồi **Tìm chuỗi** → trong tab Network, payload request có `"tags": []` (mảng rỗng tường minh, **không phải** khoá bị bỏ trống).
4. Tắt LLM → quay lại **hai ô nhập tay**, submit vẫn chạy như trước, payload request **không có** khoá `tags`.

Kiểm 3 và 4 soi payload **request** chứ không soi `tags_used` trong response: core xoá `tags_used` về rỗng mỗi khi `tag_fallback` nổ, nên trường đó không phân biệt được "user bỏ tick hết" với "tag có nhưng bị fallback". Cái cần chứng minh ở đây là FE gửi đúng `[]` vs. vắng mặt.

- [ ] **Step 6: Commit**

```bash
git add services/fe/src/hooks/useTemporalSearch.ts services/fe/src/App.tsx
git commit -m "feat(fe): nối luồng prepare vào App, giữ nhập tay khi tắt LLM"
```

---

## Kiểm tra cuối

- [ ] **Toàn bộ test ba service**

```bash
pytest services/core/tests -q && pytest services/be/tests -q && pytest services/ingest/tests -q
```

Expected: cả ba PASS.

- [ ] **Lint + format**

```bash
make lint && make fmt
```

Expected: ruff không báo lỗi. Nếu `make` không có sẵn: `ruff check services/be/src services/core/src && ruff format --check services/be/src services/core/src`.

- [ ] **Build FE**

```bash
cd services/fe && npm run build
```

Expected: thành công.
