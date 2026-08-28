# Tách sự kiện bằng LLM cho Temporal Search (TRAKE)

Ngày: 2026-08-28. Phạm vi: `services/be` + `services/fe`. **Không đổi `proto/` và không đổi
`services/core`.**

Thay luồng temporal hiện tại (2 ô nhập tay) bằng: người dùng gõ **một** câu tiếng Việt,
LLM tách thành N mô tả tiếng Anh cho CLIP, in ra UI cho người dùng **sửa** và **chọn 2**
sự kiện theo thứ tự, rồi mới search.

---

## 1. Vấn đề

`prompt.yaml` (ở gốc repo, chạy tay qua `run_prompt.py`) đã làm được việc tách một câu
truy vấn tiếng Việt thành các đoạn mô tả tiếng Anh sạch cho CLIP — bỏ hết từ nói VỀ cấu
trúc video ("sau đó", "bắt đầu bằng", "cảnh cuối"), bỏ đếm số chính xác, bỏ hướng chuyển
động. Nhưng nó chưa nằm trong BE: người dùng vẫn phải tự tay chẻ câu hỏi thành 2 sự kiện
và tự dịch sang tiếng Anh.

Ba ràng buộc va nhau:

1. `prompt.yaml` trả **N đoạn**, N có thể là 1 ("Full query" khi không có mốc thời gian)
   hoặc 3–4 (ví dụ 3 trong chính prompt trả `Context` + `E1` + `E2` + `E3`).
2. Core `SearchTemporal` nhận **đúng 2** vector. Đây là ràng buộc cứng của `proto/`.
3. Lọc tag là **CỨNG** — frame ngoài tag đã chọn là không thể với tới ở bất kỳ
   `ef_search`/`top_k` nào. Đây là lý do temporal hiện mặc định `use_llm=False`.

Không có cách tự động nào giải (1) va (2) mà không đoán bừa. Nên **người dùng quyết định**:
LLM đề xuất N đoạn, người dùng chọn 2.

## 2. Quyết định

| # | Quyết định | Lý do |
|---|---|---|
| D1 | LLM tách đoạn chạy **song song** với LLM chọn tag, cả hai trên câu gốc tiếng Việt | Tổng = max(tách, tag), không phải tổng. Cả hai cần ngữ cảnh cả câu. |
| D2 | Chia làm **2 pha**: prepare (LLM) → người dùng chọn → search | Người dùng phải bấm chọn trước khi search, nên không thể còn là 1 request. |
| D3 | In **tất cả** N đoạn, mỗi đoạn **sửa được** | LLM viết lại sai là chuyện thường; sửa tay rẻ hơn chạy lại LLM. |
| D4 | Chọn **đúng 2**, thứ tự bấm = thứ tự sự kiện | Ràng buộc cứng của core (§1.2). |
| D5 | N=1 → không temporal được, mời chạy KIS thường | Chuỗi cần 2 mốc. Nói thẳng thay vì bịa mốc thứ 2. |
| D6 | Tag **hiện ra và tick được** | Lọc tag là cứng (§1.3). Tag sai giết cả chuỗi mà không có tín hiệu gì ở kết quả. |
| D7 | Pha 1 **không** encode trước | Đoạn sửa được ⇒ vector cache hoá ôi ngay khi người dùng gõ. Encode ở pha 2. |
| D8 | Luồng nhập tay 2 ô **giữ nguyên** khi tắt LLM | Đường thoát khi LLM hỏng/không có API key. |

**Ngân sách latency 100–200ms chỉ áp cho pha 2.** Pha 1 nằm sau một cú bấm có chủ đích của
người dùng, không phải hot path — 0,5–1,5s ở đó chấp nhận được. Pha 1 nhiều khả năng bị
chặn bởi bước tách đoạn chứ không phải bước chọn tag: `prompt.yaml` dài ~134 dòng, lớn hơn
hẳn prompt trong `enrich.py`.

## 3. Luồng

```
Pha 1  POST /api/search/temporal/prepare  {query}            ← KHÔNG phải hot path
       └─ asyncio.gather(
            segment.segment(query)   ← prompt.yaml  → N đoạn tiếng Anh
            enrich.enrich(query)     ← enrich.py    → tags + confidence
          )                            tổng = max(hai cái), không phải tổng
       → {segments[], tags[], tag_names{}, confidence, warnings, timings_ms}

       ── người dùng sửa câu, bấm chọn 2 theo thứ tự, tick/bỏ tick tag ──

Pha 2  POST /api/search/temporal  {event1, event2, tags:[...], use_llm:false}
       └─ gather(encode(e1), encode(e2)) → core SearchTemporal
       → TemporalSearchResponse (giữ nguyên hình dạng cũ)
```

Tắt LLM thì bỏ hẳn pha 1: 2 ô nhập tay → thẳng pha 2 với `tags=None, use_llm=false`, y hệt
hôm nay.

## 4. BE

### 4.1 `services/be/src/app/services/segment.py` (mới)

Theo đúng hợp đồng của `enrich.py`: **không bao giờ raise**, luôn có đường thoát.

```python
@dataclass
class Segment:
    order: int
    label: str              # "Full query" | "Event 1" | "E1" | "Context" | ...
    english_clip_query: str

@dataclass
class Segmentation:
    segments: list[Segment] = field(default_factory=list)
    model: str = ""
    latency_ms: float = 0.0
    error: str = ""
```

`async def segment(query: str, settings) -> Segmentation`

Lỗi/timeout/JSON rác → trả **một** đoạn `Segment(1, "Full query", query)` kèm `error`. Hỏng
kiểu đó rơi đúng vào nhánh N=1 mà UI đã phải xử lý sẵn (D5) — không cần đường lỗi riêng.

Parse: `prompt.yaml` yêu cầu trả JSON **array**, không phải object — nên regex bóc là
`\[.*\]` (`re.DOTALL`), khác `\{.*\}` của `enrich.py`. Bỏ đoạn thiếu `english_clip_query`
hoặc có chuỗi rỗng; nếu sau khi lọc không còn đoạn nào, rơi về đường thoát trên. `order`
lấy lại theo vị trí trong mảng sau khi lọc, không tin số LLM trả (LLM có thể đánh trùng
hoặc nhảy số).

Dùng lại nguyên `settings.llm_*` hiện có — không thêm biến môi trường mới.

### 4.2 `services/be/src/app/services/segment_prompt.txt` (mới)

`prompt.yaml` chuyển vào đây, đọc một lần lúc import. Hai lý do: nó phải nằm trong
container BE, và 134 dòng prompt đang được sửa liên tục thì diff dạng file dễ đọc hơn hẳn
dạng chuỗi Python nhúng trong code. `run_prompt.py` ở gốc trỏ sang đường dẫn mới để harness
chạy tay vẫn dùng đúng một bản prompt (không được để hai bản trôi lệch nhau).

### 4.3 `services/be/src/app/api/search_temporal.py` (sửa)

Thêm endpoint:

```python
class PrepareRequest(BaseModel):
    query: str = Field(min_length=1)

class PrepareResponse(BaseModel):
    segments: list[SegmentOut]      # order, label, english_clip_query
    tags: list[int]                 # tag LLM/guard chọn
    tag_names: dict[int, str]       # {id: "tên hiển thị"} cho toàn vocab
    confidence: float
    tag_source: str                 # "llm" | "guard_low_confidence" | "llm_empty"
    warnings: list[str]             # "llm_failed_segment", "llm_failed_tags"
    snapshot_ver: str
    timings_ms: dict[str, float]    # segment, enrich, total
```

`POST /api/search/temporal/prepare` → `tagvocab.get(st)` rồi
`asyncio.gather(segment.segment(...), enrich.enrich(...))`.

`tag_names` trả **cả vocab**, không chỉ tag đã chọn — UI cần hiện tag chưa chọn để người
dùng tick thêm vào (D6), không chỉ bỏ bớt.

Lưu ý khi viết FE: khoá JSON luôn là chuỗi, nên `dict[int, str]` sang tới client thành
`{"3": "Văn hoá - Giải trí"}`. Type ở `types.ts` phải là `Record<string, string>` và tra
cứu bằng `tag_names[String(id)]` — dùng `tag_names[id]` với `id` số sẽ ra `undefined`.

Sửa `TemporalSearchRequest`:

```python
tags: list[int] | None = None
```

- `tags` **không None** → dùng nguyên xi, **bỏ hẳn** bước enrich (không gọi LLM).
- `tags is None` → giữ y nguyên hành vi hiện tại theo cờ `use_llm`.

`[]` và `None` **không được** gộp làm một: `[]` nghĩa là "người dùng đã bỏ tick hết, search
toàn kho", `None` nghĩa là "không qua pha prepare, quyết định theo `use_llm`". Gộp hai cái
này lại là cách âm thầm bật lại lọc tag mà người dùng vừa cố ý tắt.

`ResponseMeta.tags_used` từ core vẫn được trả về như cũ, nên đối chiếu được tag người dùng
gửi với tag core thật sự dùng.

## 5. FE

### 5.1 Bỏ ép tắt LLM khi vào temporal

[App.tsx:516-519](services/fe/src/App.tsx#L516-L519) đang gọi `setUseLlm(false)` khi
chuyển sang temporal mode. Bỏ dòng đó — temporal + LLM giờ là đường chính.

`useLlm` trở thành công tắc giữa hai luồng:

- **Tắt** → `TemporalQueryBuilder` cũ, 2 ô nhập tay, **không đổi gì**.
- **Bật** → `TemporalPrepare` mới.

### 5.2 `services/fe/src/components/TemporalPrepare.tsx` (mới)

```
┌─ Câu truy vấn (tiếng Việt) ─────────────────────────┐
│ Đoạn video bắt đầu bằng ảnh cận đầu một con lân...  │  [ Phân tích ]
└─────────────────────────────────────────────────────┘

Câu tiếng Anh cho CLIP — bấm số để chọn 2, theo thứ tự:

  ( )  Context   [ A close-up of a white lion-dance head...        ]
  (1)  E1        [ Golden dragons fully in frame, spinning.        ]
  ( )  E2        [ The lion figure's legs landing back on...       ]
  (2)  E3        [ A mallet striking a bronze gong...              ]
                   ↑ textarea, sửa được

Lọc theo lĩnh vực (LLM tự tin 0.82 · nguồn: llm):
  [x] 3  Văn hoá - Giải trí      [ ] 5  Thể thao
  [x] 7  Xã hội                  [ ] 9  Giáo dục
  Bỏ tick hết = search toàn kho.

                                              [ Tìm chuỗi ]  (bật khi đã chọn đủ 2)
```

**Tách rõ vùng bấm chọn và vùng sửa chữ** để hai thao tác không giẫm chân nhau: badge số
bên trái để chọn, textarea bên phải để sửa. Bấm vào chữ chỉ để sửa, không đổi lựa chọn.

Quy tắc chọn, viết đủ để không phải đoán:

- Chưa chọn gì → bấm đoạn A: A thành `1`.
- Đã có `1` → bấm đoạn B: B thành `2`.
- Bấm lại đoạn đang được chọn → bỏ chọn nó; nếu bỏ `1` thì `2` **tụt lên thành `1`** (không
  để trống số 1 rồi bắt người dùng đoán).
- Đã đủ 2 mà bấm đoạn thứ ba → **bỏ qua**, không thay thế. Thay ngầm sẽ làm mất lựa chọn
  người dùng vừa cân nhắc; muốn đổi thì bỏ chọn tường minh trước.

Sửa chữ trong textarea **không** ảnh hưởng lựa chọn.

N=1 hiện thay bằng: *"Chỉ tìm thấy 1 sự kiện — không tạo được chuỗi thời gian."* kèm nút
chạy đoạn đó như KIS thường (đổ vào `submitted` của `useSearch`).

State đoạn/tag giữ lại sau khi search, để đổi cặp khác hoặc sửa câu rồi search lại **không
phải chạy lại LLM**.

### 5.3 Hook và client

- `services/fe/src/api/client.ts`: thêm `prepareTemporal(query, signal)`.
- `services/fe/src/api/types.ts`: `TemporalSegment`, `PrepareResponse`; thêm `tags?: number[]`
  vào body của `searchTemporal`.
- `useTemporalSearch`: bỏ auto-fire theo debounce, chuyển sang chạy khi được gọi tường minh,
  và nhận thêm `tags`. Auto-fire theo `useEffect` là sai ở luồng mới — search phải nổ đúng
  lúc bấm nút, không phải mỗi lần người dùng gõ một chữ trong textarea.

## 6. Test

`services/be/tests/test_search_temporal.py` đã có sẵn cách mock: gán
`sys.modules["litellm"]` bằng một `SimpleNamespace(acompletion=...)`. Dùng lại y hệt, phân
biệt hai lời gọi bằng nội dung `messages[0]["content"]` (system prompt tách đoạn khác system
prompt chọn tag).

| Test | Khẳng định |
|---|---|
| `test_prepare_returns_segments_and_tags` | N đoạn từ mock, kèm `tags`/`tag_names`/`confidence` |
| `test_prepare_runs_both_llms` | đúng 2 lời gọi litellm |
| `test_prepare_segment_failure_falls_back_to_full_query` | LLM tách lỗi → 1 đoạn `"Full query"` = câu gốc, `warnings` có `llm_failed_segment` |
| `test_prepare_bad_json_falls_back` | trả chữ không phải JSON → như trên, không raise |
| `test_temporal_explicit_tags_skip_llm` | gửi `tags=[0]` → `llm.calls == 0`, tag tới core đúng `[0]` |
| `test_temporal_empty_tags_means_no_filter` | `tags=[]` → không lọc, **và** không gọi LLM |
| `test_temporal_tags_none_keeps_use_llm_behaviour` | `tags` vắng mặt → hành vi cũ nguyên vẹn |

Bốn test temporal hiện có phải **pass nguyên trạng, không sửa** — đó là bằng chứng luồng
nhập tay không bị đụng vào (D8).

## 7. Ngoài phạm vi

- **Chuỗi >2 sự kiện.** Cần `proto`/core mới (ghép cặp E1→E2, E2→E3, join theo hit giữa,
  ép `t1 < t2 < t3`). Đã cân nhắc và loại: đổi cả hợp đồng gRPC lẫn cách chấm điểm.
- **Encode sẵn ở pha 1** (D7).
- **Cache kết quả prepare ở server.** Client giữ state là đủ; thêm cache là thêm trạng thái
  server cho một thứ chỉ sống trong một phiên người dùng.
- **Model riêng cho bước tách đoạn** (`LLM_SEGMENT_MODEL`). Chưa có bằng chứng cần; thêm khi
  đo thấy cần.

## 8. Dọn kèm

`services/core/src/searchcore/temporal.py`, `services/be/src/app/api/search_temporal.py:3`
và `services/be/tests/test_search_temporal.py:3` đang trỏ tới
`docs/superpowers/specs/2026-08-24-temporal-search-design.md` — **file đó chưa từng tồn
tại**. Trỏ lại vào tài liệu này.
