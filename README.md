# Multimodal Video Search Platform

Truy vấn video bằng ngôn ngữ tự nhiên. Ngân sách latency phần search: **100–200ms**.
Quy mô tham chiếu: ~2M vector × 3072-dim.

## Ba nguyên tắc không thương lượng

1. **Offline và online tách tuyệt đối.** Model nặng chạy trên GPU thuê theo giờ rồi tắt. Hot path chỉ có CPU.
2. **Hot path nằm trên một máy.** BE, search core, encoder cùng host, nói chuyện qua Unix socket (~0.1ms). Mỗi lần vượt biên network là cộng 50–200ms.
3. **Embedding có ba bản.** fp32 trên R2 để rebuild. SQ8 trong RAM để search thô. fp16 trên NVMe để rerank chính xác.

## Cấu trúc

```
proto/          Hợp đồng gRPC giữa BE và core — chốt trước, đổi phải review cả 2 phía
services/core/  Search core: gRPC + FAISS + encoder. Khởi động 2-3 phút, ít deploy
services/be/    BE gateway: FastAPI, LLM enrich, fusion, hydrate. Deploy liên tục
services/ingest/Pipeline offline. Không chạy trong compose production
services/fe/    Frontend React
sql/            Schema + query đã tối ưu
scripts/        Thao tác vận hành
docs/decisions/ ADR — vì sao chọn thế này, đọc trước khi đề xuất đổi
```

## Vì sao core tách container riêng mà không tách repo

- **Cùng repo**: `.proto` là hợp đồng chung, đổi API sửa một PR thay vì PR chéo hai repo.
- **Khác container**: core load 6.7GB index + warmup mất 2–3 phút. BE deploy vài lần một ngày. Chung container thì mỗi lần sửa BE phải reload index.
- **Cùng máy**: Unix socket 0.1ms. Tách sang hai nhà cung cấp là mất trắng mọi tối ưu.

## Bắt đầu

```bash
cp .env.example .env      # điền PG_PASS, S3 keys, LLM key
make proto                # sinh stub gRPC — làm trước tiên
make dev                  # chạy hot reload
```

## Thứ tự làm việc

| # | Việc | Ai |
|---|---|---|
| 1 | Chốt `proto/searchcore/v1/search.proto` | cả nhóm review |
| 2 | Schema Postgres + migration | be |
| 3 | Ingest chạy đúng **một** video end-to-end | ingest |
| 4 | `build_index.py` → snapshot đầy đủ cho 5 video | ingest |
| 5 | Core: load snapshot, 2-tier search, warmup, bench | core |
| 6 | BE `/api/search` đường đơn giản nhất, xác nhận p50 < 60ms | be |
| 7 | FE grid view + keyframe từ R2 | fe |
| 8 | Thêm dần: pre-filter → fusion → LLM → TRAKE → region | cả nhóm |

**Nguyên tắc**: mỗi bước phải chạy end-to-end với dữ liệu nhỏ trước khi mở rộng.
Đừng build cả pipeline rồi mới test — lỗi ở stage 2 sẽ lộ ra sau 20 giờ GPU đã cháy.

## Quy ước

- Branch: `feat/<service>-<mô-tả>`, ví dụ `feat/core-temporal-search`.
- Đổi `proto/` phải có review của cả người giữ BE lẫn core.
- Không commit `.faiss`, `.f16`, `.npy`, `.parquet` — đã chặn trong `.gitignore`.
- Quyết định kiến trúc ghi thành ADR trong `docs/decisions/`, không chôn trong chat.