"""Settings: SEARCHCORE_TARGET, LLM, S3, timeout. Xem docs/search-design.md §1, §3."""
from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # ── Search core ──────────────────────────────────────────────────
    searchcore_target: str = "unix:///var/run/searchcore/sc.sock"

    # Deadline gRPC. KHÔNG để 200ms như Handoff §4.2: encoder SigLIP so400m tốn
    # 53.3 GFLOP/query = 170-420ms trên 4 core, nên 200ms cho 100% DEADLINE_EXCEEDED.
    # Nới rộng hơn cả ngân sách 250-500ms ở docs/search-design.md §1: máy chạy thật ở
    # đây chưa benchmark riêng (số liệu §1 đo trên máy khác), corpus giờ 613k điểm
    # (nặng hơn 250k lúc đo), và chế độ "exact" brute-force cả corpus khi không tag —
    # thà chậm mà chạy được còn hơn DEADLINE_EXCEEDED giữa lúc chuẩn bị thi.
    encode_timeout_s: float = 15.0
    search_timeout_s: float = 30.0

    # ── LLM (litellm: "provider/model") ──────────────────────────────
    llm_model: str = "groq/llama-3.3-70b-versatile"
    llm_api_key: str = ""
    llm_timeout_s: float = 6.0
    # temperature 0: cùng một query phải cho cùng tag. Nếu không, user gõ lại đúng câu
    # cũ sau vài phút lại ra kết quả khác và không thể biết vì sao.
    llm_temperature: float = 0.0
    llm_max_tags: int = 5

    # Bật/tắt LLM cho từng request được (giữ cờ use_llm của Handoff §4.2) — khi LLM chọn
    # sai tag thì đây là đường lùi duy nhất của người dùng.
    llm_enabled: bool = True

    # ── Vận hành ─────────────────────────────────────────────────────
    default_top_k: int = 20
    # 300: khớp thiết kế rerank (coarse top 1k -> rerank exact top 300) / exact
    # (brute-force thẳng top 300) — trước là 200, cắt mất FE xin 300.
    max_top_k: int = 300
    # 0 = tắt hẳn: trả nguyên top_k theo đúng thứ tự core xếp hạng, không dedup theo
    # video. Trước để 3 (chặn 7 keyframe/shot chiếm hết trang) nhưng làm top_k=300 xin
    # bị cắt còn vài chục nếu top match dồn vào ít video — muốn xem đủ top-N thật thì
    # tắt, đánh đổi là có thể thấy nhiều frame gần giống hệt nhau liền kề.
    diversity_max_per_video: int = 0
    diversity_min_time_gap_sec: float = 0.0
    diversity_dedup_threshold: float = 0.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
