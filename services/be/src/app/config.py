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
    # Ngân sách đã sửa thành 250-500ms tổng (docs/search-design.md §1).
    encode_timeout_s: float = 3.0
    search_timeout_s: float = 2.0

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
    max_top_k: int = 200
    # Chặn 7 keyframe/shot chiếm hết trang (corpus CỐ Ý dày như vậy).
    diversity_max_per_video: int = 3
    diversity_min_time_gap_sec: float = 0.0
    diversity_dedup_threshold: float = 0.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
