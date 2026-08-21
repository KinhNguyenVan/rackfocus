"""Validated environment configuration for ingest jobs."""

from typing import ClassVar, Literal, TypeAlias

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ReasoningLevel: TypeAlias = Literal["low", "medium", "high"]


class IngestConfig(BaseSettings):
    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        env_file=".env", extra="ignore"
    )

    database_url: str = "postgresql+asyncpg://vs:devpass123@localhost:5432/rackfocus"
    mongo_uri: str = "mongodb://localhost:27017"
    mongo_db: str = ""

    aws_access_key: str = Field(
        default="", validation_alias=AliasChoices("AWS_ACCESS_KEY", "AWS_ACCESS_KEY_ID")
    )
    aws_secret_key: str = Field(
        default="",
        validation_alias=AliasChoices("AWS_SECRET_KEY", "AWS_SECRET_ACCESS_KEY"),
    )
    aws_region: str = "ap-southeast-1"
    aws_bucket_name: str = "aic-bucket-2026"

    # ── LLM cho domain enrichment (qua litellm) ──────────────────────
    # Dạng "provider/model", cùng convention với services/be nên hai service khai báo
    # model giống nhau và dùng chung .env.
    #
    # API key: KHÔNG khai ở đây theo từng provider nữa — litellm tự đọc biến chuẩn của
    # provider (CEREBRAS_API_KEY, GEMINI_API_KEY, GROQ_API_KEY, OPENAI_API_KEY...).
    # `domain_llm_api_key` chỉ để ghi đè khi cần.
    domain_llm_model: str = "cerebras/gpt-oss-120b"
    domain_llm_api_key: str = ""
    domain_llm_api_base: str = ""
    domain_llm_timeout_seconds: float = Field(default=60, gt=0)
    domain_llm_max_tokens: int = Field(default=32768, gt=0)
    # Cerebras gọi là reasoning_effort, Gemini gọi là thinking_level — cùng một nút vặn,
    # litellm map sang tên của từng provider.
    domain_reasoning_effort: ReasoningLevel = "medium"


config = IngestConfig()
