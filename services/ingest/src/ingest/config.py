"""Validated environment configuration for ingest jobs."""

from typing import ClassVar, Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

type ReasoningLevel = Literal["low", "medium", "high"]


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

    domain_provider: Literal["cerebras", "gemini"] = "cerebras"

    cerebras_api_key: str = ""
    cerebras_base_url: str = "https://api.cerebras.ai"
    cerebras_model: str = "gpt-oss-120b"
    cerebras_timeout_seconds: float = Field(default=60, gt=0)
    cerebras_max_completion_tokens: int = Field(default=8192, gt=0)
    cerebras_reasoning_effort: ReasoningLevel = "medium"

    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.7-flash"
    gemini_timeout_seconds: float = Field(default=60, gt=0)
    gemini_max_output_tokens: int = Field(default=8192, gt=0)
    gemini_thinking_level: ReasoningLevel = "medium"


config = IngestConfig()
