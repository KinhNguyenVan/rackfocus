"""Google Gen AI SDK implementation of domain structured inference."""

from __future__ import annotations

from typing import ClassVar, Protocol, cast, override

from google import genai
from google.genai import types

from ..config import ReasoningLevel, config
from .enricher import DomainEnricher, ProviderResult, build_system_prompt
from .models import JsonObject, SegmentationResponse


class _Models(Protocol):
    def generate_content(self, **kwargs: object) -> object: ...


class _Client(Protocol):
    models: _Models


class GeminiDomainEnricher(DomainEnricher):
    provider: ClassVar[str] = "gemini"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        semantic_retries: int = 3,
        transport_retries: int = 2,
        client: object | None = None,
    ) -> None:
        key = api_key or config.gemini_api_key
        if client is None and not key:
            raise ValueError("Thiếu GEMINI_API_KEY")
        selected_model = model or config.gemini_model
        self.thinking_level: ReasoningLevel | None = (
            config.gemini_thinking_level
            if selected_model.startswith("gemini-3")
            else None
        )
        super().__init__(
            selected_model,
            semantic_retries,
            (
                {"thinking_level": self.thinking_level, "include_thoughts": False}
                if self.thinking_level
                else None
            ),
        )
        self.client: _Client = cast(
            _Client,
            client
            or genai.Client(
                api_key=key,
                http_options=types.HttpOptions(
                    timeout=round(config.gemini_timeout_seconds * 1_000),
                    retry_options=types.HttpRetryOptions(
                        attempts=transport_retries + 1
                    ),
                ),
            ),
        )

    @override
    def _generate(self, prompt: str) -> ProviderResult:
        response = cast(
            types.GenerateContentResponse,
            self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=build_system_prompt(),
                    response_mime_type="application/json",
                    response_json_schema=SegmentationResponse.model_json_schema(),
                    max_output_tokens=config.gemini_max_output_tokens,
                    thinking_config=(
                        types.ThinkingConfig(
                            include_thoughts=False,
                            thinking_level=types.ThinkingLevel(self.thinking_level),
                        )
                        if self.thinking_level
                        else None
                    ),
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(
                        disable=True
                    ),
                ),
            ),
        )
        content = response.text
        finish_reason = (
            response.candidates[0].finish_reason if response.candidates else None
        )
        if finish_reason is types.FinishReason.MAX_TOKENS:
            raise RuntimeError("Gemini output bị cắt; tăng GEMINI_MAX_OUTPUT_TOKENS")
        if not content:
            raise ValueError("Gemini trả content rỗng")
        return ProviderResult(
            proposal=SegmentationResponse.model_validate_json(content),
            content=content,
            request_id=response.response_id,
            model_version=response.model_version,
            usage=(
                cast(JsonObject, response.usage_metadata.model_dump(mode="json"))
                if response.usage_metadata
                else {}
            ),
        )

    @override
    def close(self) -> None:
        if callable(close := getattr(self.client, "close", None)):
            _ = close()
