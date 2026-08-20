"""Cerebras implementation of domain structured inference."""

from __future__ import annotations

from typing import ClassVar, Protocol, cast, override

from cerebras.cloud.sdk import Cerebras
from cerebras.cloud.sdk.types.chat import completion_create_params
from cerebras.cloud.sdk.types.chat.chat_completion import ChatCompletionResponse

from ..config import ReasoningLevel, config
from .enricher import DomainEnricher, ProviderResult, build_system_prompt
from .models import JsonObject, JsonValue, SegmentationResponse

_UNSUPPORTED_SCHEMA_KEYS = frozenset({"maxItems", "minItems", "title"})


def _cerebras_schema(value: JsonValue) -> JsonValue:
    """Remove JSON Schema annotations unsupported by Cerebras strict mode."""
    if isinstance(value, dict):
        return {
            key: _cerebras_schema(child)
            for key, child in value.items()
            if key not in _UNSUPPORTED_SCHEMA_KEYS
        }
    if isinstance(value, list):
        return [_cerebras_schema(child) for child in value]
    return value


class _Completions(Protocol):
    def create(self, **kwargs: object) -> object: ...


class _Chat(Protocol):
    completions: _Completions


class _Client(Protocol):
    chat: _Chat


class CerebrasDomainEnricher(DomainEnricher):
    provider: ClassVar[str] = "cerebras"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        semantic_retries: int = 3,
        transport_retries: int = 2,
        client: object | None = None,
    ) -> None:
        key = api_key or config.cerebras_api_key
        if client is None and not key:
            raise ValueError("Thiếu CEREBRAS_API_KEY")
        selected_model = model or config.cerebras_model
        self.reasoning_effort: ReasoningLevel | None = (
            config.cerebras_reasoning_effort
            if selected_model == "gpt-oss-120b"
            else None
        )
        inference_options: JsonObject = {"seed": 0, "temperature": 0}
        if self.reasoning_effort:
            inference_options.update(
                reasoning_effort=self.reasoning_effort,
                reasoning_format="hidden",
            )
        super().__init__(
            selected_model,
            semantic_retries,
            inference_options,
        )
        base_url = config.cerebras_base_url.rstrip("/").removesuffix("/v1")
        self.client: _Client = cast(
            _Client,
            client
            or Cerebras(
                api_key=key,
                base_url=base_url,
                timeout=config.cerebras_timeout_seconds,
                max_retries=transport_retries,
            ),
        )

    @override
    def _generate(self, prompt: str) -> ProviderResult:
        schema = cast(
            JsonObject,
            _cerebras_schema(
                cast(JsonObject, SegmentationResponse.model_json_schema())
            ),
        )
        messages: list[completion_create_params.Message] = [
            {"role": "system", "content": build_system_prompt()},
            {"role": "user", "content": prompt},
        ]
        reasoning_options: dict[str, object] = (
            {
                "reasoning_effort": self.reasoning_effort,
                "reasoning_format": "hidden",
            }
            if self.reasoning_effort
            else {}
        )
        response = cast(
            ChatCompletionResponse,
            self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "video_domain_segments",
                        "strict": True,
                        "schema": schema,
                    },
                },
                max_completion_tokens=config.cerebras_max_completion_tokens,
                seed=0,
                temperature=0,
                **reasoning_options,
            ),
        )
        choice = response.choices[0]
        if choice.finish_reason == "length":
            raise RuntimeError(
                "Cerebras output bị cắt; tăng CEREBRAS_MAX_COMPLETION_TOKENS"
            )
        content = choice.message.content
        if not content:
            raise ValueError("Cerebras trả content rỗng")
        return ProviderResult(
            proposal=SegmentationResponse.model_validate_json(content),
            content=content,
            request_id=response.id,
            system_fingerprint=response.system_fingerprint,
            usage=(
                cast(JsonObject, response.usage.to_dict()) if response.usage else {}
            ),
        )

    @override
    def close(self) -> None:
        if callable(close := getattr(self.client, "close", None)):
            _ = close()
