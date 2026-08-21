"""LiteLLM implementation of domain structured inference.

Thay cho `cerebras.py` + `gemini.py`: một lớp duy nhất, chọn provider bằng chuỗi model
dạng litellm `"provider/model"`. Cùng convention với `services/be` nên hai service dùng
chung .env và chung cách khai báo model.

GIỮ NGUYÊN CACHE MONGO. `DomainRepository.is_active` so `inference_fingerprint`, mà
fingerprint gồm cả `provider` lẫn `model`. Nếu để `provider="litellm"` hoặc để
`model="cerebras/gpt-oss-120b"` thì fingerprint đổi cho MỌI video đã tag -> `is_active`
trả False -> chạy lại toàn bộ qua LLM, tốn tiền theo số video. Vì vậy:

  - `self.provider` = phần trước dấu "/"  (ví dụ "cerebras")
  - `self.model`    = phần sau dấu "/"    (ví dụ "gpt-oss-120b")  <- đúng như bản cũ
  - `inference_options` giữ y hệt dict mà bản SDK cũ sinh ra

Đánh đổi đã biết: fingerprint không đổi tức là ta KHẲNG ĐỊNH litellm gửi request tương
đương SDK cũ. Nếu litellm gửi khác đi một chút, kết quả cũ vẫn được coi là hợp lệ. Muốn
buộc tính lại thì chạy với `--force`.
"""

from __future__ import annotations

from typing import cast

# typing.override chỉ có từ 3.12; typing_extensions là dep sẵn của pydantic.
from typing_extensions import override

import litellm

from ..config import ReasoningLevel, config
from .enricher import DomainEnricher, ProviderResult, build_system_prompt
from .models import JsonObject, JsonValue, SegmentationResponse

# Cerebras strict mode từ chối các annotation này của JSON Schema.
_UNSUPPORTED_SCHEMA_KEYS = frozenset({"maxItems", "minItems", "title"})


def _strip_schema_keys(value: JsonValue) -> JsonValue:
    """Bỏ annotation mà strict mode không nhận. An toàn: ta tự validate lại bằng
    `validate_segments` nên không mất ràng buộc nào."""
    if isinstance(value, dict):
        return {
            key: _strip_schema_keys(child)
            for key, child in value.items()
            if key not in _UNSUPPORTED_SCHEMA_KEYS
        }
    if isinstance(value, list):
        return [_strip_schema_keys(child) for child in value]
    return value


def split_model(spec: str) -> tuple[str, str]:
    """`"cerebras/gpt-oss-120b"` -> `("cerebras", "gpt-oss-120b")`.

    Thiếu dấu "/" là lỗi cấu hình: không suy được provider thì litellm sẽ đoán, và
    fingerprint sẽ khác bản cũ.
    """
    provider, separator, model = spec.partition("/")
    if not separator or not provider or not model:
        raise ValueError(
            f"DOMAIN_LLM_MODEL phải dạng 'provider/model', nhận {spec!r}. "
            "Ví dụ: cerebras/gpt-oss-120b, gemini/gemini-3.7-flash"
        )
    return provider, model


def _inference_options(provider: str, model: str) -> tuple[JsonObject, JsonObject]:
    """(options ghi vào fingerprint, kwargs gửi cho litellm).

    Phần đầu phải TRÙNG KHỚP dict mà bản SDK cũ sinh ra, nếu không cache mất hiệu lực.
    """
    if provider == "cerebras":
        options: JsonObject = {"seed": 0, "temperature": 0}
        extra: JsonObject = {"seed": 0, "temperature": 0}
        # Bản cũ chỉ bật reasoning cho đúng model này.
        if model == "gpt-oss-120b":
            effort: ReasoningLevel = config.domain_reasoning_effort
            options.update(reasoning_effort=effort, reasoning_format="hidden")
            extra.update(reasoning_effort=effort, reasoning_format="hidden")
        return options, extra

    if provider == "gemini":
        # Bản cũ chỉ bật thinking cho gemini-3.x.
        if model.startswith("gemini-3"):
            level: ReasoningLevel = config.domain_reasoning_effort
            return (
                {"thinking_level": level, "include_thoughts": False},
                {"reasoning_effort": level},
            )
        return {}, {}

    return {}, {}


class LiteLLMDomainEnricher(DomainEnricher):
    """Một provider-neutral enricher. `provider` là thuộc tính INSTANCE (không phải
    ClassVar như trước) vì nó suy từ model."""

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
        semantic_retries: int = 3,
        transport_retries: int = 2,
        completion: object | None = None,
    ) -> None:
        spec = model or config.domain_llm_model
        provider, bare_model = split_model(spec)
        options, extra = _inference_options(provider, bare_model)

        # model = TÊN TRẦN để fingerprint khớp bản SDK cũ; chuỗi đầy đủ để riêng.
        super().__init__(bare_model, semantic_retries, options)
        self.provider: str = provider
        self.litellm_model: str = spec
        self.api_key: str | None = api_key or config.domain_llm_api_key or None
        self.api_base: str | None = api_base or config.domain_llm_api_base or None
        self.transport_retries: int = transport_retries
        self._extra: JsonObject = extra
        # Tiêm được để test không cần mạng.
        self._completion = completion or litellm.completion

    def _response_format(self) -> JsonObject:
        schema = cast(JsonObject, SegmentationResponse.model_json_schema())
        if self.provider == "cerebras":
            schema = cast(JsonObject, _strip_schema_keys(schema))
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "video_domain_segments",
                "strict": True,
                "schema": schema,
            },
        }

    @override
    def _generate(self, prompt: str) -> ProviderResult:
        response = self._completion(
            model=self.litellm_model,
            messages=[
                {"role": "system", "content": build_system_prompt()},
                {"role": "user", "content": prompt},
            ],
            response_format=self._response_format(),
            max_tokens=config.domain_llm_max_tokens,
            timeout=config.domain_llm_timeout_seconds,
            num_retries=self.transport_retries,
            api_key=self.api_key,
            api_base=self.api_base,
            **self._extra,
        )

        choice = response.choices[0]
        if choice.finish_reason == "length":
            raise RuntimeError(
                f"{self.provider} output bị cắt; tăng DOMAIN_LLM_MAX_TOKENS "
                f"(đang {config.domain_llm_max_tokens})"
            )
        content = choice.message.content
        if not content:
            raise ValueError(f"{self.provider} trả content rỗng")

        usage = getattr(response, "usage", None)
        return ProviderResult(
            proposal=SegmentationResponse.model_validate_json(content),
            content=content,
            request_id=getattr(response, "id", None),
            system_fingerprint=getattr(response, "system_fingerprint", None),
            model_version=getattr(response, "model", None),
            usage=(
                cast(JsonObject, usage.model_dump(mode="json"))
                if usage is not None and hasattr(usage, "model_dump")
                else cast(JsonObject, dict(usage)) if usage is not None
                else {}
            ),
        )

    @override
    def close(self) -> None:
        """litellm không giữ client cần đóng."""
