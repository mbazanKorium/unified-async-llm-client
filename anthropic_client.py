from __future__ import annotations

from typing import AsyncIterator

import anthropic
from anthropic import AsyncAnthropic

from base import BaseLLMClient
from schemas import ChatMessage, LLMError, ModelConfig, ModelResponse, StreamChunk, Usage


class AnthropicClient(BaseLLMClient):
    def __init__(self, config: ModelConfig, api_key: str) -> None:
        super().__init__(config)
        self._client = AsyncAnthropic(api_key=api_key, timeout=config.timeout, max_retries=0)

    def _payload(self, messages: list[ChatMessage]) -> dict:
        # Anthropic recibe el system prompt aparte y no acepta role="system" en messages.
        system = "\n".join(m.content for m in messages if m.role == "system")
        payload: dict = {
            "model": self.config.model,
            "messages": [m.model_dump() for m in messages if m.role != "system"],
            "max_tokens": self.config.max_tokens,
            # La API acepta temperatura 0-1 (recortamos el rango 0-2). Va por extra_body
            # porque las versiones recientes del SDK ya no la exponen como argumento.
            "extra_body": {"temperature": min(self.config.temperature, 1.0)},
        }
        if system:
            payload["system"] = system
        return payload

    async def generate(self, messages: list[ChatMessage]) -> ModelResponse:
        async def call() -> ModelResponse:
            resp = await self._client.messages.create(**self._payload(messages))
            text = "".join(b.text for b in resp.content if b.type == "text")
            return ModelResponse(
                provider=self.config.provider,
                model=resp.model,
                content=text,
                usage=Usage(
                    input_tokens=resp.usage.input_tokens,
                    output_tokens=resp.usage.output_tokens,
                ),
            )

        return await self._call_with_retry(call)

    def stream(self, messages: list[ChatMessage]) -> AsyncIterator[StreamChunk]:
        async def open_stream() -> AsyncIterator[str]:
            async with self._client.messages.stream(**self._payload(messages)) as stream:
                async for text in stream.text_stream:
                    yield text

        return self._stream_with_retry(open_stream)

    def classify_error(self, exc: Exception) -> LLMError:
        if isinstance(exc, anthropic.AuthenticationError | anthropic.PermissionDeniedError):
            return LLMError(kind="auth", message=f"API key inválida o sin permisos: {exc}")
        if isinstance(exc, anthropic.RateLimitError):
            return LLMError(kind="rate_limit", message=f"Límite de tasa/cuota: {exc}", retryable=True)
        if isinstance(exc, anthropic.APIConnectionError):  # incluye APITimeoutError
            return LLMError(kind="network", message=f"Error de red: {exc}", retryable=True)
        if isinstance(exc, anthropic.InternalServerError):  # incluye 529 overloaded
            return LLMError(kind="network", message=f"Error del servidor: {exc}", retryable=True)
        if isinstance(exc, anthropic.BadRequestError | anthropic.NotFoundError):
            return LLMError(kind="bad_request", message=str(exc))
        return LLMError(kind="unknown", message=f"{type(exc).__name__}: {exc}")
