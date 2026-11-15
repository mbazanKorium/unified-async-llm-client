from __future__ import annotations

from typing import AsyncIterator

import openai
from openai import AsyncOpenAI

from base import BaseLLMClient
from schemas import ChatMessage, LLMError, ModelConfig, ModelResponse, StreamChunk, Usage


class OpenAIClient(BaseLLMClient):
    def __init__(self, config: ModelConfig, api_key: str) -> None:
        super().__init__(config)
        # max_retries=0: los reintentos los gestiona BaseLLMClient.
        self._client = AsyncOpenAI(api_key=api_key, timeout=config.timeout, max_retries=0)

    def _payload(self, messages: list[ChatMessage]) -> dict:
        return {
            "model": self.config.model,
            "messages": [m.model_dump() for m in messages],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }

    async def generate(self, messages: list[ChatMessage]) -> ModelResponse:
        async def call() -> ModelResponse:
            resp = await self._client.chat.completions.create(**self._payload(messages))
            usage = resp.usage
            return ModelResponse(
                provider=self.config.provider,
                model=resp.model,
                content=resp.choices[0].message.content or "",
                usage=Usage(
                    input_tokens=usage.prompt_tokens if usage else None,
                    output_tokens=usage.completion_tokens if usage else None,
                ),
            )

        return await self._call_with_retry(call)

    def stream(self, messages: list[ChatMessage]) -> AsyncIterator[StreamChunk]:
        async def open_stream() -> AsyncIterator[str]:
            stream = await self._client.chat.completions.create(
                **self._payload(messages), stream=True
            )
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        return self._stream_with_retry(open_stream)

    def classify_error(self, exc: Exception) -> LLMError:
        if isinstance(exc, openai.AuthenticationError | openai.PermissionDeniedError):
            return LLMError(kind="auth", message=f"API key inválida o sin permisos: {exc}")
        if isinstance(exc, openai.RateLimitError):
            return LLMError(kind="rate_limit", message=f"Límite de tasa/cuota: {exc}", retryable=True)
        if isinstance(exc, openai.APIConnectionError):  # incluye APITimeoutError
            return LLMError(kind="network", message=f"Error de red: {exc}", retryable=True)
        if isinstance(exc, openai.InternalServerError):
            return LLMError(kind="network", message=f"Error del servidor: {exc}", retryable=True)
        if isinstance(exc, openai.BadRequestError | openai.NotFoundError):
            return LLMError(kind="bad_request", message=str(exc))
        return LLMError(kind="unknown", message=f"{type(exc).__name__}: {exc}")
