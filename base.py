"""Interfaz común para todos los proveedores + lógica de reintentos."""
from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import AsyncIterator, Awaitable, Callable

from schemas import ChatMessage, LLMError, ModelConfig, ModelResponse, StreamChunk


class BaseLLMClient(ABC):
    def __init__(self, config: ModelConfig) -> None:
        self.config = config

    @abstractmethod
    async def generate(self, messages: list[ChatMessage]) -> ModelResponse:
        """Respuesta completa. Nunca lanza por errores de API: devuelve ModelResponse.error."""

    @abstractmethod
    def stream(self, messages: list[ChatMessage]) -> AsyncIterator[StreamChunk]:
        """Generador asíncrono de fragmentos de texto."""

    @abstractmethod
    def classify_error(self, exc: Exception) -> LLMError:
        """Traduce excepciones del SDK a un LLMError estructurado."""

    # ---- helpers compartidos -------------------------------------------------
    async def _backoff(self, attempt: int) -> None:
        await asyncio.sleep(min(2**attempt * 0.5, 8.0))

    async def _call_with_retry(
        self, call: Callable[[], Awaitable[ModelResponse]]
    ) -> ModelResponse:
        attempt = 0
        while True:
            try:
                return await call()
            except Exception as exc:  # noqa: BLE001 - se convierte en error controlado
                err = self.classify_error(exc)
                if err.retryable and attempt < self.config.max_retries:
                    attempt += 1
                    await self._backoff(attempt)
                    continue
                return ModelResponse(
                    provider=self.config.provider, model=self.config.model, error=err
                )

    async def _stream_with_retry(
        self, open_stream: Callable[[], AsyncIterator[str]]
    ) -> AsyncIterator[StreamChunk]:
        attempt = 0
        while True:
            started = False
            try:
                async for text in open_stream():
                    started = True
                    yield StreamChunk(delta=text)
                yield StreamChunk(done=True)
                return
            except Exception as exc:  # noqa: BLE001
                err = self.classify_error(exc)
                # Solo reintentamos si aún no emitimos tokens (evita texto duplicado).
                if err.retryable and not started and attempt < self.config.max_retries:
                    attempt += 1
                    await self._backoff(attempt)
                    continue
                yield StreamChunk(done=True, error=err)
                return
