"""AsyncLLMManager: elige el proveedor según configuración."""
from __future__ import annotations

import os
from typing import AsyncIterator

from base import BaseLLMClient
from schemas import ChatMessage, ModelConfig, ModelResponse, Provider, StreamChunk

DEFAULT_MODELS = {
    Provider.OPENAI: "gpt-4o-mini",
    Provider.ANTHROPIC: "claude-sonnet-5-5",
}
API_KEY_VARS = {
    Provider.OPENAI: "OPENAI_API_KEY",
    Provider.ANTHROPIC: "ANTHROPIC_API_KEY",
}


class ConfigurationError(ValueError):
    """Error de configuración (falta la API key, proveedor desconocido...)."""


class AsyncLLMManager:
    def __init__(self, config: ModelConfig, api_key: str | None = None) -> None:
        key = api_key or os.getenv(API_KEY_VARS[config.provider])
        if not key:
            raise ConfigurationError(
                f"Falta la variable {API_KEY_VARS[config.provider]} para {config.provider.value}."
            )
        self.config = config
        self.client = self._build(config, key)

    @staticmethod
    def _build(config: ModelConfig, key: str) -> BaseLLMClient:
        # Import perezoso: solo se carga el SDK del proveedor elegido.
        if config.provider is Provider.OPENAI:
            from openai_client import OpenAIClient

            return OpenAIClient(config, key)
        from anthropic_client import AnthropicClient

        return AnthropicClient(config, key)

    @classmethod
    def from_env(cls, **overrides) -> "AsyncLLMManager":
        """Lee LLM_PROVIDER, LLM_MODEL, LLM_TEMPERATURE y LLM_MAX_TOKENS del entorno."""
        raw = os.getenv("LLM_PROVIDER", "openai").lower()
        try:
            provider = Provider(raw)
        except ValueError as exc:
            raise ConfigurationError(f"LLM_PROVIDER inválido: {raw!r} (openai | anthropic)") from exc
        data = {
            "provider": provider,
            "model": os.getenv("LLM_MODEL") or DEFAULT_MODELS[provider],
            "temperature": float(os.getenv("LLM_TEMPERATURE", "0.7")),
            "max_tokens": int(os.getenv("LLM_MAX_TOKENS", "1024")),
        }
        data.update(overrides)
        return cls(ModelConfig(**data))

    async def generate(self, messages: list[ChatMessage]) -> ModelResponse:
        return await self.client.generate(messages)

    def stream(self, messages: list[ChatMessage]) -> AsyncIterator[StreamChunk]:
        return self.client.stream(messages)
