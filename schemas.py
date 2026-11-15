"""Esquemas Pydantic: mensajes, configuración, respuestas y errores."""
from __future__ import annotations

from enum import StrEnum
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class Provider(StrEnum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class ChatMessage(BaseModel):
    model_config = ConfigDict(frozen=True)

    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)


class ModelConfig(BaseModel):
    """Parámetros de entrada del modelo (validados al instanciar)."""

    provider: Provider = Provider.OPENAI
    model: str
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, gt=0, le=64_000)
    timeout: float = Field(default=60.0, gt=0)
    max_retries: int = Field(default=3, ge=0, le=10)


class LLMError(BaseModel):
    """Error controlado: se devuelve en lugar de lanzar una excepción."""

    kind: Literal["auth", "rate_limit", "network", "bad_request", "unknown"]
    message: str
    retryable: bool = False


class Usage(BaseModel):
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None


class ModelResponse(BaseModel):
    provider: Provider
    model: str
    content: str = ""
    usage: Usage = Field(default_factory=Usage)
    error: Optional[LLMError] = None

    @property
    def ok(self) -> bool:
        return self.error is None


class StreamChunk(BaseModel):
    """Fragmento emitido por el streaming. El último trae done=True (y error si falló)."""

    delta: str = ""
    done: bool = False
    error: Optional[LLMError] = None
