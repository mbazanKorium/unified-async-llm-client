# Unified Async LLM Client

Cliente asíncrono con interfaz común para **OpenAI** y **Anthropic** (Python 3.12).

- **Intercambiable:** `AsyncLLMManager` elige el proveedor según `LLM_PROVIDER`.
- **Asíncrono:** usa `AsyncOpenAI` / `AsyncAnthropic` con `await`; nunca bloquea el event loop.
- **Streaming:** `manager.stream(messages)` es un generador asíncrono de `StreamChunk`.
- **Validación:** Pydantic (`ChatMessage`, `ModelConfig` con `temperature` 0–2 y `max_tokens`, `ModelResponse`).
- **Errores controlados:** auth, rate limit, red y timeouts se devuelven como `LLMError`
  (en `ModelResponse.error` o en el último `StreamChunk`) en lugar de lanzar excepciones.
  Rate limit, red y errores 5xx se reintentan con backoff exponencial (`max_retries`, por defecto 3).

## Estructura

| Archivo | Contenido |
|---|---|
| `schemas.py` | Modelos Pydantic |
| `base.py` | `BaseLLMClient` abstracta + lógica de reintentos |
| `openai_client.py` / `anthropic_client.py` | Implementaciones por proveedor |
| `manager.py` | `AsyncLLMManager` (selección de proveedor) |
| `main.py` | Script de prueba: modo normal + streaming |

## Instalación

```bash
python3.12 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # Windows: copy .env.example .env
```

## Variables de entorno

| Variable | Descripción |
|---|---|
| `LLM_PROVIDER` | `openai` (defecto) o `anthropic` |
| `LLM_MODEL` | Opcional. Defecto: `gpt-4o-mini` / `claude-sonnet-5-5` |
| `LLM_TEMPERATURE` | 0–2 (defecto 0.7). En Anthropic se recorta a 1.0 |
| `LLM_MAX_TOKENS` | Defecto 1024 |
| `OPENAI_API_KEY` | Requerida si usas OpenAI |
| `ANTHROPIC_API_KEY` | Requerida si usas Anthropic |

## Ejecutar la prueba

```bash
python main.py
```

Pregunta "¿Qué es la entropía?" primero en modo normal y luego en streaming.
Para cambiar de proveedor, edita `LLM_PROVIDER` en `.env`.

## Uso como librería

```python
from manager import AsyncLLMManager
from schemas import ChatMessage, ModelConfig, Provider

manager = AsyncLLMManager(ModelConfig(provider=Provider.ANTHROPIC, model="claude-sonnet-5-5"))
resp = await manager.generate([ChatMessage(role="user", content="Hola")])
print(resp.content if resp.ok else resp.error)
```
