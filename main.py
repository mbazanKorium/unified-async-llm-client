"""Script de validación: modo normal y streaming con el proveedor configurado."""
import asyncio

from dotenv import load_dotenv

from manager import AsyncLLMManager, ConfigurationError
from schemas import ChatMessage

QUESTION = "¿Qué es la entropía?"


async def main() -> None:
    load_dotenv()
    try:
        manager = AsyncLLMManager.from_env()
    except ConfigurationError as exc:
        print(f"[config] {exc}")
        return

    cfg = manager.config
    messages = [
        ChatMessage(role="system", content="Responde en español, en máximo 3 frases."),
        ChatMessage(role="user", content=QUESTION),
    ]

    print(f"== Proveedor: {cfg.provider.value} | modelo: {cfg.model}\n")

    print("--- Modo normal ---")
    resp = await manager.generate(messages)
    if resp.ok:
        print(resp.content)
        print(f"(tokens in/out: {resp.usage.input_tokens}/{resp.usage.output_tokens})")
    else:
        print(f"[error {resp.error.kind}] {resp.error.message}")

    print("\n--- Streaming ---")
    async for chunk in manager.stream(messages):
        if chunk.error:
            print(f"\n[error {chunk.error.kind}] {chunk.error.message}")
        else:
            print(chunk.delta, end="", flush=True)
    print()


if __name__ == "__main__":
    asyncio.run(main())
