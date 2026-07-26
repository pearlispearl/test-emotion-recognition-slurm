"""Model-provider strategies and adapters.

Strategy: every provider implements ModelProvider.infer().
Adapter: OllamaVisionAdapter converts our common request into Ollama format.
Factory: ProviderFactory creates the configured provider.
"""

from __future__ import annotations

import asyncio
import base64
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence


@dataclass(frozen=True)
class ProviderResponse:
    raw_output: str
    elapsed_ms: float


class ModelProvider(Protocol):
    """Common strategy used by the worker pipeline."""

    async def infer(
        self,
        *,
        system_prompt: str,
        user_message: str,
        image_paths: Sequence[Path],
    ) -> ProviderResponse:
        ...


class OllamaVisionAdapter:
    """Adapter from the common ModelProvider interface to Ollama."""

    def __init__(
        self,
        *,
        host: str,
        model: str,
        temperature: float,
        top_p: float,
        num_ctx: int,
        timeout_s: float,
        retries: int,
        retry_delay_s: float,
        semaphore: asyncio.Semaphore,
    ) -> None:
        try:
            from ollama import AsyncClient
        except ImportError as exc:
            raise RuntimeError("Python package 'ollama' is required") from exc

        self._client = AsyncClient(host=host, timeout=timeout_s)
        self._model = model
        self._temperature = temperature
        self._top_p = top_p
        self._num_ctx = num_ctx
        self._retries = retries
        self._retry_delay_s = retry_delay_s
        self._semaphore = semaphore

    @staticmethod
    def _encode_image(path: Path) -> str:
        with path.open("rb") as handle:
            return base64.b64encode(handle.read()).decode("utf-8")

    async def infer(
        self,
        *,
        system_prompt: str,
        user_message: str,
        image_paths: Sequence[Path],
    ) -> ProviderResponse:
        encoded_images = [self._encode_image(path) for path in image_paths]
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": user_message,
                "images": encoded_images,
            },
        ]

        last_error: Exception | None = None
        for attempt in range(self._retries + 1):
            try:
                async with self._semaphore:
                    started = time.perf_counter()
                    response = await self._client.chat(
                        model=self._model,
                        messages=messages,
                        options={
                            "temperature": self._temperature,
                            "top_p": self._top_p,
                            "num_ctx": self._num_ctx,
                        },
                    )
                    elapsed_ms = (time.perf_counter() - started) * 1000
                return ProviderResponse(
                    raw_output=str(response["message"]["content"]),
                    elapsed_ms=elapsed_ms,
                )
            except Exception as exc:  # provider-specific transport failures
                last_error = exc
                if attempt >= self._retries:
                    break
                await asyncio.sleep(self._retry_delay_s * (attempt + 1))

        raise RuntimeError(f"Ollama request failed: {last_error}")


class ProviderFactory:
    """Create the selected model-provider strategy in one place."""

    @staticmethod
    def create(
        provider: str,
        *,
        host: str,
        model: str,
        temperature: float,
        top_p: float,
        num_ctx: int,
        timeout_s: float,
        retries: int,
        retry_delay_s: float,
        semaphore: asyncio.Semaphore,
    ) -> ModelProvider:
        name = provider.strip().lower()
        if name == "ollama":
            return OllamaVisionAdapter(
                host=host,
                model=model,
                temperature=temperature,
                top_p=top_p,
                num_ctx=num_ctx,
                timeout_s=timeout_s,
                retries=retries,
                retry_delay_s=retry_delay_s,
                semaphore=semaphore,
            )
        raise ValueError(
            f"Unsupported model provider: {provider!r}. "
            "Add a new adapter and register it in ProviderFactory."
        )
