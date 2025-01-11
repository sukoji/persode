"""Optional LLM / image adapters.

The paper uses GPT-4o (text) and DALL·E 3 (image). To keep the reproduction
runnable with no keys, everything degrades gracefully:

- :class:`OpenAIClient` talks to the real API when ``OPENAI_API_KEY`` is set and
  the ``openai`` package is installed.
- :class:`OfflineLLM` is a deterministic stub used everywhere by default.

Both expose the same tiny interface (:meth:`complete`), so the analyzer, agent
and templates never need to know which one they hold.
"""

from __future__ import annotations

import os
from typing import Optional, Protocol


class LLMClient(Protocol):
    def complete(self, system: str, user: str, temperature: float = 0.7,
                 json_mode: bool = False) -> str: ...


class OfflineLLM:
    """Deterministic stub. Echoes a compact, structured response.

    It never fails, so pipelines relying on an LLM still run in CI / offline.
    Real text quality comes from :class:`OpenAIClient`.
    """

    def complete(self, system: str, user: str, temperature: float = 0.7,
                 json_mode: bool = False) -> str:
        if json_mode:
            return "{}"
        return f"[offline-llm] {user.strip()[:280]}"


class OpenAIClient:
    """Thin GPT-4o wrapper (used only when a key is available)."""

    def __init__(self, model: str = "gpt-4o", api_key: Optional[str] = None) -> None:
        from openai import OpenAI  # lazy import; optional dependency

        self.model = model
        self._client = OpenAI(api_key=api_key or os.environ["OPENAI_API_KEY"])

    def complete(self, system: str, user: str, temperature: float = 0.7,
                 json_mode: bool = False) -> str:
        kwargs = {}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = self._client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            **kwargs,
        )
        return resp.choices[0].message.content or ""

    def generate_image(self, prompt: str, size: str = "1024x1024") -> str:
        """Return a DALL·E 3 image URL for ``prompt``."""
        resp = self._client.images.generate(
            model="dall-e-3", prompt=prompt, size=size, n=1,
        )
        return resp.data[0].url


def get_llm(prefer_openai: bool = True) -> LLMClient:
    """Return the best available client (OpenAI if a key exists, else offline)."""
    if prefer_openai and os.environ.get("OPENAI_API_KEY"):
        try:
            return OpenAIClient()
        except Exception:
            pass
    return OfflineLLM()
