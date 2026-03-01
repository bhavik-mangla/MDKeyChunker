"""Unified LLM client supporting OpenAI, Anthropic, and OpenAI-compatible endpoints."""
import json
import re
import time
import logging
from typing import Any
from .config import Config

log = logging.getLogger(__name__)


class LLMClient:
    def __init__(self, config: Config):
        self.config = config
        self.provider = config.llm_provider.lower()
        self._client: Any = None
        self._init_client()

    def _init_client(self) -> None:
        if self.provider in ("openai", "openai_compatible"):
            from openai import OpenAI
            kwargs: dict = {"api_key": self.config.llm_api_key or "not-needed"}
            if self.config.llm_base_url:
                kwargs["base_url"] = self.config.llm_base_url
            self._client = OpenAI(**kwargs)
            return
        if self.provider == "anthropic":
            from anthropic import Anthropic
            self._client = Anthropic(api_key=self.config.llm_api_key)
            return

    def call(self, prompt: str, max_tokens: int = 1000, retries: int = 4) -> str:
        """Make a single LLM call with exponential-backoff retries. Returns raw text."""
        if not self._client:
            raise RuntimeError(
                "LLM client not initialized. Check LLM_PROVIDER and LLM_API_KEY."
            )
        for attempt in range(retries + 1):
            try:
                if self.provider in ("openai", "openai_compatible"):
                    r = self._client.chat.completions.create(
                        model=self.config.llm_model,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=max_tokens,
                        temperature=0.1,
                        timeout=120,  # local models can be slow
                    )
                    return str(r.choices[0].message.content)
                elif self.provider == "anthropic":
                    r = self._client.messages.create(
                        model=self.config.llm_model,
                        max_tokens=max_tokens,
                        temperature=0.1,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    return str(r.content[0].text)
                else:
                    raise RuntimeError(f"Unknown provider: {self.provider}")
            except RuntimeError:
                raise
            except Exception as e:
                if attempt < retries:
                    wait = min(2 ** attempt, 30)  # cap at 30s; give local model time to recover
                    log.warning(
                        "LLM call attempt %d failed, retrying in %ds: %s",
                        attempt + 1, wait, e,
                    )
                    time.sleep(wait)
                else:
                    raise
        raise RuntimeError(
            "LLM call failed after all retries")  # pragma: no cover

    def call_json(self, prompt: str, max_tokens: int = 1000) -> dict | None:
        """Make LLM call and parse JSON response.

        For OpenAI/OpenAI-compatible providers, uses response_format=json_object
        when available for more reliable JSON output, falling back to plain call.
        """
        if self.provider in ("openai", "openai_compatible"):
            try:
                if not self._client:
                    raise RuntimeError("LLM client not initialized.")
                r = self._client.chat.completions.create(
                    model=self.config.llm_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=0.1,
                    response_format={"type": "json_object"},
                )
                return self._parse_json(str(r.choices[0].message.content))
            except Exception:
                pass  # Fall back to regular call with text parsing
        raw = self.call(prompt, max_tokens)
        return self._parse_json(raw)

    @staticmethod
    def _extract_json_object(text: str) -> str | None:
        """Extract the first balanced JSON object from text using bracket counting."""
        start = text.find('{')
        if start == -1:
            return None
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            c = text[i]
            if escape:
                escape = False
                continue
            if c == '\\' and in_string:
                escape = True
                continue
            if c == '"' and not escape:
                in_string = not in_string
                continue
            if in_string:
                continue
            if c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
        return None

    @staticmethod
    def _parse_json(text: str) -> dict | None:
        """Extract JSON from LLM response, handling markdown code blocks and preamble."""
        # Direct parse
        try:
            return dict(json.loads(text))
        except (json.JSONDecodeError, TypeError):
            pass
        # From code block
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if m:
            try:
                return dict(json.loads(m.group(1)))
            except (json.JSONDecodeError, TypeError):
                pass
        # Balanced bracket extraction (handles nested structures)
        obj_str = LLMClient._extract_json_object(text)
        if obj_str:
            try:
                return dict(json.loads(obj_str))
            except (json.JSONDecodeError, TypeError):
                pass
        log.warning("Failed to parse JSON from LLM response: %s", text[:200])
        return None
