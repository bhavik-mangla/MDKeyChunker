"""One completion interface over Ollama (local), OpenRouter (paid) and Gemini (free).

    from llm_api import complete
    r = complete("openrouter:openai/gpt-oss-120b@deepinfra/bf16", prompt,
                 json_mode=True, max_tokens=400, temperature=0, seed=0, reasoning="low")
    r["text"], r["parsed_json"], r["in_tokens"], r["out_tokens"], r["cost_usd"], r["provider"]

Model specs
    ollama:<model>                       e.g. ollama:qwen2.5:7b
    openrouter:<model>@<slug>[,<slug>]   e.g. openrouter:meta-llama/llama-3.3-70b-instruct@deepinfra/turbo
    gemini:<model>                       e.g. gemini:gemini-3.1-flash-lite

OpenRouter calls are always pinned (provider.order + allow_fallbacks=false), refuse
data collection, ask for usage.cost, and never route to Together or Fireworks
(their terms bar benchmarking; voxparity billing doc). An unpinned OpenRouter spec
is rejected: an aggregator may otherwise serve one model id from different
quantizations between runs.

Every call (success or failure) appends one row to a JSONL spend ledger
(results/spend_ledger.jsonl by default): time, backend, model, upstream provider,
sha256 of the prompt, tokens, cost, latency, attempts, status. Prompt text and API
keys are never written. Before each paid call the ledger total plus in-flight
reservations is compared with a hard cap (default $5.50, env LLM_BUDGET_USD); the
call is refused with BudgetExceeded when it would cross the cap.

Retries (patterns proven in voxparity/adapters/openrouter.py): transport errors,
timeouts, HTTP 429/5xx, and HTTP-200 bodies that wrap an upstream failure
({"error": {"code": 502, ...}}) are retried with exponential backoff and jitter,
honouring Retry-After. Other 4xx errors fail at once. Gemini uses one free key
(GEMINI_API_KEYS index 0) and never rotates keys.

Secrets are read inside Python from the MDKeyChunker .env (python-dotenv), or from
the process environment, and are never logged or echoed in exceptions.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import random
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import requests

ROOT = Path(__file__).parent
DEFAULT_LEDGER = ROOT / "results" / "spend_ledger.jsonl"
DEFAULT_ENV_FILE = Path(os.environ.get("MDK_ENV_FILE", "/Users/bhavikmangla/Developer/MDKeyChunker/.env"))
DEFAULT_BUDGET_USD = float(os.environ.get("LLM_BUDGET_USD", "5.50"))
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
# Ollama reloads the model whenever num_ctx changes; keep one value per run (see build.py).
NUM_CTX = int(os.environ.get("NUM_CTX", "16384"))
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
BANNED_PROVIDERS = ["together", "fireworks"]
RETRY_STATUSES = {408, 429, 500, 502, 503, 504, 520, 522, 524, 529}


class LLMError(RuntimeError):
    """A call failed after all retries, or with a non-retryable error."""


class BudgetExceeded(LLMError):
    """The spend ledger says this call could cross the budget cap."""


class _Retryable(Exception):
    def __init__(self, msg: str, retry_after: float | None = None, payload: dict | None = None):
        super().__init__(msg)
        self.retry_after = retry_after
        self.payload = payload


# --------------------------------------------------------------------------- specs

@dataclass(frozen=True)
class ModelSpec:
    backend: str            # ollama | openrouter | gemini
    model: str
    providers: tuple[str, ...] = ()

    @classmethod
    def parse(cls, spec: str) -> "ModelSpec":
        backend, sep, rest = spec.partition(":")
        if not sep or backend not in ("ollama", "openrouter", "gemini") or not rest:
            raise ValueError(f"bad model spec {spec!r}; want ollama:|openrouter:|gemini:<model>")
        providers: tuple[str, ...] = ()
        if backend == "openrouter":
            model, at, pin = rest.partition("@")
            providers = tuple(p.strip() for p in pin.split(",") if p.strip()) if at else ()
            rest = model
        return cls(backend, rest, providers)

    def __str__(self) -> str:
        pin = "@" + ",".join(self.providers) if self.providers else ""
        return f"{self.backend}:{self.model}{pin}"


def model_family(spec: str | ModelSpec) -> str:
    """Coarse model family, used to keep the judge out of the generator's family."""
    m = (spec.model if isinstance(spec, ModelSpec) else ModelSpec.parse(spec).model).lower()
    for fam, pats in (
        ("qwen", ("qwen", "qwq")), ("llama", ("llama",)), ("gpt-oss", ("gpt-oss",)),
        ("openai-gpt", ("gpt-", "/o1", "/o3", "/o4")), ("gemini", ("gemini", "gemma")),
        ("deepseek", ("deepseek",)), ("mistral", ("mistral", "mixtral", "ministral")),
        ("claude", ("claude",)), ("grok", ("grok",)), ("glm", ("glm",)), ("kimi", ("kimi", "moonshot")),
    ):
        if any(p in m for p in pats):
            return fam
    return m.split("/")[-1].split(":")[0]


# --------------------------------------------------------------------------- secrets

_SECRETS: dict[str, str] | None = None
_SECRETS_LOCK = threading.Lock()


def _secret(name: str) -> str:
    """Read one variable from the .env file (or the environment). Never logged."""
    global _SECRETS
    with _SECRETS_LOCK:
        if _SECRETS is None:
            vals: dict[str, str] = {}
            if DEFAULT_ENV_FILE.exists():
                from dotenv import dotenv_values
                vals = {k: v for k, v in dotenv_values(DEFAULT_ENV_FILE).items() if v}
            _SECRETS = vals
    v = os.environ.get(name) or _SECRETS.get(name)
    if not v:
        raise LLMError(f"{name} is not set (expected in {DEFAULT_ENV_FILE} or the environment)")
    return v


def _gemini_key() -> str:
    # Index 0 only; key rotation to stretch free quota is not authorised.
    raw = _secret("GEMINI_API_KEYS")
    keys = [k.strip() for k in raw.split(",") if k.strip()]
    if not keys:
        raise LLMError("GEMINI_API_KEYS is empty")
    return keys[0]


# --------------------------------------------------------------------------- ledger

class Ledger:
    """Append-only JSONL spend log, safe across threads (lock) and processes (flock).

    The running total is read incrementally from the file, so several processes
    sharing one ledger all see each other's spend before each paid call.
    """

    def __init__(self, path: str | Path = DEFAULT_LEDGER):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._offset = 0
        self._total = 0.0
        self._rows = 0

    def append(self, row: dict) -> None:
        line = json.dumps(row, sort_keys=True) + "\n"
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                try:
                    f.write(line)
                    f.flush()
                    os.fsync(f.fileno())
                finally:
                    fcntl.flock(f, fcntl.LOCK_UN)

    def total_spend(self) -> float:
        """Cumulative cost_usd over every row (rows without a cost count as 0)."""
        with self._lock:
            if not self.path.exists():
                return 0.0
            with open(self.path, "r", encoding="utf-8") as f:
                fcntl.flock(f, fcntl.LOCK_SH)
                try:
                    f.seek(self._offset)
                    chunk = f.read()
                finally:
                    fcntl.flock(f, fcntl.LOCK_UN)
            # Only consume complete lines; a partial tail is re-read next time.
            end = chunk.rfind("\n") + 1
            for line in chunk[:end].splitlines():
                if not line.strip():
                    continue
                try:
                    c = json.loads(line).get("cost_usd")
                except json.JSONDecodeError:
                    continue
                if isinstance(c, (int, float)):
                    self._total += float(c)
                self._rows += 1
            self._offset += len(chunk[:end].encode("utf-8"))
            return self._total


class BudgetGuard:
    def __init__(self, ledger: Ledger, cap_usd: float = DEFAULT_BUDGET_USD, reserve_usd: float = 0.005):
        self.ledger = ledger
        self.cap = cap_usd
        self.reserve = reserve_usd
        self._inflight = 0.0
        self._lock = threading.Lock()

    def acquire(self) -> float:
        with self._lock:
            spent = self.ledger.total_spend()
            if spent + self._inflight + self.reserve > self.cap:
                raise BudgetExceeded(
                    f"budget cap ${self.cap:.4f}: spent ${spent:.4f} + in flight "
                    f"${self._inflight:.4f} + reserve ${self.reserve:.4f} would exceed it")
            self._inflight += self.reserve
            return self.reserve

    def release(self, amount: float) -> None:
        with self._lock:
            self._inflight = max(0.0, self._inflight - amount)


# --------------------------------------------------------------------------- client

def extract_json(text: str) -> Any:
    """Parse a JSON reply, tolerating code fences or prose around one object/array."""
    if text is None:
        return None
    s = text.strip()
    m = re.match(r"^```(?:json)?\s*(.*?)\s*```$", s, re.S)
    if m:
        s = m.group(1)
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    for open_, close in (("{", "}"), ("[", "]")):
        i, j = s.find(open_), s.rfind(close)
        if 0 <= i < j:
            try:
                return json.loads(s[i:j + 1])
            except json.JSONDecodeError:
                continue
    return None


def _retry_after(resp: Any) -> float | None:
    try:
        v = resp.headers.get("retry-after") or resp.headers.get("Retry-After")
        return float(v) if v is not None else None
    except (TypeError, ValueError, AttributeError):
        return None


@dataclass
class LLMClient:
    ledger: Ledger = field(default_factory=Ledger)
    budget_usd: float = DEFAULT_BUDGET_USD
    reserve_usd: float = 0.005       # per in-flight paid call; covers one call's worst case here
    max_retries: int = 5
    backoff_base: float = 3.0        # seconds; 3, 6, 12, 24, 48 (+ jitter), as in voxparity
    backoff_cap: float = 60.0
    timeout_s: float = 180.0
    http: Any = None                 # anything with .post(url, json=, headers=, timeout=)
    sleep: Callable[[float], None] = time.sleep
    secret: Callable[[str], str] = _secret
    gemini_key: Callable[[], str] = _gemini_key

    def __post_init__(self) -> None:
        self.http = self.http or requests.Session()
        self.guard = BudgetGuard(self.ledger, self.budget_usd, self.reserve_usd)

    # -- public ------------------------------------------------------------------
    def complete(self, model_spec: str, prompt: str, *, json_mode: bool = False, max_tokens: int = 512,
                 temperature: float = 0.0, seed: int | None = 0, reasoning: Any = None,
                 system: str | None = None, tag: str = "") -> dict:
        spec = ModelSpec.parse(model_spec)
        fn = {"ollama": self._ollama, "openrouter": self._openrouter, "gemini": self._gemini}[spec.backend]
        reserved = self.guard.acquire() if spec.backend == "openrouter" else 0.0
        t0 = time.time()
        row = {"ts": round(t0, 3), "backend": spec.backend, "model": spec.model, "spec": str(spec),
               "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
               "prompt_chars": len(prompt), "json_mode": json_mode, "max_tokens": max_tokens,
               "temperature": temperature, "seed": seed, "reasoning": reasoning, "tag": tag}
        attempts = 0
        try:
            last_err: Exception | None = None
            for attempt in range(self.max_retries + 1):
                attempts = attempt + 1
                try:
                    out = fn(spec, prompt, json_mode, max_tokens, temperature, seed, reasoning, system)
                    break
                except _Retryable as e:
                    last_err = e
                    if e.payload:  # an HTTP-200 error body may still be billed; record it
                        self._log_failed_attempt(row, e)
                    if attempt == self.max_retries:
                        raise LLMError(f"{spec}: gave up after {attempts} attempts: {e}") from None
                    delay = min(self.backoff_cap, self.backoff_base * 2 ** attempt) * (1 + 0.25 * random.random())
                    if e.retry_after is not None:
                        delay = max(delay, min(e.retry_after, self.backoff_cap))
                    self.sleep(delay)
            else:  # pragma: no cover
                raise LLMError(f"{spec}: unreachable ({last_err})")
        except Exception as e:
            self.ledger.append({**row, "status": "error", "error": _redact(str(e))[:300],
                                "attempts": attempts, "latency_s": round(time.time() - t0, 3),
                                "cost_usd": 0.0})
            raise
        finally:
            self.guard.release(reserved)
        latency = time.time() - t0
        out["latency_s"] = round(latency, 3)
        out["model"] = out.get("model") or spec.model
        out["parsed_json"] = extract_json(out["text"]) if json_mode else None
        out["attempts"] = attempts
        self.ledger.append({**row, "status": "ok", "attempts": attempts, "latency_s": out["latency_s"],
                            "provider": out.get("provider"), "served_model": out.get("model"),
                            "in_tokens": out.get("in_tokens"), "out_tokens": out.get("out_tokens"),
                            "reasoning_tokens": out.get("reasoning_tokens"), "cost_usd": out.get("cost_usd"),
                            "cost_missing": out.get("cost_usd") is None, "finish_reason": out.get("finish_reason"),
                            "generation_id": out.get("generation_id"),
                            "json_ok": (out["parsed_json"] is not None) if json_mode else None})
        return out

    def _log_failed_attempt(self, row: dict, e: _Retryable) -> None:
        usage = (e.payload or {}).get("usage") or {}
        cost = usage.get("cost")
        self.ledger.append({**row, "status": "retry", "error": _redact(str(e))[:300],
                            "provider": (e.payload or {}).get("provider"),
                            "cost_usd": float(cost) if isinstance(cost, (int, float)) else 0.0})

    # -- backends ------------------------------------------------------------------
    def _post(self, url: str, body: dict, headers: dict | None = None) -> Any:
        try:
            return self.http.post(url, json=body, headers=headers or {}, timeout=self.timeout_s)
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError, TimeoutError,
                ConnectionError) as e:
            raise _Retryable(f"transport: {type(e).__name__}") from None

    @staticmethod
    def _json(resp: Any) -> dict:
        try:
            data = resp.json()
        except ValueError:
            data = None
        return data if isinstance(data, dict) else {}

    def _check_http(self, resp: Any, what: str) -> dict:
        data = self._json(resp)
        if resp.status_code in RETRY_STATUSES:
            raise _Retryable(f"{what} HTTP {resp.status_code}: {_err_msg(data, resp)}", _retry_after(resp), data)
        if resp.status_code != 200:
            raise LLMError(f"{what} HTTP {resp.status_code}: {_err_msg(data, resp)}")
        return data

    def _ollama(self, spec, prompt, json_mode, max_tokens, temperature, seed, reasoning, system) -> dict:
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        think: Any = False if reasoning in (None, False, "off", "none") else (
            True if reasoning is True else reasoning)
        opts = {"temperature": temperature, "num_predict": max_tokens, "num_ctx": NUM_CTX}
        if seed is not None:
            opts["seed"] = seed
        body = {"model": spec.model, "stream": False, "think": think, "messages": msgs, "options": opts}
        if json_mode:
            body["format"] = "json"
        resp = self._post(f"{OLLAMA_URL}/api/chat", body)
        data = self._check_http(resp, "ollama")
        if "error" in data:
            raise LLMError(f"ollama: {str(data['error'])[:200]}")
        return {"text": (data.get("message") or {}).get("content", ""),
                "in_tokens": data.get("prompt_eval_count", 0), "out_tokens": data.get("eval_count", 0),
                "reasoning_tokens": None, "cost_usd": 0.0, "provider": "ollama-local",
                "model": data.get("model", spec.model), "finish_reason": data.get("done_reason")}

    def _openrouter(self, spec, prompt, json_mode, max_tokens, temperature, seed, reasoning, system) -> dict:
        if not spec.providers:
            raise LLMError(f"{spec}: OpenRouter specs must pin a provider (model@slug)")
        if any(p.split("/")[0].lower() in BANNED_PROVIDERS for p in spec.providers):
            raise LLMError(f"{spec}: Together/Fireworks are excluded (terms bar benchmarking)")
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        body: dict[str, Any] = {
            "model": spec.model, "messages": msgs, "max_tokens": max_tokens, "temperature": temperature,
            "provider": {"order": list(spec.providers), "allow_fallbacks": False,
                         "data_collection": "deny", "ignore": BANNED_PROVIDERS},
            "usage": {"include": True},
        }
        if seed is not None:
            body["seed"] = seed
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        r = _openrouter_reasoning(reasoning)
        if r is not None:
            body["reasoning"] = r
        if spec.model.startswith("mistralai/"):
            body["top_p"] = 1.0  # Mistral rejects greedy decoding unless top_p == 1 (voxparity)
        headers = {"Authorization": f"Bearer {self.secret('OPENROUTER_API_KEY')}",
                   "HTTP-Referer": "https://github.com/mdkeychunker", "X-Title": "MDKeyChunker-eval"}
        resp = self._post(OPENROUTER_URL, body, headers)
        data = self._check_http(resp, "openrouter")
        err = data.get("error")
        if err:
            # OpenRouter wraps some upstream failures in HTTP-200 bodies (voxparity D074).
            code = int((err or {}).get("code") or 0) if isinstance(err, dict) else 0
            msg = f"openrouter in-body error {code}: {_err_msg(data, resp)}"
            if code >= 500 or code in (408, 429) or code == 0:
                raise _Retryable(msg, None, data)
            raise LLMError(msg)
        try:
            choice = data["choices"][0]
            msg_ = choice.get("message") or {}
        except (KeyError, IndexError, TypeError):
            raise _Retryable(f"openrouter: unexpected body shape {str(data)[:200]}", None, data) from None
        cerr = choice.get("error")
        if cerr:  # per-choice upstream error, also seen with HTTP 200
            raise _Retryable(f"openrouter choice error: {str(cerr)[:200]}", None, data)
        usage = data.get("usage") or {}
        details = usage.get("completion_tokens_details") or {}
        cost = usage.get("cost")
        return {"text": msg_.get("content") or "",
                "in_tokens": usage.get("prompt_tokens", 0), "out_tokens": usage.get("completion_tokens", 0),
                "reasoning_tokens": details.get("reasoning_tokens"),
                "cost_usd": float(cost) if isinstance(cost, (int, float)) else None,
                "provider": data.get("provider"), "model": data.get("model", spec.model),
                "finish_reason": choice.get("finish_reason"), "generation_id": data.get("id")}

    def _gemini(self, spec, prompt, json_mode, max_tokens, temperature, seed, reasoning, system) -> dict:
        gen: dict[str, Any] = {"temperature": temperature, "maxOutputTokens": max_tokens}
        if seed is not None:
            gen["seed"] = seed
        if json_mode:
            gen["responseMimeType"] = "application/json"
        tc = _gemini_thinking(spec.model, reasoning)
        if tc is not None:
            gen["thinkingConfig"] = tc
        body: dict[str, Any] = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
                                "generationConfig": gen}
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        # Key in a header, never in the URL, so it cannot leak into exception text.
        resp = self._post(GEMINI_URL.format(model=spec.model), body,
                          {"x-goog-api-key": self.gemini_key(), "Content-Type": "application/json"})
        data = self._check_http(resp, "gemini")
        cands = data.get("candidates") or []
        if not cands:
            fb = data.get("promptFeedback")
            raise LLMError(f"gemini: no candidates ({str(fb)[:200]})")
        parts = (cands[0].get("content") or {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        um = data.get("usageMetadata") or {}
        return {"text": text, "in_tokens": um.get("promptTokenCount", 0),
                "out_tokens": um.get("candidatesTokenCount", 0) + um.get("thoughtsTokenCount", 0),
                "reasoning_tokens": um.get("thoughtsTokenCount"), "cost_usd": 0.0,
                "provider": "google-ai-studio-free", "model": data.get("modelVersion", spec.model),
                "finish_reason": cands[0].get("finishReason")}


def _openrouter_reasoning(reasoning: Any) -> dict | None:
    """None -> provider default; "low"/"medium"/"high"/"minimal" -> effort;
    "off"/False -> disabled; "exclude" -> reason but omit it from the reply; dict -> as is."""
    if reasoning is None:
        return None
    if isinstance(reasoning, dict):
        return reasoning
    if reasoning is False or reasoning in ("off", "none", "disabled"):
        return {"enabled": False}
    if reasoning == "exclude":
        return {"exclude": True}
    if reasoning in ("minimal", "low", "medium", "high"):
        return {"effort": reasoning}
    raise ValueError(f"unknown reasoning setting {reasoning!r}")


def _gemini_thinking(model: str, reasoning: Any) -> dict | None:
    if reasoning is None:
        return None
    if isinstance(reasoning, dict):
        return reasoning
    off = reasoning is False or reasoning in ("off", "none", "disabled")
    if model.startswith("gemini-2.5"):
        budgets = {"minimal": 0, "low": 1024, "medium": 4096, "high": 16384}
        return {"thinkingBudget": 0 if off else budgets.get(reasoning, 1024)}
    return {"thinkingLevel": "minimal" if off else reasoning}


def _err_msg(data: dict, resp: Any) -> str:
    err = data.get("error") if isinstance(data, dict) else None
    if isinstance(err, dict):
        meta = err.get("metadata") or {}
        extra = f" [{meta.get('provider_name')}]" if meta.get("provider_name") else ""
        return _redact(f"{err.get('message', '')}{extra}")[:300]
    if err:
        return _redact(str(err))[:300]
    return _redact(getattr(resp, "text", "") or "")[:300]


def _redact(s: str) -> str:
    return re.sub(r"(sk-or-[A-Za-z0-9-]{8,}|AIza[0-9A-Za-z_-]{20,}|Bearer\s+\S+)", "[REDACTED]", s)


# --------------------------------------------------------------------------- module API

_DEFAULT: LLMClient | None = None
_DEFAULT_LOCK = threading.Lock()


def default_client() -> LLMClient:
    global _DEFAULT
    with _DEFAULT_LOCK:
        if _DEFAULT is None:
            _DEFAULT = LLMClient()
        return _DEFAULT


def complete(model_spec: str, prompt: str, *, json_mode: bool = False, max_tokens: int = 512,
             temperature: float = 0.0, seed: int | None = 0, reasoning: Any = None, **kw) -> dict:
    """Returns {text, parsed_json, in_tokens, out_tokens, cost_usd, provider, latency_s, model, ...}."""
    return default_client().complete(model_spec, prompt, json_mode=json_mode, max_tokens=max_tokens,
                                     temperature=temperature, seed=seed, reasoning=reasoning, **kw)
