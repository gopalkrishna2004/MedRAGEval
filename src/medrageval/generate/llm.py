"""Chat-completions client for OpenRouter, with JSON parsing and a response cache.

OpenRouter speaks the OpenAI API, so the `openai` SDK is pointed at its base
URL. Every call is cached on disk by (model, temperature, messages): a rerun
with the same config and seed makes no calls and writes the same dataset.
Transport errors (429, 5xx, timeouts) are retried by the SDK; a response that
is not JSON is re-asked once and then raised, so a wrong model or key stops
the run instead of producing an empty dataset.
"""

import hashlib
import json
import os
import re
import threading
from collections import Counter
from collections.abc import Callable
from pathlib import Path

from dotenv import load_dotenv

ENV_KEY = "OPENROUTER_API_KEY"
_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.S)


class LLMJsonError(RuntimeError):
    """The model did not return a JSON object, even after being asked again."""


class LLMError(RuntimeError):
    """The API call failed after the SDK's retries."""


def parse_json_object(text: str) -> dict:
    text = _FENCE.sub("", text.strip())
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise
        obj = json.loads(text[start:end + 1])
    if not isinstance(obj, dict):
        raise json.JSONDecodeError("not an object", text, 0)
    return obj


class LLM:
    def __init__(self, model: str, base_url: str, api_key: str, temperature: float, max_tokens: int,
                 cache_dir: Path | None = None, timeout: float = 120.0, max_retries: int = 5):
        from openai import OpenAI
        self.model, self.temperature, self.max_tokens = model, temperature, max_tokens
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=max_retries)
        self.stats = Counter()
        self._lock = threading.Lock()

    def _cache_path(self, key: str) -> Path | None:
        return self.cache_dir / key[:2] / f"{key}.json" if self.cache_dir else None

    def _count(self, **kw) -> None:
        with self._lock:
            self.stats.update(kw)

    def _call(self, messages: list[dict], _retry: bool = False) -> tuple[str, dict]:
        from openai import APIError
        try:
            r = self.client.chat.completions.create(
                model=self.model, messages=messages, temperature=self.temperature,
                max_tokens=self.max_tokens, response_format={"type": "json_object"})
        except APIError as e:
            raise LLMError(f"{self.model}: {e}") from e
        usage = {"prompt_tokens": getattr(r.usage, "prompt_tokens", 0) or 0,
                 "completion_tokens": getattr(r.usage, "completion_tokens", 0) or 0} if r.usage else {}
        text = (r.choices[0].message.content or "") if r.choices else ""
        self._count(calls=1, **usage)
        if not text.strip() and not _retry:
            self._count(empty_responses=1)
            return self._call(messages, _retry=True)
        return text, usage

    def complete_json(self, messages: list[dict], tag: str = "") -> dict:
        key = hashlib.sha256(json.dumps({"model": self.model, "temperature": self.temperature, "messages": messages},
                                        sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        path = self._cache_path(key)
        if path and path.exists():
            self._count(cache_hits=1)
            return parse_json_object(json.loads(path.read_text())["response"])

        text, usage = self._call(messages)
        try:
            obj = parse_json_object(text)
        except (json.JSONDecodeError, ValueError):
            self._count(bad_json=1)
            retry = messages + [{"role": "assistant", "content": text},
                                {"role": "user", "content": "That was not valid JSON. Return only the JSON object."}]
            text, usage2 = self._call(retry)
            try:
                obj = parse_json_object(text)
            except (json.JSONDecodeError, ValueError) as e:
                raise LLMJsonError(f"{tag}: model returned no JSON object: {text[:200]!r}") from e
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(f".{threading.get_ident()}.tmp")
            tmp.write_text(json.dumps({"tag": tag, "model": self.model, "temperature": self.temperature,
                                       "response": text, "usage": usage}, ensure_ascii=False))
            os.replace(tmp, path)
        return obj


class FakeLLM:
    """Test double: answers from a callable (messages, tag) -> dict, or a list of dicts in order."""

    def __init__(self, responses: Callable[[list[dict], str], dict] | list[dict]):
        self.responses = responses
        self.calls: list[tuple[list[dict], str]] = []
        self.stats = Counter()
        self.model, self.temperature = "fake", 0.0
        self._lock = threading.Lock()

    def complete_json(self, messages: list[dict], tag: str = "") -> dict:
        with self._lock:
            self.calls.append((messages, tag))
            self.stats["calls"] += 1
            if callable(self.responses):
                return self.responses(messages, tag)
            if not self.responses:
                raise LLMJsonError(f"{tag}: no canned response left")
            return self.responses.pop(0)


def api_key(env_file: Path | None = None) -> str:
    load_dotenv(env_file) if env_file else load_dotenv()
    key = os.environ.get(ENV_KEY, "").strip()
    if not key:
        raise RuntimeError(f"{ENV_KEY} is not set; add it to .env (see .env.example) or export it")
    return key


def make_llm(section: dict, cfg: dict, use_cache: bool = True) -> LLM:
    """A client for a stage with its own config section (`facts`): its model, temperature and token limit,
    with the endpoint and response cache of the `generate` section."""
    key = api_key()
    cache = Path(cfg["cache_dir"]) if use_cache and cfg.get("cache_dir") else None
    return LLM(section.get("model") or cfg["model"], cfg["base_url"], key, section["temperature"], section["max_tokens"], cache)


def make_llms(cfg: dict, use_cache: bool = True) -> tuple[LLM, LLM]:
    """(generator, judge) clients from the `generate` config section."""
    key = api_key()
    cache = Path(cfg["cache_dir"]) if use_cache and cfg.get("cache_dir") else None
    gen = LLM(cfg["model"], cfg["base_url"], key, cfg["temperature"], cfg["max_tokens"], cache)
    judge = LLM(cfg.get("judge_model") or cfg["model"], cfg["base_url"], key, cfg["judge_temperature"], cfg["max_tokens"], cache)
    return gen, judge
