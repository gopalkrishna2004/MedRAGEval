"""A chat-completions client for the answering model, with a disk cache.

Any OpenAI-compatible server works (vLLM by default). Replies are cached by
(model, temperature, messages), so a rerun of the same questions over the
same passages makes no calls and gives the same answers.
"""

import hashlib
import json
import os
import threading
from collections import Counter
from pathlib import Path

from medrag.config import env_or


class LLMError(RuntimeError):
    pass


class ChatLLM:
    def __init__(self, model: str, base_url: str, api_key: str, temperature: float = 0.0, max_tokens: int = 512,
                 cache_dir: str | Path | None = None, timeout: float = 300.0, max_retries: int = 3):
        from openai import OpenAI
        self.model, self.temperature, self.max_tokens = model, temperature, max_tokens
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=max_retries)
        self.stats: Counter = Counter()
        self._lock = threading.Lock()

    def _key(self, messages: list[dict]) -> str:
        payload = {"model": self.model, "temperature": self.temperature, "messages": messages, "format": "text"}
        return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

    def _call(self, messages: list[dict]) -> tuple[str, dict]:
        from openai import APIError
        try:
            r = self.client.chat.completions.create(model=self.model, messages=messages, temperature=self.temperature, max_tokens=self.max_tokens)
        except APIError as e:
            raise LLMError(f"{self.model}: {e}") from e   # no server address: this text is saved in run files
        text = (r.choices[0].message.content or "") if r.choices else ""
        usage = {"prompt_tokens": getattr(r.usage, "prompt_tokens", 0) or 0,
                 "completion_tokens": getattr(r.usage, "completion_tokens", 0) or 0} if r.usage else {}
        if r.choices and getattr(r.choices[0], "finish_reason", None):
            usage["finish_reason"] = r.choices[0].finish_reason
        with self._lock:
            self.stats.update({"calls": 1, "prompt_tokens": usage.get("prompt_tokens", 0), "completion_tokens": usage.get("completion_tokens", 0)})
        return text, usage

    def complete(self, messages: list[dict], tag: str = "") -> tuple[str, dict]:
        """(reply, usage); usage["cached"] is True when served from disk."""
        key = self._key(messages)
        path = self.cache_dir / key[:2] / f"{key}.json" if self.cache_dir else None
        if path and path.exists():
            rec = json.loads(path.read_text())
            with self._lock:
                self.stats["cache_hits"] += 1
            return rec["response"], {**(rec.get("usage") or {}), "cached": True}
        text, usage = self._call(messages)
        if not text.strip():   # a server hiccup, or a model that spent its tokens thinking: ask once more
            with self._lock:
                self.stats["empty_responses"] += 1
            text, usage = self._call(messages)
        if path and text.strip():
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(f".{threading.get_ident()}.tmp")
            tmp.write_text(json.dumps({"tag": tag, "model": self.model, "temperature": self.temperature, "response": text, "usage": usage}, ensure_ascii=False))
            os.replace(tmp, path)
        return text, usage


def make_llm(section: dict, use_cache: bool = True) -> ChatLLM:
    return ChatLLM(section["model"], section["base_url"], env_or(section, "api_key_env", fallback="EMPTY"),
                   section.get("temperature", 0.0), section.get("max_tokens", 512),
                   section.get("cache_dir") if use_cache else None, section.get("timeout", 300.0), section.get("max_retries", 3))
