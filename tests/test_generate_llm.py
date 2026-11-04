import json

import pytest

from medrageval.generate.llm import ENV_KEY, LLM, FakeLLM, LLMJsonError, api_key, parse_json_object


@pytest.mark.parametrize(
    "text, expected",
    [
        ('{"a": 1}', {"a": 1}),
        ('```json\n{"a": 1}\n```', {"a": 1}),
        ('Here is the item:\n{"question": "q", "n": [1, 2]}\nDone.', {"question": "q", "n": [1, 2]}),
    ],
)
def test_parse_json_object(text, expected):
    assert parse_json_object(text) == expected


@pytest.mark.parametrize("text", ["not json", "[1, 2]", ""])
def test_parse_json_object_rejects(text):
    with pytest.raises((json.JSONDecodeError, ValueError)):
        parse_json_object(text)


def test_fake_llm_records_calls_and_runs_dry():
    llm = FakeLLM([{"a": 1}])
    assert llm.complete_json([{"role": "user", "content": "x"}], "t") == {"a": 1}
    assert llm.calls[0][1] == "t" and llm.stats["calls"] == 1
    with pytest.raises(LLMJsonError):
        llm.complete_json([], "t")


def _llm(tmp_path, answers):
    llm = LLM("test-model", "http://localhost:1/v1", "key", 0.5, 100, cache_dir=tmp_path / "cache")
    llm._call = lambda messages: (answers.pop(0), {"prompt_tokens": 10, "completion_tokens": 5})   # no network
    return llm


def test_cache_round_trip(tmp_path):
    messages = [{"role": "user", "content": "hello"}]
    llm = _llm(tmp_path, ['{"x": 1}'])
    assert llm.complete_json(messages, "a") == {"x": 1}
    assert llm.complete_json(messages, "a") == {"x": 1}          # second call served from disk; no answer left to pop
    assert llm.stats["cache_hits"] == 1
    files = list((tmp_path / "cache").rglob("*.json"))
    assert len(files) == 1 and json.loads(files[0].read_text())["response"] == '{"x": 1}'
    other = _llm(tmp_path, ['{"x": 2}'])
    other.temperature = 0.9                                       # temperature is part of the key
    assert other.complete_json(messages, "a") == {"x": 2}


def test_bad_json_is_asked_again_then_raised(tmp_path):
    llm = _llm(tmp_path, ["garbage", '{"ok": true}'])
    assert llm.complete_json([{"role": "user", "content": "q"}], "t") == {"ok": True}
    llm = _llm(tmp_path, ["garbage", "still garbage"])
    with pytest.raises(LLMJsonError):
        llm.complete_json([{"role": "user", "content": "q2"}], "t")
    assert not list((tmp_path / "cache").rglob("*q2*"))


def test_missing_api_key_fails_loudly(tmp_path, monkeypatch):
    monkeypatch.delenv(ENV_KEY, raising=False)
    with pytest.raises(RuntimeError, match=ENV_KEY):
        api_key(tmp_path / "absent.env")
    (tmp_path / ".env").write_text(f"{ENV_KEY}=sk-test\n")
    assert api_key(tmp_path / ".env") == "sk-test"


def test_json_cache_key_is_unchanged_by_text_mode(tmp_path):
    """Responses cached before text mode existed must still hit."""
    import hashlib
    messages = [{"role": "user", "content": "hello"}]
    llm = _llm(tmp_path, [])
    old = hashlib.sha256(json.dumps({"model": "test-model", "temperature": 0.5, "messages": messages},
                                    sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    assert llm._key(messages) == old and llm._key(messages, "text") != old


def test_text_mode_has_its_own_cache_entries(tmp_path):
    messages = [{"role": "user", "content": "hello"}]
    llm = LLM("test-model", "http://localhost:1/v1", "key", 0.5, 100, cache_dir=tmp_path / "cache")
    sent = []

    def call(messages, _retry=False, json_mode=True):
        sent.append(json_mode)
        return ('{"x": 1}' if json_mode else "plain answer"), {"prompt_tokens": 3}
    llm._call = call
    assert llm.complete_json(messages, "j") == {"x": 1}
    assert llm.complete_text(messages, "t") == ("plain answer", {"prompt_tokens": 3})
    assert llm.complete_text(messages, "t") == ("plain answer", {"prompt_tokens": 3, "cached": True})
    assert sent == [True, False] and len(list((tmp_path / "cache").rglob("*.json"))) == 2


def test_empty_text_reply_is_not_cached(tmp_path):
    llm = LLM("test-model", "http://localhost:1/v1", "key", 0.5, 100, cache_dir=tmp_path / "cache")
    llm._call = lambda messages, json_mode=True: ("", {})
    assert llm.complete_text([{"role": "user", "content": "q"}]) == ("", {})
    assert not list((tmp_path / "cache").rglob("*.json"))


def test_text_mode_sends_no_response_format():
    from types import SimpleNamespace
    sent = {}

    class Completions:
        def create(self, **kw):
            sent.clear()
            sent.update(kw)
            content = '{"a": 1}' if "response_format" in kw else "hi"
            return SimpleNamespace(usage=None, choices=[SimpleNamespace(message=SimpleNamespace(content=content))])
    llm = LLM("test-model", "http://localhost:1/v1", "key", 0.0, 100)
    llm.client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    assert llm.complete_text([{"role": "user", "content": "q"}]) == ("hi", {})
    assert "response_format" not in sent
    assert llm.complete_json([{"role": "user", "content": "q"}]) == {"a": 1}
    assert sent["response_format"] == {"type": "json_object"}


def test_reasoning_setting_is_sent_and_keys_the_cache():
    from types import SimpleNamespace
    sent = {}

    class Completions:
        def create(self, **kw):
            sent.clear()
            sent.update(kw)
            return SimpleNamespace(usage=None, choices=[SimpleNamespace(message=SimpleNamespace(content='{"a": 1}'))])
    messages = [{"role": "user", "content": "q"}]
    plain = LLM("test-model", "http://localhost:1/v1", "key", 0.0, 100)
    off = LLM("test-model", "http://localhost:1/v1", "key", 0.0, 100, reasoning={"enabled": False})
    for llm in (plain, off):
        llm.client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    plain.complete_json(messages)
    assert "extra_body" not in sent
    off.complete_json(messages)
    assert sent["extra_body"] == {"reasoning": {"enabled": False}}
    assert plain._key(messages) != off._key(messages)   # a verdict reached with reasoning is not served to a run without it


def test_judge_clients_take_their_reasoning_from_the_config(monkeypatch):
    from medrageval.config import load_config
    from medrageval.generate.llm import make_llm, make_llms
    monkeypatch.setenv(ENV_KEY, "sk-test")
    full = load_config()
    gen, judge = make_llms(full["generate"], use_cache=False)
    rag_judge = make_llm(full["rag"]["judge"], full["generate"], use_cache=False)
    assert gen.reasoning is None   # only the judges: the writer keeps the model's default
    assert judge.model == rag_judge.model == "deepseek/deepseek-v4.1-flash"
    assert judge.reasoning == rag_judge.reasoning == {"enabled": False}
