import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

DEFAULT_PATH = Path("configs/rag.yaml")


def load_config(path: str | Path | None = None) -> dict:
    load_dotenv()
    p = Path(path or DEFAULT_PATH)
    with p.open() as f:
        cfg = yaml.safe_load(f)
    cfg["_path"] = str(p)
    return use_reader(cfg)


def providers(llm: dict) -> dict:
    """The answering servers by name. A config without `providers` has one, its llm section's own server and model."""
    if llm.get("providers"):
        return llm["providers"]
    own = {k: llm[k] for k in ("base_url", "base_url_env", "api_key_env") if k in llm}
    return {llm.get("provider") or "default": {**own, "models": [llm["model"]]}}


def default_reader(llm: dict) -> tuple[str, str]:
    """(provider, model) the config answers with."""
    return llm.get("provider") or next(iter(providers(llm))), llm["model"]


def use_reader(cfg: dict, provider: str | None = None, model: str | None = None) -> dict:
    """The config answering with `model` on `provider`: the provider's server and key replace the llm section's.
    No provider keeps the configured one; no model keeps the configured model there, else the provider's first."""
    llm = cfg["llm"]
    ps = providers(llm)
    default, default_model = default_reader(llm)
    name = provider or default
    if name not in ps:
        raise ValueError(f"no answering provider named {name!r}; the config has {', '.join(ps)}")
    p = ps[name]
    model = model or (default_model if name == default else p["models"][0])
    return {**cfg, "llm": {**llm, "provider": name, "model": model, "base_url": env_or(p, "base_url_env", "base_url"),
                           "api_key_env": p.get("api_key_env")}}


def check_reader(llm: dict, provider: str, model: str) -> None:
    """A choice from the page must be one the config offers, so the server only asks the servers and models it lists."""
    ps = providers(llm)
    if provider not in ps:
        raise ValueError(f"no answering provider named {provider!r}; the config has {', '.join(ps)}")
    if model not in ps[provider].get("models", []):
        raise ValueError(f"{provider} offers {', '.join(ps[provider].get('models', []))}, not {model}")


def missing_key(llm: dict) -> str | None:
    """The environment variable the chosen provider needs for an API key and does not have, if any."""
    name = llm.get("api_key_env")
    needs = providers(llm).get(llm.get("provider"), {}).get("needs_key")
    return name if needs and not os.environ.get(name or "", "").strip() else None


def use_pdf(cfg: dict, pdf: str | Path) -> dict:
    """The config pointed at another PDF. Its topic describes the configured PDF, so a different file goes without one."""
    path = Path(pdf)
    if not path.is_file():
        raise SystemExit(f"error: no such PDF: {path}")
    same = cfg.get("pdf") and path.resolve() == Path(cfg["pdf"]).resolve()
    return {**cfg, "pdf": str(path), "topic": cfg.get("topic") if same else None}


def env_or(section: dict, env_key: str, default_key: str | None = None, fallback: str = "") -> str:
    """The value of the environment variable named by `section[env_key]`, else `section[default_key]`, else `fallback`."""
    name = section.get(env_key) or ""
    val = os.environ.get(name, "").strip() if name else ""
    if val:
        return val
    return (section.get(default_key) if default_key else None) or fallback
