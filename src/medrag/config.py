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
    cfg["llm"]["base_url"] = env_or(cfg["llm"], "base_url_env", "base_url")
    return cfg


def env_or(section: dict, env_key: str, default_key: str | None = None, fallback: str = "") -> str:
    """The value of the environment variable named by `section[env_key]`, else `section[default_key]`, else `fallback`."""
    name = section.get(env_key) or ""
    val = os.environ.get(name, "").strip() if name else ""
    if val:
        return val
    return (section.get(default_key) if default_key else None) or fallback
