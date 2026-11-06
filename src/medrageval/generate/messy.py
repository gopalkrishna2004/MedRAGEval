"""The messy question setting: every accepted item asked again as a real user would type it.

The clean question is rewritten by the generator in one or more user habits
(informal, shorthand, lay terms, keyword query, ...), picked by the seed with
a severity that sets how many, then code adds surface noise: typos, lower
case, dropped punctuation, at rates that follow the severity. Nothing checks
the rewrite: the setting measures what sloppy wording costs a system, and a
rewrite that drifts from the clean question is part of that cost. Numbers are
never altered, since a changed number asks a different question. An
unanswerable item reuses its source item's rewrite, so the two stay paired.
"""

import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from random import Random

from medrageval.generate.llm import LLMError, LLMJsonError
from medrageval.generate.prompts import build_messy

# keyboard neighbours for the "wrong key" typo
NEIGHBOURS = {"a": "qwsz", "b": "vghn", "c": "xdfv", "d": "serfcx", "e": "wsdr", "f": "drtgvc", "g": "ftyhbv", "h": "gyujnb", "i": "ujko",
              "j": "huikmn", "k": "jiolm", "l": "kop", "m": "njk", "n": "bhjm", "o": "iklp", "p": "ol", "q": "wa", "r": "edft", "s": "awedxz",
              "t": "rfgy", "u": "yhji", "v": "cfgb", "w": "qase", "x": "zsdc", "y": "tghu", "z": "asx"}
PUNCTUATION = re.compile(r"[?!;:]|(?<!\d)[.,]|[.,](?!\d)")   # a full stop or comma inside a number ("7.3", "52,681") stays


def pick(rng: Random, mcfg: dict) -> tuple[str, str, list[str]]:
    """(severity, persona, habits) for one item, from the config's mix."""
    mix = mcfg["severity_mix"]
    severity = rng.choices(list(mix), weights=list(mix.values()))[0]
    persona = rng.choice(mcfg["personas"])
    n = min(mcfg["habits_per_severity"][severity], len(mcfg["habits"]))
    habits = rng.sample(list(mcfg["habits"]), n)
    if "keyword_query" in habits:   # a keyword string cannot also ramble or hedge
        habits = ["keyword_query"]
    return severity, persona, habits


def typo(word: str, rng: Random) -> str:
    """One typo in a word: swap two adjacent letters, drop one, double one, or hit a neighbouring key."""
    letters = [i for i, ch in enumerate(word) if ch.isalpha()]
    if len(letters) < 2:
        return word
    kind = rng.choice(["swap", "drop", "double", "neighbour"])
    i = rng.choice(letters[:-1] if kind == "swap" else letters)
    if kind == "swap":
        j = i + 1
        return word[:i] + word[j] + word[i] + word[j + 1:] if j < len(word) else word
    if kind == "drop":
        return word[:i] + word[i + 1:]
    if kind == "double":
        return word[:i] + word[i] + word[i:]
    near = NEIGHBOURS.get(word[i].lower())
    if not near:
        return word
    ch = rng.choice(near)
    return word[:i] + (ch.upper() if word[i].isupper() else ch) + word[i + 1:]


def surface_noise(text: str, rng: Random, rates: dict, min_word_len: int) -> tuple[str, list[str]]:
    """Typos at `typo_rate` per eligible word (long enough, no digit), lower case and stripped punctuation by their chances.
    Returns the text and the operations applied."""
    ops = []
    words = text.split()
    for k, w in enumerate(words):
        if len(w) >= min_word_len and not any(ch.isdigit() for ch in w) and rng.random() < rates["typo_rate"]:
            new = typo(w, rng)
            if new != w:
                words[k] = new
                ops.append(f"typo:{w}>{new}")
    out = " ".join(words)
    if rng.random() < rates["lowercase"]:
        out, ops = out.lower(), ops + ["lowercase"]
    if rng.random() < rates["drop_punctuation"]:
        out, ops = re.sub(r"\s+", " ", PUNCTUATION.sub("", out)).strip(), ops + ["drop_punctuation"]
    return out, ops


def rewrite(item: dict, gen, mcfg: dict, seed: int) -> dict:
    """The messy twin of one item's question. On an LLM failure the clean question gets surface noise alone, and says so."""
    rng = Random(f"{seed}:messy:{item['item_id']}")
    severity, persona, habits = pick(rng, mcfg)
    out = {"severity": severity, "persona": persona, "habits": habits}
    try:
        raw = gen.complete_json(build_messy(item["question"], persona, habits), f"messy:{item['item_id']}")
        text = str(raw.get("question") or "").strip()
    except LLMJsonError as e:
        text, out["error"] = "", f"no_json:{str(e)[:120]}"
    except LLMError as e:
        text, out["error"] = "", f"llm_error:{str(e)[:120]}"
    if not text:
        out.setdefault("error", "empty_rewrite")
        out["habits"], text = [], item["question"]
    out["rewrite"] = text
    out["text"], out["surface"] = surface_noise(text, rng, mcfg["surface"][severity], mcfg["min_word_len"])
    return out


def messy_pass(accepted: list[dict], gen, mcfg: dict, seed: int, max_workers: int) -> dict:
    """Add `question_messy` to every accepted record, in place, and return the pass's report. Unanswerable items copy their
    source item's twin when it is in the run; otherwise they get one of their own."""
    own = [r for r in accepted if not (r.get("source_item_id") and r["source_item_id"] in {a["item_id"] for a in accepted})]
    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as pool:
        twins = list(pool.map(lambda r: rewrite(r, gen, mcfg, seed), own))
    by_id = {r["item_id"]: t for r, t in zip(own, twins)}
    for r in accepted:
        r["question_messy"] = dict(by_id[r["source_item_id"]] if r["item_id"] not in by_id else by_id[r["item_id"]])
    errors = [r["item_id"] for r in own if r["question_messy"].get("error")]
    return {"items": len(accepted), "rewritten": len(own) - len(errors), "copied_from_source": len(accepted) - len(own),
            "severity": dict(Counter(r["question_messy"]["severity"] for r in accepted)),
            "habits": dict(Counter(h for r in accepted for h in r["question_messy"]["habits"])),
            "surface": dict(Counter(op.split(":")[0] for r in accepted for op in r["question_messy"]["surface"])),
            "errors": errors}
