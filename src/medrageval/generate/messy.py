"""The messy question setting: every accepted item asked again as a real user would type it.

The clean question is rewritten by the generator in one or more user habits
(informal, shorthand, lay terms, keyword query, ...), picked by the seed with
a severity that sets how many, then code adds surface noise: typos, lower
case, dropped punctuation, at rates that follow the severity. Nothing checks
the rewrite: the setting measures what sloppy wording costs a system, and a
rewrite that drifts from the clean question is part of that cost. Numbers are
never altered, since a changed number asks a different question. An
unanswerable item reuses its source item's rewrite, so the two stay paired.
`medrageval messy <dataset dir>` runs the pass again on an existing dataset
(`retwin`), replacing the twins and nothing else.
"""

import json
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from random import Random

from medrageval.generate.llm import LLMError, LLMJsonError
from medrageval.generate.prompts import build_messy

# keyboard neighbours for the "wrong key" typo
NEIGHBOURS = {"a": "qwsz", "b": "vghn", "c": "xdfv", "d": "serfcx", "e": "wsdr", "f": "drtgvc", "g": "ftyhbv", "h": "gyujnb", "i": "ujko",
              "j": "huikmn", "k": "jiolm", "l": "kop", "m": "njk", "n": "bhjm", "o": "iklp", "p": "ol", "q": "wa", "r": "edft", "s": "awedxz",
              "t": "rfgy", "u": "yhji", "v": "cfgb", "w": "qase", "x": "zsdc", "y": "tghu", "z": "asx"}
# text-speak spellings a hurried user types, applied to whole words, case kept lower
TEXTSPEAK = {"you": "u", "your": "ur", "are": "r", "please": "pls", "thanks": "thx", "with": "w/", "without": "w/o", "and": "&", "about": "abt",
             "versus": "vs", "between": "btw", "because": "bc", "before": "b4", "people": "ppl", "patients": "pts", "treatment": "tx",
             "question": "q", "something": "smth", "really": "rly", "probably": "prob", "though": "tho", "through": "thru",
             "for": "4", "to": "2", "okay": "ok", "number": "num", "percentage": "pct", "percent": "pct", "months": "mo", "years": "yrs"}
# "for" > "4" and "to" > "2" are text-speak a clinician rarely types, so they come at a quarter of the rate
RARE_TEXTSPEAK = {"for", "to"}
PUNCTUATION = re.compile(r"[?!;:]|(?<!\d)[.,]|[.,](?!\d)")   # a full stop or comma inside a number ("7.3", "52,681") stays


def pick(rng: Random, mcfg: dict) -> tuple[str, str, list[str]]:
    """(severity, persona, habits) for one item, from the config's mix."""
    mix = mcfg["severity_mix"]
    severity = rng.choices(list(mix), weights=list(mix.values()))[0]
    persona = rng.choice(mcfg["personas"])
    n = min(mcfg["habits_per_severity"][severity], len(mcfg["habits"]))
    habits = rng.sample(list(mcfg["habits"]), n)
    if "keyword_query" in habits:   # a keyword string cannot also ramble or hedge; misspelt names still fit
        habits = ["keyword_query"] + [h for h in habits if h == "misspell_names"]
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


def textspeak(word: str, rng: Random, rate: float) -> str | None:
    """The text-speak spelling of a word, by the rate (a quarter of it for the everyday words), keeping trailing punctuation."""
    m = re.fullmatch(r"([A-Za-z]+)([?!.,;:]*)", word)
    if not m or m.group(1).lower() not in TEXTSPEAK:
        return None
    base = m.group(1).lower()
    if rng.random() >= (rate / 4 if base in RARE_TEXTSPEAK else rate):
        return None
    return TEXTSPEAK[base] + m.group(2)


def surface_noise(text: str, rng: Random, rates: dict, min_word_len: int) -> tuple[str, list[str]]:
    """Per word: a text-speak spelling (`textspeak`), else a typo (`typo_rate`, words long enough and without a digit); then a
    missing space to the next word (`join_words`). Per question: lower case and stripped punctuation by their chances.
    Returns the text and the operations applied."""
    ops = []
    words = text.split()
    for k, w in enumerate(words):
        if any(ch.isdigit() for ch in w):
            continue
        if (ts := textspeak(w, rng, rates.get("textspeak", 0.0))) is not None:
            words[k] = ts
            ops.append(f"textspeak:{w}>{ts}")
        elif len(w) >= min_word_len and rng.random() < rates["typo_rate"]:
            new = typo(w, rng)
            if new != w:
                words[k] = new
                ops.append(f"typo:{w}>{new}")
    joined, k = [], 0
    while k < len(words):   # a missing space: this word runs into the next, unless a number is involved
        w = words[k]
        if k + 1 < len(words) and not any(ch.isdigit() for ch in w + words[k + 1]) and rng.random() < rates.get("join_words", 0.0):
            w = w + words[k + 1]
            ops.append(f"join:{words[k]} {words[k + 1]}")
            k += 1
        joined.append(w)
        k += 1
    out = " ".join(joined)
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


# ----- standalone: twins for an existing dataset -----
def retwin(dataset_dir: Path, cfg: dict, gen, seed: int | None = None) -> dict:
    """Write (or replace) `question_messy` on every item of an existing dataset.jsonl, in place, and record the pass under
    `messy` in its generate_report.json. Nothing else about the items changes, so a hand-edited dataset keeps its edits."""
    path = dataset_dir / "dataset.jsonl"
    items = [json.loads(line) for line in path.open() if line.strip()]
    if not items:
        raise RuntimeError(f"no items in {path}")
    seed = cfg["seed"] if seed is None else seed
    report = messy_pass(items, gen, cfg["messy"], seed, cfg["max_workers"])
    with path.open("w") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    report_path = dataset_dir / "generate_report.json"
    full = json.loads(report_path.read_text()) if report_path.exists() else {}
    full["messy"] = {**report, "model": gen.model, "seed": seed, "retwinned_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    report_path.write_text(json.dumps(full, indent=2, ensure_ascii=False))
    return report


def add_arguments(ap) -> None:
    ap.add_argument("dataset_dir", type=Path, help="a dataset directory from `medrageval generate`; its dataset.jsonl is rewritten in place")
    ap.add_argument("--config", type=Path)
    ap.add_argument("--seed", type=int, help="which severity, persona and habits each item gets (default: generate.seed)")
    ap.add_argument("--no-cache", action="store_true", help="ignore and do not write the LLM response cache")


def run_cli(args) -> None:
    from medrageval.config import load_config
    from medrageval.generate.llm import make_llms

    cfg = load_config(args.config)["generate"]
    gen, _ = make_llms(cfg, use_cache=not args.no_cache)
    report = retwin(args.dataset_dir, cfg, gen, args.seed)
    print(f"{report['items']} items in {args.dataset_dir}: {report['rewritten']} rewritten, {report['copied_from_source']} copied from source items, "
          f"{len(report['errors'])} fell back to surface noise")
    print(f"  severity {report['severity']}; habits {report['habits']}; surface {report['surface']}")
    print(f"  LLM calls: {gen.stats.get('calls', 0)} ({gen.stats.get('cache_hits', 0)} cached)")
    if report["errors"]:
        print(f"  WARNING: no rewrite for {', '.join(report['errors'])}")
