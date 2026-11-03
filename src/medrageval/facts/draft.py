"""Draft a fact-sheet schema for a new document family with an LLM.

The schema is summarised from one or more parsed bundles: per document
role, the section outline and a few sample passages go to the LLM, which
proposes groups, slots, qualifiers and links in the schema format. The draft is checked with
`parse_schema` and written as YAML with a header saying it needs review.
A person reviews it before it is used, as `configs/schema/nice_sta.yaml` was.
"""

import argparse
from collections import defaultdict
from pathlib import Path

import yaml

from medrageval.config import load_config
from medrageval.facts.prompts import build_schema_draft
from medrageval.facts.schema import SchemaError, known_roles, parse_schema
from medrageval.generate.corpus import Corpus, load_corpus
from medrageval.generate.llm import make_llm

HEADER = ("# DRAFT fact-sheet schema proposed by an LLM from: {sources}\n"
          "# Review before use: merge or split slots, check every qualifier, set the weights,\n"
          "# and test it with `medrageval extract <interim dir> --schema <this file> --dry-run`.\n")


def outline(corpora: list[Corpus], max_titles: int, samples_per_role: int, sample_chars: int) -> tuple[dict, dict]:
    """Section titles per role (deduplicated across bundles), and the longest few passages per role as samples."""
    titles: dict[str, list[str]] = defaultdict(list)
    samples: dict[str, list[str]] = defaultdict(list)
    for corpus in corpora:
        for d in corpus.docs.values():
            if d.role in ("cover", "other"):
                continue
            for s in corpus.sections.values():
                if s["doc_id"] == d.doc_id and s.get("depth", 1) <= 3:
                    t = "  " * (s.get("depth", 1) - 1) + " ".join(s["title"].split())[:100]
                    if t not in titles[d.role] and len(titles[d.role]) < max_titles:
                        titles[d.role].append(t)
            units = sorted((corpus.units[u] for u in corpus.by_doc.get(d.doc_id, []) if corpus.units[u].evidence),
                           key=lambda u: -len(u.plain))
            for u in units[:samples_per_role]:
                if len(samples[d.role]) < samples_per_role:
                    samples[d.role].append(u.plain[:sample_chars])
    return dict(titles), dict(samples)


def draft_schema(interim_dirs: list[Path], cfg: dict, fcfg: dict, llm) -> dict:
    corpora = [load_corpus(p, cfg) for p in interim_dirs]
    dc = fcfg["draft"]
    titles, samples = outline(corpora, dc["max_titles_per_role"], dc["samples_per_role"], dc["sample_chars"])
    roles = sorted(titles)
    return llm.complete_json(build_schema_draft(corpora[0].topic, titles, samples, roles), "schema:draft")


def add_arguments(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("interim_dirs", type=Path, nargs="+", help="one or more `medrageval parse` output directories of the document family")
    ap.add_argument("--out", type=Path, required=True, help="where to write the draft YAML")
    ap.add_argument("--config", type=Path)
    ap.add_argument("--no-cache", action="store_true")


def run_cli(args: argparse.Namespace) -> None:
    full = load_config(args.config)
    cfg, fcfg = full["generate"], full["facts"]
    llm = make_llm(fcfg, cfg, use_cache=not args.no_cache)
    data = draft_schema(args.interim_dirs, cfg, fcfg, llm)
    try:
        parse_schema(data, known_roles(cfg))
        error = None
    except SchemaError as e:   # still written: a draft is for a person to fix
        error = str(e)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    header = HEADER.format(sources=", ".join(str(p) for p in args.interim_dirs)) + (f"# INVALID as drafted: {error}\n" if error else "")
    args.out.write_text(header + yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=140))
    n = len(data.get("groups") or {}) if isinstance(data, dict) else 0
    if error:
        raise SystemExit(f"error: the drafted schema ({n} groups) is invalid: {error}; written to {args.out} for fixing")
    print(f"draft schema with {n} groups written to {args.out}; review it before use")
