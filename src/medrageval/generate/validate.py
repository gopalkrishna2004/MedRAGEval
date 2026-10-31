"""Programmatic checks on a generated item, before the LLM judge sees it.

Every check is cheap and deterministic: quotes are matched against the cited
units with `contains_quote`, answer keys against the quotes, numbers against
the selector's hint, and the per-type rules from the spec (both values and a
direction for comparative, two documents for cross-document, the speaker and
exclusivity of the claim for attribution, the [REDACTED] marker for
unanswerable). A failed check names its reason, which goes to rejects.jsonl.
"""

import re
from dataclasses import dataclass, field

from medrageval.generate.candidates import Candidate, garbled
from medrageval.generate.corpus import REDACTED, Corpus, tokenize
from medrageval.text import contains_quote, normalize

NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
RANGE = re.compile(r"\d[\d,]*(?:\.\d+)?\s?[-–‒—]\s?\d[\d,]*(?:\.\d+)?")
# "the second trial named", "the last-listed study": a bridge identified by its place in a list, not by a property
POSITIONAL = re.compile(r"\b(first|second|third|fourth|fifth|last|latter|former)[- ](?:\w+[- ])?(named|listed|mentioned|cited)\b|\b(named|listed|mentioned|cited) (first|second|third|last)\b", re.I)


def question_overlap(question: str, quotes: str) -> tuple[float, int]:
    """(share, count) of the question's content words that occur in the quotes: 1.0 is the quote turned into a question.

    Numbers are left out: a question about a figure has to name it. The
    count matters because a short question is mostly names (drug, trial,
    measure) and shares nearly all of them without copying anything.
    """
    q = {t for t in token_set(question) if not NUMBER.fullmatch(t)}
    shared = len(q & token_set(quotes))
    return shared / max(len(q), 1), shared


@dataclass
class Verdict:
    ok: bool = True
    reasons: list[str] = field(default_factory=list)
    checks: dict[str, bool] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)   # measured, not judged: written to the item for reporting

    def check(self, name: str, ok: bool, reason: str | None = None) -> bool:
        self.checks[name] = bool(ok)
        if not ok:
            self.ok = False
            self.reasons.append(reason or name)
        return bool(ok)


def quote_in(quote: str, plain: str) -> bool:
    """A verbatim span, or several table rows each of which is a verbatim span (rows need not be adjacent)."""
    if contains_quote(quote, plain):
        return True
    lines = [x for x in quote.split("\n") if x.strip()]
    return len(lines) > 1 and all(contains_quote(x, plain) for x in lines)


def token_set(text: str) -> set[str]:
    return set(tokenize(text))


def jaccard(a: str, b: str) -> float:
    sa, sb = token_set(a), token_set(b)
    return len(sa & sb) / max(len(sa | sb), 1)


def is_duplicate(question: str, accepted: list[str], threshold: float) -> bool:
    return any(jaccard(question, q) >= threshold for q in accepted)


def _number(s: str) -> float | None:
    m = NUMBER.search(s or "")
    try:
        return float(m.group(0).replace(",", "")) if m else None
    except ValueError:
        return None


def _range(s: str) -> str:
    return re.sub(r"[\s,]", "", s).replace("–", "-").replace("‒", "-").replace("—", "-")


def _mentions(label: str, text: str) -> bool:
    """A label is 'mentioned' when one of its content words of 4+ letters, or its number, appears in the text."""
    words = [t for t in tokenize(label) if len(t) >= 4 or t.replace(".", "").isdigit()]
    tt = token_set(text)
    return any(w in tt for w in words) if words else contains_quote(label, text)


def _aliases(speaker: str, cfg: dict) -> list[str]:
    return [speaker, speaker.removeprefix("the ")] + list(cfg["attribution"]["speakers"].get(speaker, []))


def validate(item: dict, cand: Candidate, corpus: Corpus, cfg: dict, accepted_questions: list[str]) -> Verdict:
    v = Verdict()
    lim = cfg["limits"]
    units = {u.unit_id: u for u in cand.units}
    q = str(item.get("question") or "").strip()
    if not v.check("question_present", bool(q)):
        return v
    n_words = len(q.split())
    v.check("question_length", lim["question_words"][0] <= n_words <= lim["question_words"][1], f"question_length:{n_words}")
    v.check("no_deictic", not any(re.search(p, q, re.I) for p in cfg["validate"]["deictic_patterns"]), "deictic_question")
    v.check("not_duplicate", not is_duplicate(q, accepted_questions, cfg["dedupe_jaccard"]), "duplicate_question")

    quotes = [qt for qt in (item.get("quotes") or []) if isinstance(qt, dict)]
    v.check("quotes_present", len(quotes) >= 1, "no_quotes")
    found, quoted_units = [], []
    for qt in quotes:
        uid, quote = str(qt.get("unit_id") or ""), str(qt.get("quote") or "").strip()
        if uid not in units:   # a truncated id ("d12-e035") is accepted when it is a prefix of exactly one evidence unit
            prefix = [k for k in units if k.startswith(uid)] if uid else []
            if len(prefix) == 1:
                uid = qt["unit_id"] = prefix[0]
        u = units.get(uid)
        if not v.check("quote_unit_known", u is not None, f"quote_unit_unknown:{uid}"):
            continue
        n = len(normalize(quote).split())
        v.check("quote_length", lim["quote_tokens"][0] <= n <= lim["quote_tokens"][1], f"quote_length:{n}")
        if v.check("quote_found", quote_in(quote, u.plain), f"quote_not_found:{uid}"):
            found.append(quote)
            quoted_units.append(u)
    joined = "\n".join(found)

    if cand.type == "unanswerable":
        v.check("answer_null", not item.get("answer"), "unanswerable_has_answer")
        if cand.subtype == "redacted":
            v.check("quote_redacted", any(REDACTED in qt.get("quote", "") for qt in quotes), "quote_lacks_redacted_marker")
            context = " ".join(cand.hint.get("labels", [])) + " " + (cand.units[0].caption or "") + " " + cand.units[0].plain[:300]
            v.check("in_domain", len(token_set(q) & token_set(context)) >= 2, "question_off_topic")
        return v

    answer = str(item.get("answer") or "").strip()
    v.check("answer_present", bool(answer), "no_answer")
    v.check("answer_length", len(answer) <= lim["answer_max_chars"], "answer_too_long")
    keys = [str(k).strip() for k in (item.get("answer_keys") or []) if str(k).strip()]
    v.check("answer_keys_present", 1 <= len(keys) <= 6, "answer_keys_missing")
    t = cand.type
    labels = {normalize(str(item.get(k) or "")) for k in ("label_a", "label_b")} if t == "comparative" else set()
    for k in keys:
        if t != "attribution":   # the speaker of a form answer is implied by the document, not written in it
            v.check("answer_key_in_quotes", contains_quote(k, joined), f"answer_key_not_in_quotes:{k[:40]}")
        if not any(normalize(k) in lb or lb in normalize(k) for lb in labels if lb):   # a comparative question has to name both compared things
            v.check("answer_key_not_in_question", not contains_quote(k, q), f"answer_key_in_question:{k[:40]}")
    v.check("keys_clean", not any(garbled(k) for k in keys), "answer_key_garbled")
    for k in keys:   # words, not tokens: "NHS reference costs 2021-22" is four
        v.check("key_length", len(k.split()) <= cfg["validate"]["max_key_tokens"], f"answer_key_too_long:{k[:40]}")
    for phrase in cand.hint.get("must_mention", []):   # a table with twins: the question has to say which one
        v.check("question_mentions_discriminator", contains_quote(phrase, q), f"question_lacks_discriminator:{phrase[:40]}")
    # a quoted caption does not count as copying: the question has to reuse its words to say which table is meant
    substantive = [qt for qt, u in zip(found, quoted_units) if not (u.caption and contains_quote(qt, u.caption))] or found
    if found:
        overlap, shared = question_overlap(q, "\n".join(substantive))
        v.metrics["question_overlap"] = round(overlap, 2)
        if t != "attribution":   # an attribution question has to state the claim; the others must not echo the quote
            vc = cfg["validate"]
            v.check("question_not_echo", overlap <= vc["max_question_overlap"] or shared < vc["echo_min_shared"], f"question_echoes_quote:{overlap:.2f}")

    if t == "numerical":
        nums = [n for k in keys for n in NUMBER.findall(k)]
        v.check("key_has_number", bool(nums), "answer_key_without_number")
        hint_ranges = {_range(x) for x in cand.hint.get("numbers", []) if RANGE.fullmatch(x)}
        hint = {float(x.replace(",", "")) for x in cand.hint.get("numbers", []) if not RANGE.fullmatch(x) and _number(x) is not None}
        if (hint or hint_ranges) and nums:
            ranges = {_range(r) for k in keys for r in RANGE.findall(k)}
            # a range in the evidence ("40-50%") is one value: a key must carry the whole range, not one end
            v.check("number_in_hint", bool(ranges & hint_ranges) or any(_number(n) in hint for n in nums), "number_not_in_evidence_numbers")

    elif t == "comparative":
        va, vb = item.get("value_a"), item.get("value_b")
        fa, fb = _number(str(va)), _number(str(vb))
        v.check("values_numeric", fa is not None and fb is not None, "comparative_values_not_numeric")
        v.check("values_in_quotes", bool(va) and bool(vb) and contains_quote(str(va), joined) and contains_quote(str(vb), joined), "comparative_values_not_in_quotes")
        higher = str(item.get("higher") or "").lower()
        if fa is not None and fb is not None:
            expect = "a" if fa > fb else "b" if fb > fa else "equal"
            v.check("direction_consistent", higher == expect, f"direction_inconsistent:{higher}!={expect}")
        la, lb = str(item.get("label_a") or ""), str(item.get("label_b") or "")
        v.check("labels_present", bool(la) and bool(lb), "comparative_labels_missing")
        if cand.hint.get("mode") in ("columns", "rows"):
            src = cand.units[0].plain + "\n" + (cand.units[0].caption or "")
            v.check("labels_in_table", _mentions(la, src) and _mentions(lb, src), "comparative_labels_not_in_table")
        v.check("answer_has_direction", any(re.search(rf"\b{w}\b", answer, re.I) for w in cfg["comparative"]["direction_words"]) or higher == "equal", "answer_lacks_direction")
        winner = la if higher == "a" else lb if higher == "b" else ""
        if winner:
            v.check("answer_names_winner", _mentions(winner, answer), "answer_lacks_winning_label")
        v.check("question_mentions_both", _mentions(la, q + " " + cand.hint.get("measure", "")) or _mentions(lb, q), "question_lacks_compared_labels")

    elif t == "multihop":
        v.check("two_units_quoted", len({u.unit_id for u in quoted_units}) >= 2, "quotes_from_one_unit")
        link = str(cand.hint.get("link") or "")
        if cand.subtype == "entity" and link:
            v.check("bridge_not_named", not contains_quote(link, q), f"bridge_named_in_question:{link}")
            v.check("bridge_described", not POSITIONAL.search(q), "bridge_identified_by_position")
        if cand.subtype == "xref" and link:
            v.check("table_not_named", not contains_quote(link, q), f"table_named_in_question:{link}")

    elif t == "crossdoc":
        v.check("two_docs_quoted", len({u.doc_id for u in quoted_units}) >= 2, "quotes_from_one_document")
        if len(found) >= 2:
            v.check("quotes_not_copies", jaccard(found[0], found[1]) < cfg["cross_doc"]["max_overlap"], "quotes_are_copies")

    elif t == "attribution":
        speaker = cand.hint.get("speaker", "")
        v.check("answer_is_speaker", normalize(answer) == normalize(speaker) or normalize(answer).removeprefix("the ") == normalize(speaker).removeprefix("the "),
                f"answer_not_speaker:{answer[:40]}")
        v.check("speaker_not_in_question", not any(re.search(rf"\b{re.escape(a)}\b", q, re.I) for a in _aliases(speaker, cfg)), "speaker_named_in_question")
        if cand.subtype == "stance":
            v.check("quote_has_speaker", any(re.search(rf"\b{re.escape(a)}\b", joined, re.I) for a in _aliases(speaker, cfg)), "quote_lacks_speaker")
        _check_exclusivity(v, joined, cand, corpus, cfg)
    return v


def _check_exclusivity(v: Verdict, claim: str, cand: Candidate, corpus: Corpus, cfg: dict) -> None:
    """The claim must not be found, unattributed, in a document that speaks for someone else."""
    ac = cfg["attribution"]
    speaker = cand.hint.get("speaker", "")
    limit = ac["rare_df_max"] * len(corpus.units)
    rare = {t for t in token_set(claim) if len(t) >= 3 and corpus.df.get(t, 0) <= limit}
    if not v.check("claim_specific", len(rare) >= ac["min_rare_tokens"], "claim_too_generic"):
        return
    own = {u.unit_id for u in cand.units} | {uid for u in cand.units for uid in corpus.by_element.get(u.element_id, [])}
    aliases = _aliases(speaker, cfg)
    for hit, _ in corpus.bm25.search(sorted(rare), k=10, exclude=own):
        u = corpus.units[hit]
        role_speaker = ac["role_speakers"].get(corpus.docs[u.doc_id].role)
        if role_speaker == speaker or any(re.search(rf"\b{re.escape(a)}\b", u.plain, re.I) for a in aliases):
            continue
        containment = len(rare & set(u.tokens)) / len(rare)
        if containment >= ac["restated_containment"]:
            v.check("claim_exclusive", False, f"claim_restated_in:{hit}")
            return
    v.check("claim_exclusive", True)
