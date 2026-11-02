"""Programmatic checks on a generated item, before the LLM judge sees it.

Every check is cheap and deterministic:
- quotes are matched against the cited units with `contains_quote`;
- answer keys must appear in the facts' values or the quotes;
- the question must name every term the fact sheet says sets the fact apart from its siblings;
- the question must not mention tables, figures, column headers, rows or columns;
- per type: both values and a meaningful difference for comparative, the
  bridge unnamed for multi-hop, two documents for cross-document, the speaker
  and the claim's exclusivity for attribution, one supported keypoint per
  fact for summary, the [REDACTED] marker for unanswerable.

A failed check names its reason, which goes to rejects.jsonl and, for a
rewrite, back to the generator.
"""

import re
from dataclasses import dataclass, field

from medrageval.facts.factsheet import NUMBER, content_tokens, numbers_in, term_mentioned
from medrageval.generate.candidates import Candidate, first_number, garbled, meaningful, too_small
from medrageval.generate.corpus import REDACTED, Corpus, tokenize
from medrageval.text import contains_quote, normalize


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


def _mentions(label: str, text: str) -> bool:
    """A label is 'mentioned' when one of its content words of 4+ letters, or its number, appears in the text."""
    words = [t for t in tokenize(label) if len(t) >= 4 or t.replace(".", "").isdigit()]
    tt = token_set(text)
    return any(w in tt for w in words) if words else contains_quote(label, text)


def _aliases(speaker: str, cfg: dict) -> list[str]:
    return [speaker, speaker.removeprefix("the ")] + list(cfg["attribution"]["speakers"].get(speaker, []))


def bridge_named(question: str, cand: Candidate) -> str | None:
    """The bridged value a multi-hop question gives away, or None.

    Named when the question contains the `via` value ("PFS"), or two content
    words of the bridge fact's value that the bridge description and the
    target fact's qualifiers do not already use ("progression-free survival").
    """
    h = cand.hint
    via, value = str(h.get("via_value") or ""), str(h.get("bridge_value") or "")
    q = token_set(question)
    if via and (contains_quote(via, question) or (content_tokens(via) and content_tokens(via) <= q)):
        return via
    allowed = content_tokens(h.get("bridge", ""))
    for f in cand.facts[1:]:
        allowed |= {t for k, x in f.qualifiers.items() if k != h.get("via") for t in content_tokens(x)}
    leaked = (content_tokens(value) - allowed) & q
    return value if len(leaked) >= 2 or (leaked and leaked == content_tokens(value) - allowed) else None


def validate(item: dict, cand: Candidate, corpus: Corpus, cfg: dict, accepted_questions: list[str]) -> Verdict:
    v = Verdict()
    lim, vc = cfg["limits"], cfg["validate"]
    units = {u.unit_id: u for u in cand.units}
    t = cand.type
    q = str(item.get("question") or "").strip()
    if not v.check("question_present", bool(q), "no_question"):
        return v
    n_words = len(q.split())
    v.check("question_length", lim["question_words"][0] <= n_words <= lim["question_words"][1], f"question_length:{n_words}")
    v.check("no_deictic", not any(re.search(p, q, re.I) for p in vc["deictic_patterns"]), "deictic_question")
    banned = next((m.group(0) for p in vc["banned_patterns"] if (m := re.search(p, q, re.I))), None)
    v.check("no_table_language", banned is None, f"table_language:{banned}")
    if t in vc["single_ask_types"]:
        joined_ask = next((m.group(0) for p in vc["compound_patterns"] if (m := re.search(p, q, re.I))), None)
        v.check("single_ask", joined_ask is None, f"compound_question:{joined_ask}")
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
    support = joined + "\n" + "\n".join(f.value_text() for f in cand.facts)

    for terms in cand.hint.get("must_mention") or []:
        v.check("question_disambiguates", any(term_mentioned(term, q) for term in terms), f"question_lacks_discriminator:{terms[0][:60]}")

    if t == "unanswerable":
        v.check("answer_null", not item.get("answer"), "unanswerable_has_answer")
        if cand.subtype == "redacted":
            v.check("quote_redacted", any(REDACTED in qt.get("quote", "") for qt in quotes), "quote_lacks_redacted_marker")
            context = " ".join(cand.hint.get("labels", [])) + " " + (cand.units[0].caption or "") + " " + cand.units[0].plain[:300]
            v.check("in_domain", len(token_set(q) & token_set(context)) >= 2, "question_off_topic")
        return v

    answer = str(item.get("answer") or "").strip()
    v.check("answer_present", bool(answer), "no_answer")
    if t == "summary":
        v.check("answer_length", len(answer) <= cfg["summary"]["answer_max_chars"], "summary_answer_too_long")
    else:
        v.check("answer_length", len(answer) <= lim["answer_max_chars"], "answer_too_long")
    keys = [str(k).strip() for k in (item.get("answer_keys") or []) if str(k).strip()]
    if t != "summary":   # a summary is scored on its keypoints
        v.check("answer_keys_present", 1 <= len(keys) <= 6, "answer_keys_missing")
    compares = t == "comparative"
    labels = {normalize(str(item.get(k) or "")) for k in ("label_a", "label_b")} if compares else set()
    generic = {normalize(g) for g in vc["generic_keys"]}
    for k in keys:
        if t != "attribution":   # the speaker of a form answer is implied by the document, not written in it
            v.check("answer_key_supported", contains_quote(k, support), f"answer_key_unsupported:{k[:40]}")
            v.check("answer_key_specific", not re.fullmatch(r"[<>≤≥]?\d{1,2}", k) and normalize(k) not in generic, f"answer_key_generic:{k[:40]}")
            v.check("answer_key_unlabelled", not re.match(r"^[A-Za-z][^:]{2,60}:\s*\S", k), f"answer_key_has_label:{k[:40]}")
        if not any(normalize(k) in lb or lb in normalize(k) for lb in labels if lb):   # a comparative question has to name both compared things
            v.check("answer_key_not_in_question", not contains_quote(k, q), f"answer_key_in_question:{k[:40]}")
    v.check("keys_clean", not any(garbled(k) or REDACTED in k for k in keys), "answer_key_garbled")
    for k in keys:   # words, not tokens: "NHS reference costs 2021-22" is four
        v.check("key_length", len(k.split()) <= vc["max_key_tokens"], f"answer_key_too_long:{k[:40]}")
    if found and t not in ("attribution", "summary"):
        overlap, shared = question_overlap(q, joined)
        v.metrics["question_overlap"] = round(overlap, 2)
        v.check("question_not_echo", overlap <= vc["max_question_overlap"] or shared < vc["echo_min_shared"], f"question_echoes_quote:{overlap:.2f}")

    if t == "numerical":
        nums = {n.replace(",", "") for k in keys for n in NUMBER.findall(k)}
        v.check("key_has_number", bool(nums), "answer_key_without_number")
        hint = set(cand.hint.get("numbers") or [])
        if hint and nums:
            v.check("number_in_fact", bool(nums & hint), "number_not_in_fact")

    elif t == "comparative":
        la, lb = _check_comparison(v, item, support, answer, cfg)
        v.check("question_mentions_both", _mentions(la, q) and _mentions(lb, q), "question_lacks_compared_labels")

    elif t == "multihop":
        v.check("two_units_quoted", len({u.unit_id for u in quoted_units}) >= 2, "quotes_from_one_unit")
        named = bridge_named(q, cand)
        v.check("bridge_not_named", named is None, f"bridge_named_in_question:{str(cand.hint.get('via', 'value')).replace('_', ' ')}")

    elif t == "crossdoc":
        v.check("two_docs_quoted", len({u.doc_id for u in quoted_units}) >= 2, "quotes_from_one_document")
        if len(found) >= 2:
            v.check("quotes_not_copies", jaccard(found[0], found[1]) < cfg["cross_doc_max_overlap"], "quotes_are_copies")

    elif t == "attribution":
        speaker = cand.hint.get("speaker", "")
        name = cand.hint.get("speaker_name") or ""
        targets = {normalize(x).removeprefix("the ") for x in (speaker, name) if x}
        v.check("answer_is_speaker", normalize(answer).removeprefix("the ") in targets, f"answer_not_speaker:{answer[:40]}")
        names = _aliases(speaker, cfg) + ([name] if name else [])
        v.check("speaker_not_in_question", not any(re.search(rf"\b{re.escape(a)}\b", q, re.I) for a in names), "speaker_named_in_question")
        if cand.subtype == "stance":
            v.check("quote_has_speaker", any(re.search(rf"\b{re.escape(a)}\b", joined, re.I) for a in _aliases(speaker, cfg)), "quote_lacks_speaker")
        _check_exclusivity(v, joined, cand, corpus, cfg)

    elif t == "summary":
        _check_keypoints(v, item, cand, cfg)
    return v


def _check_comparison(v: Verdict, item: dict, support: str, answer: str, cfg: dict) -> tuple[str, str]:
    """Two numeric values from the facts, a meaningful difference, a direction that matches, an answer that names the winner."""
    va, vb = str(item.get("value_a") or ""), str(item.get("value_b") or "")
    fa, fb = first_number(va), first_number(vb)
    v.check("values_numeric", fa is not None and fb is not None, "comparative_values_not_numeric")
    v.check("values_in_facts", bool(va) and bool(vb) and contains_quote(va, support) and contains_quote(vb, support), "comparative_values_not_in_facts")
    threshold = cfg["comparative"]["min_relative_difference"]
    if fa is not None and fb is not None:
        v.check("comparison_meaningful", meaningful(fa, fb, threshold), f"comparison_not_meaningful:{va} vs {vb}")
        v.check("comparison_large_enough", not too_small(va, vb, cfg), f"comparison_too_small:{va} vs {vb}")
    higher = str(item.get("higher") or "").lower()
    if fa is not None and fb is not None:
        expect = "a" if fa > fb else "b" if fb > fa else "equal"
        v.check("direction_consistent", higher == expect, f"direction_inconsistent:{higher}!={expect}")
    la, lb = str(item.get("label_a") or ""), str(item.get("label_b") or "")
    v.check("labels_present", bool(la) and bool(lb), "comparative_labels_missing")
    v.check("answer_has_direction", any(re.search(rf"\b{w}\b", answer, re.I) for w in cfg["comparative"]["direction_words"]) or higher == "equal", "answer_lacks_direction")
    winner = la if higher == "a" else lb if higher == "b" else ""
    if winner:
        v.check("answer_names_winner", _mentions(winner, answer), "answer_lacks_winning_label")
    return la, lb


def _check_keypoints(v: Verdict, item: dict, cand: Candidate, cfg: dict) -> None:
    """One keypoint per fact (at least min_facts), each citing a fact of the candidate and stating it."""
    facts = {f.fact_id: f for f in cand.facts}
    kps = [kp for kp in (item.get("keypoints") or []) if isinstance(kp, dict)]
    need = min(len(facts), cfg["summary"]["min_facts"])
    v.check("keypoints_present", need <= len(kps) <= len(facts) + 1, f"keypoints_missing:{len(kps)} of {len(facts)}")
    for kp in kps:
        fid = str(kp.get("fact_id") or "")
        f = facts.get(fid)
        if not v.check("keypoint_fact_known", f is not None, f"keypoint_unknown_fact:{fid[:40]}"):
            continue
        point = str(kp.get("point") or "")
        source = f.value_text() + "\n" + f.quote
        ok = len(content_tokens(point) & content_tokens(source)) >= 2 and numbers_in(point) <= numbers_in(source)
        v.check("keypoint_supported", ok, f"keypoint_unsupported:{fid}")


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
