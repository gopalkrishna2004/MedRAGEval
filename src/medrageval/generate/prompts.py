"""Prompts for item generation, the judge pass and the unanswerable check.

Generation is written from facts, not from excerpts. The writer sees each
fact of the fact sheet (what it is, its qualifiers, its value, its document)
and, when the bundle has siblings, the terms the question must name to mean
only this fact. The excerpts follow so it can quote them. A question written
from "median PFS; trial CAPItello-291; population PI3K/AKT pathway-altered"
has no table number or column header to copy.

PROMPT_VERSION is written into the system message, so a prompt change
invalidates the response cache and is recorded in every item's provenance.
The prompts never name the appraisal or its documents themselves: the corpus
supplies the topic and document roles, so the same prompts work on another
bundle.
"""

import json

from medrageval.facts.factsheet import Fact, FactSheet
from medrageval.generate.candidates import Candidate
from medrageval.generate.corpus import Corpus, Unit

PROMPT_VERSION = "17"

SCHEMA = ('{"question": str, "answer": str, "answer_keys": [str, ...], '
          '"quotes": [{"unit_id": str, "quote": str}, ...]}')

RULES = """Rules for every item:
1. The question must be answerable from the facts and excerpts given, and must stand on its own for a reader who has not seen them: name the drug, trial, population, comparator, outcome, analysis or document involved. Never write "this table", "the excerpt", "above", "the following", or a document id ("d04"): the reader does not see ids. Keep it under 35 words.
2. The system answering will search all the documents listed above. Similar figures exist elsewhere in the bundle, so the question must say what sets this one apart; when the task lists terms the question must name, name them.
3. Ask for one thing. Never join two questions ("what was X, and how did it affect Y?", "which tool and which estimate"); only the comparative, cross-document and summary tasks combine facts, as they say.
4. Ask as a clinician, health economist or appraisal committee member would. Never mention table or figure numbers, column headers such as "(N=155)", rows, columns, or how a document is laid out. Spell out abbreviations a reader may not know ("the company's cost-effectiveness model", not "the company's CEM"); common ones (PFS, OS, ICER, QALY, EAG, NHS) are fine. Put the question in your own words: reuse the excerpt's wording only for names, numbers and technical terms.
5. Name each drug, arm or regimen once, in plain words, even when the documents spell it differently ("everolimus plus exemestane", not "Everolimus + exemestane (Everolimus with exemestane)").
6. Claim no more than the excerpts say. Put a qualifier (data cut, timepoint, population, scenario, whose analysis) in the question only if the excerpts or their Context state it: a fact may carry qualifiers its excerpt does not show, so leave those out. Describe a position in the excerpt's own terms ("suggested" is not "preferred", "considered" is not "concluded").
7. Do not put the answer in the question, and do not use knowledge from outside the facts and excerpts. Never ask for a value shown as [REDACTED] unless the task is about a redacted value.
8. "answer": one sentence, specific. "answer_keys": 1 to 4 strings a correct answer must contain, each a number with its unit, a name, or a phrase of at most four words, never a clause, written as in the fact's value or the quotes. A key is the value itself, never "label: value". Each key answers the question as asked.
9. "quotes": entries of {"unit_id": the excerpt id shown in brackets, "quote": a contiguous span copied verbatim from that excerpt, 5 to 80 words}. The quote must contain the fact the answer rests on. From a table, quote one row per entry; never join rows or cells from different lines.
10. Text headed "Context" is the passage before an excerpt in its section, shown so you know what the excerpt is about. Never quote it or ask about its facts.
11. If the facts cannot carry a sound item of this type (a trivial or generic fact; a comparison that means nothing, such as small event counts of single digits or zero against one, or a difference no decision-maker would care about; a value that is not what the fact claims), return {"skip": <one short reason>} instead of forcing one.
Return only a JSON object."""

TYPE_INSTRUCTIONS = {
    "factual": "Write a FACTUAL question asking for fact {f0}: {describe}. Name its subject as the qualifiers give it (the drug, trial, population, comparator, analysis, or whose position it is). "
               "The answer states the fact in one sentence; answer_keys are its key terms (a name, a number with its unit, or a phrase of at most four words) that a correct answer must contain.",
    "numerical": "Write a NUMERICAL question asking for {describe} ({label}), from fact {f0}. The answer is {value}. "
                 "Name the measure{label_clause} and the subject as the qualifiers give them. answer_keys: the number with its unit, as written in the fact ({value}).",
    "comparative": "Write a COMPARATIVE question comparing {measure} between {label_a} ({value_a}) and {label_b} ({value_b}). "
                   "Name both and the measure, and the subject as the qualifiers give it. The answer says which is higher (or lower) and gives both values. "
                   "Add the fields \"measure\", \"label_a\", \"value_a\", \"label_b\", \"value_b\" (the two values as the facts give them) and \"higher\" (\"a\", \"b\" or \"equal\"). "
                   "answer_keys: the two values. Quote the row or sentence holding each value.",
    "multihop": "Write a MULTI-HOP question. Fact {f0} says that {bridge} is \"{bridge_value}\". Fact {f1} gives a value for {via} \"{via_value}\". "
                "Ask for fact {f1}'s value, but refer to the {via} only as \"{bridge}\" (or words meaning the same), never by name: the question must not contain \"{via_value}\" "
                "or any wording of \"{bridge_value}\". A reader must find the first fact to know which {via} is meant, then the second for the value. "
                "Name everything else that tells the value apart. The answer is the second fact's value; answer_keys hold it. "
                "Add \"hop_a\" and \"hop_b\": one short sentence each on what each fact contributes. Quote from both excerpts.",
    "crossdoc": "Write a CROSS-DOCUMENT question: facts {f0} ({doc_a}) and {f1} ({doc_b}) give different accounts of {describe}. "
                "Ask how the two documents differ on it, naming the subject and both documents, and naming each arm or regimen once in plain words. The answer gives both values or positions, each with its document; "
                "answer_keys hold the two values, or the key term of each position. Quote from both excerpts.",
    "attribution": "Write an ATTRIBUTION question: fact {f0} records a position held by {speaker_label}: \"{claim}\". Ask WHO took it (\"Who argued that ...?\", \"Which party considered that ...?\"), "
                   "stating the position precisely enough to identify it, but do NOT name or hint at the speaker in the question. "
                   "Keep the part of the claim that is this party's own (its argument, judgement, survey or experience), not background any party could state. "
                   "No first person, no wording only the speaker would use (\"our model\", \"the submission\", \"the EAG's critique\"). "
                   "The answer must be exactly \"{answer}\" and answer_keys must be [\"{answer_key}\"]. Quote the sentence that carries the claim.",
    "summary": "Write a SUMMARY question: the facts below are all about {group_describe}, in the {doc}{respondent}. "
               "Find the specific theme they share and ask one open question about it, naming the theme, the drug and the document{or_respondent} "
               "(\"How did the company justify applying the severity modifier and its dose-intensity scenarios?\", not \"How did the company present the cost-effectiveness evidence?\"). "
               "Do not list the facts in the question. If they share no theme a reader would ask about, return {{\"skip\": reason}}. "
               "\"answer\": one sentence per keypoint, in order, and nothing beyond them. "
               "\"keypoints\": [{{\"point\": one sentence stating one fact, \"fact_id\": that fact's id}}, ...], one per fact. "
               "\"answer_keys\": []. \"quotes\": [] (the system attaches each fact's quote).",
    "unanswerable_redacted": "Write an UNANSWERABLE question asking for a value the bundle redacts: {labels}. "
                             "Ask for it exactly as a reader who expects it to be there would (\"What was the ...?\"), naming the subject. The correct response of a QA system is that the value is not disclosed. "
                             "Set \"answer\" to null and \"answer_keys\" to []. Quote the row or sentence holding the [REDACTED] marker, marker included.",
    "unanswerable_not_covered": "Write 3 UNANSWERABLE questions anchored in fact {f0}: {describe}. Across the whole bundle this measure is reported only for: {reported}. "
                                "Each question asks for the same measure for a version the bundle does NOT report (another data cut, timepoint, subgroup, population, dose, scenario or comparator), "
                                "phrased as if the answer existed, naming the drug and the version it asks for. A system that answers with the nearest figure it finds is wrong. "
                                "The fact must not follow from what the documents do say. Never ask about software, files, finances, manufacturing or personnel. "
                                "Return {{\"questions\": [{{\"question\": str, \"missing_fact\": str}}, ...]}} and nothing else.",
}


def _doc_list(corpus: Corpus) -> str:
    return "\n".join(f"- {d.doc_id}: {d.label} (pages {d.start_page}-{d.end_page})" for d in corpus.docs.values() if d.role != "cover")


def system_message(corpus: Corpus) -> str:
    return (f"You write evaluation items for a question-answering system over a bundle of documents from one health technology appraisal: "
            f"\"{corpus.topic}\".\nThe documents in the bundle are:\n{_doc_list(corpus)}\n\n"
            f"You will be given facts from the bundle's fact sheet, each headed by its id, and the excerpts they come from, each headed by its id in brackets, document, section and pages. "
            f"Write exactly what the task asks, as one JSON object with this shape: {SCHEMA} (plus any extra fields the task names).\n\n{RULES}\n"
            f"(prompt version {PROMPT_VERSION})")


def fact_lines(f: Fact, sheet: FactSheet) -> str:
    lines = [f"Fact {f.fact_id}: {sheet.describe(f)} (from the {sheet.doc_label(f)})"]
    if f.qualifiers:
        lines.append("  qualifiers: " + "; ".join(f"{k}: {v}" for k, v in f.qualifiers.items()))
    if f.values:
        lines.append("  values: " + "; ".join(f"{k}: {v}" for k, v in f.values.items()))
    elif f.positions:
        lines.append("  positions: " + "; ".join(f"{k}: {v}" for k, v in f.positions.items()))
    elif f.value:
        lines.append(f"  value: {f.value}")
    if f.speaker:
        lines.append(f"  respondent: {f.speaker}")
    return "\n".join(lines)


def must_mention_text(sets: list[list[str]]) -> str:
    if not sets:
        return ""
    return ("\nThe bundle reports this for other cases too, so the question must name "
            + " and ".join("(" + " or ".join(f"\"{t}\"" for t in s) + ")" for s in sets)
            + ", in these or equivalent words. Work them into the sentence as a reader would say them: expand table shorthand "
            "(\"Capi + ful\" is capivasertib plus fulvestrant) and never open with a bare label (\"For Female sex, ...\").")


def _context(u: Unit, corpus: Corpus, cfg: dict) -> str:
    """The passages before the excerpt in its section, without unit ids so they cannot be quoted."""
    cc = cfg["context"]
    texts = []
    for p in corpus.preceding(u, cc["units"]):
        t = p.plain if len(p.plain) <= cc["max_chars"] else "[...] " + p.plain[-cc["max_chars"]:]
        texts.append(t)
    return "\n".join(texts)


def _excerpts(cand: Candidate, corpus: Corpus, cfg: dict, max_chars: int = 3500) -> str:
    max_chars = 1500 if len(cand.units) > 3 else max_chars
    parts = []
    for u in cand.units:
        context = _context(u, corpus, cfg) if cand.subtype != "not_covered" and len(cand.units) <= 3 else ""
        parts.append((f"Context before excerpt [{u.unit_id}] (not quotable):\n{context}\n\n" if context else "") + f"Excerpt:\n{corpus.render(u, max_chars)}")
    return "\n\n".join(parts)


def build_generation(cand: Candidate, corpus: Corpus, cfg: dict, sheet: FactSheet) -> list[dict]:
    h, facts = cand.hint, cand.facts
    key = f"{cand.type}_{cand.subtype}" if cand.type == "unanswerable" else cand.type
    f0 = facts[0] if facts else None
    labels = list(h.get("labels") or []) + ["", ""]
    values = list(h.get("values") or []) + ["", ""]
    speaker_name = h.get("speaker_name")
    answer = speaker_name or h.get("speaker", "")
    fmt = dict(
        f0=f0.fact_id if f0 else "", f1=facts[1].fact_id if len(facts) > 1 else "",
        describe=sheet.describe(f0) if f0 else "", value=h.get("value", ""), label=h.get("label") or (f0.name.replace("_", " ") if f0 else ""),
        label_clause=f", the {h['label']} it belongs to" if h.get("label") else "",
        measure=h.get("measure", ""), label_a=labels[0], label_b=labels[1], value_a=values[0], value_b=values[1],
        bridge=h.get("bridge", ""), bridge_value=h.get("bridge_value", ""), via=str(h.get("via", "")).replace("_", " "), via_value=h.get("via_value", ""),
        doc_a=(h.get("docs") or ["", ""])[0], doc_b=(h.get("docs") or ["", ""])[1],
        speaker_label=(f"{speaker_name} ({h.get('speaker', '')})" if speaker_name else h.get("speaker", "")), claim=str(h.get("claim", ""))[:400],
        answer=answer, answer_key=answer.removeprefix("the ").strip(),
        group_describe=h.get("group_describe", ""), doc=h.get("doc", ""),
        respondent=f", submitted by {speaker_name}" if speaker_name else "", or_respondent=" or the respondent" if speaker_name else "",
        max_sentences=max(3, len(facts) + 1),
        labels=" / ".join(h.get("labels", [])),
        reported="; ".join(f"{k}: {', '.join(v[:8])}" for k, v in (h.get("reported") or {}).items()) or "the cases shown",
    )
    task = TYPE_INSTRUCTIONS[key].format(**fmt) + must_mention_text(h.get("must_mention") or [])
    fact_block = "\n".join(fact_lines(f, sheet) for f in facts)
    body = f"Facts:\n{fact_block}\n\n{_excerpts(cand, corpus, cfg)}"
    if cand.subtype == "not_covered":
        near = corpus.neighbours(cand.units[0], cfg["unanswerable"]["neighbours"])
        if near:
            body += "\n\nNearby passages (their facts are also off limits):\n" + "\n\n".join(corpus.render(u, 800) for u in near)
    return [{"role": "system", "content": system_message(corpus)},
            {"role": "user", "content": f"{task}\n\n{body}"}]


JUDGE_CRITERIA = {
    "factual": "The answer is stated by the quotes, and the question is specific enough that this fact is the natural answer.",
    "numerical": "The answer's number and unit appear in the quotes and belong to the measure, arm and population the question names.",
    "comparative": "Both values appear in the quotes, the direction (higher/lower) is correct, and the two labels are the right ones. The question is not answerable from one value alone, and the difference is one a reader would care about.",
    "multihop": "The question identifies what it asks about through a description (hop A) and asks for a value (hop B). Pass when the description is resolved in one excerpt and the value is in the other. "
                "Fail if one excerpt holds both the description's answer and the value, if the question names the bridged thing outright, or if it identifies it only by its position in a list.",
    "crossdoc": "The quotes come from two different documents, and the answer reports what each says about the same figure, assumption or claim. Fail if one document alone gives everything the answer needs, "
                "or if the question does not say which documents and subject it is about.",
    "attribution": "The quoted claim is made by the party given as the answer, the question identifies that claim unambiguously, and the question does not name or hint at the party.",
    "summary": "The question is open but specific (names the drug and the document or respondent); every keypoint is supported by the quotes; the answer covers all keypoints and adds nothing unsupported.",
    "unanswerable": "The question asks for a value that is redacted in the quote, as a reader who expected it to be present would ask.",
}


def build_judge(item: dict, cand: Candidate, corpus: Corpus, cfg: dict) -> list[dict]:
    """The item with its excerpts, each preceded by the Context the writer saw.

    Without the context the judge rejected sound items: a results table under
    a section that names the data cut "does not identify DCO1" on its own (four
    of the fresh smoke run's judge failures).
    """
    units = {u.unit_id: u for u in cand.units}
    quoted = [q["unit_id"] for q in item.get("quotes", []) if isinstance(q, dict) and q.get("unit_id") in units]
    chosen = [units[u] for u in dict.fromkeys(quoted)] or cand.units
    parts = []
    for u in chosen:
        context = _context(u, corpus, cfg)
        parts.append((f"Context before [{u.unit_id}]:\n{context}\n\n" if context else "") + corpus.render(u, cfg["judge"]["max_chars_per_unit"]))
    shown = "\n\n".join(parts)
    payload = {k: item.get(k) for k in ("question", "answer", "answer_keys", "keypoints", "quotes") if k in item}
    return [
        {"role": "system", "content":
            f"You check evaluation items for a question-answering system over documents of one health technology appraisal: \"{corpus.topic}\". "
            "Be strict: an item passes only if every criterion holds. Return only a JSON object: "
            '{"pass": bool, "answer_supported": bool, "self_contained": bool, "type_fits": bool, "realistic": bool, '
            '"answerable_from_single_excerpt": {"<unit_id>": bool, ...}, "issues": [str, ...]}. '
            "\"self_contained\" means a reader with access to the whole document bundle but not to these excerpts could tell what is being asked. "
            "\"realistic\" means a clinician, health economist or appraisal committee member could plausibly ask it: it does not depend on table or figure numbers, column headers or the layout of a document, and the answer matters to the appraisal. "
            "Text headed Context is the passage before an excerpt in its section: what it states (the trial, population, data cut, whose analysis) supports the question as much as the excerpt does. "
            "\"answerable_from_single_excerpt\" gives, for each excerpt id, whether that excerpt alone suffices: it does only if that excerpt contains both the information the question uses to identify its subject and the answer. "
            f"(prompt version {PROMPT_VERSION})"},
        {"role": "user", "content":
            f"Item type: {cand.type}\nCriteria: {JUDGE_CRITERIA.get(f'{cand.type}_{cand.subtype}', JUDGE_CRITERIA[cand.type])}\n\n"
            f"Item:\n{json.dumps(payload, ensure_ascii=False, indent=1)}\n\nSource excerpts:\n{shown}"},
    ]


def build_alone_check(question: str, unit: Unit, corpus: Corpus, cfg: dict) -> list[dict]:
    """The question with ONE excerpt: can a reader holding only this passage answer it?"""
    return [
        {"role": "system", "content":
            f"You answer a question about one health technology appraisal (\"{corpus.topic}\") from ONE passage and nothing else. "
            'Return only a JSON object: {"answer": str | null, "reason": str}. '
            "Give the answer when the passage settles it: it states the answer for the subject the question describes; or only one item in the passage can be meant, "
            "given everything else the question says; or the answer is the same whichever item in the passage is meant. "
            "Return null when the passage lacks the answer, or holds different values for items the question could mean and nothing in it says which. "
            "Do not use knowledge from outside the passage. "
            f"(prompt version {PROMPT_VERSION})"},
        {"role": "user", "content": f"Question: {question}\n\nPassage:\n{corpus.render(unit, cfg['judge']['max_chars_per_unit'])}"},
    ]


# Why an item was rejected, as an instruction the generator can act on. Keyed by reason code (the part before ":"),
# or by the whole reason for the judge's verdicts; {d} is the detail after the colon.
FEEDBACK = {
    "answer_key_garbled": "An answer key is garbled or is the [REDACTED] marker: keys are readable values copied from the fact or the quotes.",
    "answer_key_in_question": "The answer key \"{d}\" is already in the question: the question must not give the answer away.",
    "answer_key_unsupported": "The answer key \"{d}\" is in neither the facts' values nor your quotes: copy keys from them.",
    "answer_key_too_long": "The answer key \"{d}\" is too long: a key is a value, a name or a phrase of at most four words.",
    "answer_key_generic": "The answer key \"{d}\" is too generic to score an answer (a bare small number or a word like \"yes\" or \"weak\"): use the specific value with its unit or name.",
    "answer_key_without_number": "A numerical item needs the number with its unit among the answer keys.",
    "answer_keys_missing": "Give 1 to 4 answer keys.",
    "answer_lacks_direction": "The answer must say which value is higher or lower.",
    "answer_lacks_winning_label": "The answer must name the side with the higher value.",
    "answer_not_speaker": "The answer must be exactly the speaker named in the task.",
    "answer_too_long": "The answer is too long.",
    "answer_key_has_label": "The answer key \"{d}\" carries a label: a key is the value itself (\"0\", not \"Placebo plus fulvestrant: 0\").",
    "comparison_too_small": "The two values are small counts ({d}): compare something a decision-maker would care about, or skip.",
    "bridge_named_in_question": "The question names the bridged {d}: refer to it only through the description the task gives.",
    "comparative_labels_missing": "Fill label_a and label_b with the two things compared.",
    "comparative_values_not_in_facts": "value_a and value_b must be the two values the task names, as the facts give them.",
    "comparative_values_not_numeric": "value_a and value_b must be numbers.",
    "comparison_not_meaningful": "The two values differ too little to be worth comparing ({d}).",
    "compound_question": "The question asks two things at once (\"{d}\"): ask for one fact only.",
    "deictic_question": "The question points at the excerpt (\"this table\", \"above\", a document id): name what it is about instead.",
    "table_language": "The question mentions a table, figure, column header, row or column (\"{d}\"): ask about the subject itself.",
    "direction_inconsistent": "\"higher\" does not match the two values ({d}).",
    "keypoints_missing": "Give one keypoint per fact, each with the fact's id ({d}).",
    "keypoint_unknown_fact": "A keypoint cites \"{d}\", which is not one of the facts given.",
    "keypoint_unsupported": "The keypoint for {d} does not state that fact: write it from the fact's value.",
    "no_answer": "The answer is missing.",
    "no_question": "The question is missing.",
    "no_quotes": "Quote the span or row the answer rests on.",
    "number_not_in_fact": "The answer's number is not the fact's value.",
    "question_echoes_quote": "The question copies the quote's wording (share of its words in the quote: {d}): ask it in your own words, keeping only names, numbers and technical terms.",
    "question_lacks_compared_labels": "The question must name both things compared.",
    "question_lacks_discriminator": "The question must name \"{d}\" (or equivalent words), so it means only this fact.",
    "question_length": "The question has {d} words: keep it within the word limit.",
    "question_off_topic": "The question is not about the fact's subject.",
    "quote_lacks_redacted_marker": "Quote the row or sentence that holds the [REDACTED] marker, marker included.",
    "quote_lacks_speaker": "Quote the sentence that names the party making the claim.",
    "quote_length": "A quote has {d} words: quotes are 5 to 80 words (a table row may be shorter).",
    "quote_not_found": "A quote from {d} is not a verbatim span of that excerpt: copy it exactly, one table row per entry, never joining rows or cells.",
    "quote_unit_unknown": "A quote cites \"{d}\", which is not an excerpt id: use the id shown in brackets.",
    "quotes_are_copies": "The two quotes say the same thing: quote what each document says.",
    "quotes_from_one_document": "Quote from both documents.",
    "quotes_from_one_unit": "Quote from both excerpts.",
    "speaker_named_in_question": "The question names or hints at the speaker: state the claim without it.",
    "summary_answer_too_long": "The summary answer is too long: cover each fact in about one sentence.",
    "unanswerable_has_answer": "An unanswerable item has \"answer\": null and no answer keys.",
    "judge_failed": "A reviewer rejected the item (reasons below, if given).",
    "judge:answer_supported": "A reviewer found the answer is not supported by the quotes.",
    "judge:self_contained": "A reviewer could not tell what is asked without the excerpts: name the drug, trial, population, comparator or document.",
    "judge:type_fits": "A reviewer found the item does not do what its type asks.",
    "judge:realistic": "A reviewer found the question is not one a clinician, health economist or committee member would ask: ask about what matters in the appraisal, not about the document's layout.",
    "judge:answerable_from_single_excerpt": "A reviewer found that one excerpt answers the question alone: it must need both.",
    "answerable_from": "Given only excerpt {d}, a reader answered \"{answer}\": describe the subject only through the other fact, or ask for something only the two together give.",
    "answered": "The documents answer it ({answer}): ask for a near miss they do not report.",
}


def rewrite_feedback(results: list[dict], cand: Candidate) -> list[str]:
    """Why the item failed, one instruction per failed check, plus the reviewer's issues and what one excerpt alone answered."""
    lines = []
    for r in results:
        if r["stage"] not in ("programmatic", "judge", "unanswerable_check"):
            continue
        prefix = f"Question \"{r['item'].get('question', '')}\": " if cand.subtype == "not_covered" else ""
        for reason in r["reasons"]:
            code, _, detail = reason.partition(":")
            if code == "answerable_from":
                answer = ((r.get("alone") or {}).get(detail) or {}).get("answer")
                lines.append(prefix + FEEDBACK[code].format(d=detail, answer=answer))
            elif code == "answered":
                lines.append(prefix + FEEDBACK[code].format(answer=(r.get("judge") or {}).get("answer_if_any") or "a passage covers it"))
            else:
                text = FEEDBACK.get(reason) or FEEDBACK.get(code)
                lines.append(prefix + (text.format(d=detail) if text else reason))
        if r["stage"] == "judge":
            lines += [f"Reviewer: {i}" for i in (r.get("judge") or {}).get("issues") or [] if isinstance(i, str)]
    return list(dict.fromkeys(lines))


def build_rewrite(messages: list[dict], raw: dict, feedback: list[str]) -> list[dict]:
    """The generation conversation, the rejected item and why: one chance to fix it."""
    notes = "\n".join(f"- {f}" for f in feedback)
    return messages + [
        {"role": "assistant", "content": json.dumps(raw, ensure_ascii=False)},
        {"role": "user", "content": f"That item was rejected:\n{notes}\n"
            "Rewrite it once so that every point is fixed, from the same facts and excerpts and under the same task and rules. "
            "Return the complete JSON object in the same shape, or {\"skip\": <reason>} if these facts cannot carry a sound item."},
    ]


def build_unanswerable_check(questions: list[str], passages: list[Unit], corpus: Corpus, max_chars: int = 1200) -> list[dict]:
    shown = "\n\n".join(corpus.render(u, max_chars) for u in passages)
    qs = "\n".join(f"{i + 1}. {q}" for i, q in enumerate(questions))
    return [
        {"role": "system", "content":
            f"You check whether questions about one health technology appraisal (\"{corpus.topic}\") are answered by passages retrieved from its documents. "
            'Return only a JSON object: {"results": [{"n": int, "answered": "yes" | "partly" | "no", "passage_ids": [str, ...], "answer_if_any": str | null}, ...]} with one entry per question, in order. '
            "\"yes\" means a passage states the answer, the answer follows directly from what a passage states (a trial described as UK-based answers which country enrolled its patients), or the question is open enough that a passage on its topic answers it; "
            "\"partly\" means a passage gives a closely related fact but not the fact asked for; \"no\" means nothing here bears on it. "
            f"(prompt version {PROMPT_VERSION})"},
        {"role": "user", "content": f"Questions:\n{qs}\n\nPassages:\n{shown}"},
    ]
