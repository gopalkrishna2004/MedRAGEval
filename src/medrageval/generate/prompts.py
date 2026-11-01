"""Prompts for item generation, the judge pass and the unanswerable check.

PROMPT_VERSION is written into the system message, so a prompt change
invalidates the response cache and is recorded in every item's provenance.
The prompts never name the appraisal or its documents themselves: the
corpus supplies the topic and document roles, so the same prompts work on
another bundle.
"""

import json

from medrageval.generate.candidates import Candidate
from medrageval.generate.corpus import Corpus, Unit

PROMPT_VERSION = "12"

SCHEMA = ('{"question": str, "answer": str, "answer_keys": [str, ...], '
          '"quotes": [{"unit_id": str, "quote": str}, ...]}')

RULES = """Rules for every item:
1. The question must be answerable from the excerpt(s) alone, and must stand on its own for a reader who has not seen them: name the drug, trial, population, comparator, outcome or document type involved. Never write "this table", "the excerpt", "above", "the following", or a document id ("d04"): the reader does not see ids. Keep it under 35 words.
2. The system answering will search all the documents listed above, not just these excerpts. Similar figures exist elsewhere in the bundle, so say what sets this one apart: the trial, population, arm, data cut, scenario, timepoint, or whose figure it is. A document often reports another party's work (an EAG report describes the company's analyses), so name an owner only as the excerpt or its context states it.
3. Put the question in your own words. Reuse the excerpt's wording only for names, numbers and technical terms, never its phrases: a question that can be answered by matching its words to the excerpt is too easy.
4. Do not put the answer in the question, and do not use knowledge from outside the excerpts. Never ask for a value shown as [REDACTED]: it is not disclosed, so it cannot be an answer.
5. "answer": one sentence at most, specific. "answer_keys": 1 to 4 strings that a correct answer must contain and that appear verbatim in your quotes; each is a number with its unit, a name, or a phrase of at most four words, never a clause. Each key answers the question as asked: leave out facts the question does not ask for, and anything the question already says.
6. "quotes": entries of {"unit_id": the id shown in brackets, "quote": a contiguous span copied verbatim from that excerpt, 5 to 80 words}. The quote must contain the fact the answer rests on. From a table, quote one row per entry (several entries are fine); never join rows or cells from different lines.
7. Prefer facts specific to this appraisal (results, assumptions, costs, criticisms, positions) over boilerplate and generic background.
8. Text headed "Context" is the passage before an excerpt in its section, shown so you know what the excerpt is about (its trial, population, arms, whose analysis). Never quote it or ask about its facts.
9. If the excerpt cannot carry a sound item of this type (garbled text, a trivial or generic fact, rows or values that cannot be told apart, a comparison that means nothing), return {"skip": <one short reason>} instead of forcing one.
Return only a JSON object."""

TYPE_INSTRUCTIONS = {
    "factual": "Write a FACTUAL question: it asks for one fact that a single sentence of the excerpt states (a definition, a design feature, a decision, a reason).",
    "numerical": "Write a NUMERICAL question: the answer is a specific number with its unit (a percentage, a count, a cost, a duration, a ratio) taken from the excerpt. "
                 "The number must be one of these, written as in the excerpt: {numbers}. A range (40-50%) is one value: ask for the range, never for one end of it. "
                 "Put the number with its unit in answer_keys. Say which measure, arm and population it belongs to in the question.",
    "comparative": "Write a COMPARATIVE question: it compares two values of the SAME measure for two {what}. Use these two: {labels}. "
                   "Name both compared things in the question (the two arms, groups, timepoints or scenarios), and the measure; if the table holds several timepoints, subgroups or measures, say which one you mean, so exactly one pair of values fits. "
                   "The answer must say which is higher (or lower) and give both values. Add the fields \"measure\", \"label_a\", \"value_a\", \"label_b\", \"value_b\" (the two values as written) and \"higher\" (\"a\", \"b\" or \"equal\"). "
                   "Quote each row holding a value as its own entry; add the header row as another entry if the labels are only there.",
    "multihop": "Write a MULTI-HOP question with a bridge. The bridge is {link}: excerpt A describes or characterises it, excerpt B states a value or fact about it. "
                "Ask for the value or fact in B, but refer to the bridge ONLY through A's description (what it is, how it was chosen, what was said about it){bridge}. {shown}"
                "Describe it by a property from that description, never by its place in a list or the order in which A names things (\"the second trial mentioned\" is not a description). "
                "The question must not contain the bridge's name as written in B ({link}), nor the table or figure number, nor B's row or column label verbatim. "
                "The answer is a value or fact B states about the bridge, never the bridge itself, and it must be the only thing in B that fits the question (if B names several studies or values, say which one you mean). "
                "Test yourself: a reader holding only B must be unable to tell which row or value is asked for; a reader holding only A must not know the answer. "
                "Add \"hop_a\" and \"hop_b\": one short sentence each on what each excerpt contributes. Quote from both excerpts.",
    "crossdoc": "Write a CROSS-DOCUMENT question: the two excerpts come from different documents and concern the same figure, assumption or claim ({link}). "
                "Ask what the two documents say about it, or what document A concluded about document B's figure. The answer must draw on both, and you must quote from both excerpts.",
    "attribution": "Write an ATTRIBUTION question: the excerpt contains a position, argument or claim made by {speaker}. Ask WHO made it (\"Who argued that ...?\", \"Which party considered that ...?\"), "
                   "stating the claim precisely enough to identify it, but do NOT name or hint at the speaker in the question. "
                   "Keep the part of the claim that is this party's own (its argument, judgement, survey or experience), not background any party in the bundle could state (how severe the disease is, what the trial was). "
                   "State the claim as a position any of the parties in the bundle could have taken: no first person, no wording that only the speaker would use (\"our model\", \"the submission\", \"the EAG's critique\"). "
                   "The answer must be exactly \"{speaker}\" and answer_keys must be [\"{speaker_key}\"]. Quote the sentence that carries the claim.",
    "unanswerable_redacted": "Write an UNANSWERABLE question about a value that is redacted in the excerpt (shown as [REDACTED]); candidates: {labels}. "
                             "Ask for that value exactly as a reader would who expects it to be there (\"What was the ...?\"). The correct response of a QA system is that the value is not disclosed. "
                             "Set \"answer\" to null and \"answer_keys\" to []. Quote the row or sentence containing the [REDACTED] marker, including the marker.",
    "unanswerable_not_covered": "Write 3 UNANSWERABLE questions anchored in the excerpt: specific questions a reader of this appraisal ({topic}) would plausibly ask, on the same subject as the excerpt, whose answers are NOT in the excerpt or the nearby passages shown. "
                                "Each is a near miss: it asks for the kind of fact these documents do report (a result, rate, count, cost, parameter or definition) for a version of the excerpt's subject that the bundle does not cover: another timepoint or data cut, subgroup, endpoint, dose, scenario or comparator. A system that answers it with the nearest figure it finds is wrong. "
                                "Each asks for ONE specific fact, phrased as if the answer existed. The fact must not follow from what the documents do say (do not ask which country enrolled most patients in a trial the documents call UK-based). Not: software, files, company finances, manufacturing, addresses, personnel. "
                                "Never ask open questions that any passage on the topic answers (\"which issue did the EAG raise\"). "
                                "Return {{\"questions\": [{{\"question\": str, \"missing_fact\": str}}, ...]}} and nothing else.",
}


def _doc_list(corpus: Corpus) -> str:
    return "\n".join(f"- {d.doc_id}: {d.label} (pages {d.start_page}-{d.end_page})" for d in corpus.docs.values() if d.role != "cover")


def system_message(corpus: Corpus) -> str:
    return (f"You write evaluation items for a question-answering system over a bundle of documents from one health technology appraisal: "
            f"\"{corpus.topic}\".\nThe documents in the bundle are:\n{_doc_list(corpus)}\n\n"
            f"You will be given one or two excerpts, each headed by its id in brackets, document, section and pages. "
            f"Write exactly what the task asks, as one JSON object with this shape: {SCHEMA} (plus any extra fields the task names).\n\n{RULES}\n"
            f"(prompt version {PROMPT_VERSION})")


def _context(u: Unit, corpus: Corpus, cfg: dict) -> str:
    """The passages before the excerpt in its section, without unit ids so they cannot be quoted."""
    cc = cfg["context"]
    texts = []
    for p in corpus.preceding(u, cc["units"]):
        t = p.plain if len(p.plain) <= cc["max_chars"] else "[...] " + p.plain[-cc["max_chars"]:]
        texts.append(t)
    return "\n".join(texts)


def _excerpts(cand: Candidate, corpus: Corpus, cfg: dict, max_chars: int = 3500) -> str:
    parts = []
    for i, u in enumerate(cand.units):
        tag = f"Excerpt {'AB'[i]}" if len(cand.units) > 1 else "Excerpt"
        body = corpus.render(u, max_chars)
        if cand.type == "comparative" and cand.hint.get("mode") in ("columns", "rows"):
            head = body.split("\n", 1)[0]
            cap = f"{u.caption}\n" if u.caption else ""
            body = f"{head}\n{cap}{cand.hint['header']}\n" + "\n".join(cand.hint["rows"])
        context = _context(u, corpus, cfg) if cand.subtype != "not_covered" else ""
        parts.append((f"Context before {tag.lower()} (not quotable):\n{context}\n\n" if context else "") + f"{tag}:\n{body}")
    return "\n\n".join(parts)


def build_generation(cand: Candidate, corpus: Corpus, cfg: dict) -> list[dict]:
    h = cand.hint
    key = cand.type if cand.type != "unanswerable" else f"unanswerable_{cand.subtype}"
    fmt = dict(numbers=", ".join(h.get("numbers", [])[:25]), labels=" / ".join(h.get("labels", [])),
               what="arms, groups, timepoints or scenarios" if h.get("mode") != "rows" else "rows (treatments, states or categories) on the measure " + repr(h.get("measure", "")),
               link=h.get("link", ""), bridge=f': "{h["bridge"]}"' if h.get("bridge") else "", speaker=h.get("speaker", ""), speaker_key=h.get("speaker", "").removeprefix("the ").strip(),
               shown=("Excerpt B itself already shows, next to the bridge: " + "; ".join(f'"{s}"' for s in h["shown"])
                      + ". Do not describe the bridge by any of these or by words meaning the same: a reader holding only B would find the row from them. "
                      "Use what A says about it that B does not. ") if h.get("shown") else "",
               topic=h.get("topic", corpus.topic))
    if cand.type == "comparative" and h.get("mode") == "prose":
        fmt["labels"] = "the two things the excerpt compares, using numbers " + ", ".join(h.get("numbers", [])[:10])
    task = TYPE_INSTRUCTIONS[key].format(**fmt)
    if cand.type == "attribution" and h.get("claim"):
        task += f"\nThe claim to use: \"{h['claim'][:400]}\""
    if h.get("must_mention"):
        phrases = ", ".join(f"\"{p}\"" for p in h["must_mention"])
        task += (f"\nThe bundle has other tables with exactly the same rows. The question must contain the words {phrases} "
                 f"(from this table's caption, or the document it is in), so a reader knows which table is meant.")
    body = _excerpts(cand, corpus, cfg)
    if cand.subtype == "not_covered":
        near = corpus.neighbours(cand.units[0], cfg["unanswerable"]["neighbours"])
        if near:
            body += "\n\nNearby passages (their facts are also off limits):\n" + "\n\n".join(corpus.render(u, 800) for u in near)
    return [{"role": "system", "content": system_message(corpus)},
            {"role": "user", "content": f"{task}\n\n{body}"}]


JUDGE_CRITERIA = {
    "factual": "The answer is stated by the quotes. The question is specific enough that the quoted sentence is the natural answer.",
    "numerical": "The answer's number and unit appear in the quotes and belong to the measure, arm and population the question names.",
    "comparative": "Both values appear in the quotes, the direction (higher/lower) is correct, and the two labels are the right ones. The question is not answerable from one value alone.",
    "multihop": "The question identifies its subject by a property, description or role (hop A) and asks for a value or fact about that subject (hop B). Pass when the identifying property is stated in one excerpt and the value in the other, and the value excerpt does not itself state that property. "
                "The value excerpt naming the subject (the trial's or model's name) does NOT make it sufficient on its own: without the other excerpt a reader cannot tell that this named subject is the one the question describes. Fail if one excerpt contains both the identifying property and the value, if the question names the subject outright, or if it identifies the subject only by its position in a list (\"the second trial named\").",
    "crossdoc": "The quotes come from two different documents and the answer reports what each says about the same figure, assumption or claim. It is fine that one document restates the other's figure; fail only if the answer is not supported by the quotes or the question does not say which documents and subject it is about.",
    "attribution": "The quoted claim is made by the party given as the answer, the question identifies that claim unambiguously, and the question does not name or hint at the party.",
    "unanswerable": "The question asks for a value that is redacted in the quote, as a reader who expected it to be present would ask.",
}


def build_judge(item: dict, cand: Candidate, corpus: Corpus, cfg: dict) -> list[dict]:
    units = {u.unit_id: u for u in cand.units}
    shown = "\n\n".join(corpus.render(units[q["unit_id"]], cfg["judge"]["max_chars_per_unit"]) for q in item.get("quotes", []) if q.get("unit_id") in units) \
        or "\n\n".join(corpus.render(u, cfg["judge"]["max_chars_per_unit"]) for u in cand.units)
    payload = {k: item.get(k) for k in ("question", "answer", "answer_keys", "quotes") if k in item}
    return [
        {"role": "system", "content":
            f"You check evaluation items for a question-answering system over documents of one health technology appraisal: \"{corpus.topic}\". "
            "Be strict: an item passes only if every criterion holds. Return only a JSON object: "
            '{"pass": bool, "answer_supported": bool, "self_contained": bool, "type_fits": bool, '
            '"answerable_from_single_excerpt": {"<unit_id>": bool, ...}, "issues": [str, ...]}. '
            "\"self_contained\" means a reader with access to the whole document bundle but not to these excerpts could tell what is being asked. "
            "\"answerable_from_single_excerpt\" gives, for each excerpt id, whether that excerpt alone suffices: it does only if that excerpt contains both the information the question uses to identify its subject and the answer. An excerpt that merely names the subject and gives the value, while the question identifies the subject by a property stated elsewhere, does not suffice. "
            f"(prompt version {PROMPT_VERSION})"},
        {"role": "user", "content":
            f"Item type: {cand.type}\nCriteria: {JUDGE_CRITERIA[cand.type]}\n\n"
            f"Item:\n{json.dumps(payload, ensure_ascii=False, indent=1)}\n\nSource excerpts:\n{shown}"},
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
