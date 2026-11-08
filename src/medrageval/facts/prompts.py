"""Prompts for fact extraction and for drafting a schema.

EXTRACT_PROMPT_VERSION is in the system message, so changing a prompt
invalidates the response cache and is recorded in the extraction report. Like
the generation prompts, these never name a document family: the corpus
supplies the topic and document labels, and the schema's framing what the
documents are and the examples the rules give.
"""

import json

from medrageval.facts.framing import Framing
from medrageval.facts.schema import Group, Slot
from medrageval.generate.corpus import Corpus, Unit

EXTRACT_PROMPT_VERSION = "3"


def slot_line(slot: Slot) -> str:
    if slot.kind == "number":
        if slot.per:
            # v1 asked for "value" or "values" and got per-technology totals as separate facts without their technology:
            # £51,365 and £25,714 with identical qualifiers, which no question can tell apart
            shape = (f'"values": {{<{slot.per}>: value with its unit, ...}}, keyed by the {slot.per} as written, '
                     f'with every {slot.per} the row or sentence gives; use this shape even for one {slot.per}')
        else:
            shape = '"value": the value with its unit'
    elif slot.kind == "text":
        shape = '"value": a statement of at most 30 words, in the excerpt\'s own terms'
    else:
        shape = f'"positions": {{<party>: that party\'s position in at most 30 words}} for each of {", ".join(slot.parties)} the excerpt reports'
    quals = ", ".join(slot.qualifiers) or "none"
    return f"- {slot.name}: {slot.describe}. Shape: {shape}. Qualifiers: {quals}."


EXTRACT_RULES = """Rules:
1. Record only what an excerpt states outright. Never use outside knowledge, never infer, convert or calculate.
2. "quote": a contiguous span copied verbatim from the excerpt whose id you give, 5 to 80 words, that holds the value. From a table, quote the one row that holds it (a short row may have fewer than 5 words). Never join text from different rows or excerpts.
3. "qualifiers": fill each of the slot's qualifiers that is stated: the trial, endpoint, measure, population, arm, data cut, timepoint, scenario, and whose analysis it is{owners} for "owner". The trial, population and owner may come from the section heading or the excerpts just before it in the same section (a results table under a heading about one trial is that trial's). A data cut, timepoint or scenario must be stated in the excerpt itself or its table caption, never taken from a neighbouring excerpt. Leave out qualifiers nothing states. Never put the value itself in a qualifier. Name the same thing the same way in every fact: the trial as the document writes it, an endpoint by the abbreviation the document uses if it uses one (PFS, OS).
4. Write qualifiers and the labels of "values" as a reader would say them, not as a table prints them. Expand abbreviations the document defines {extract_shorthand}, and turn table notation into words or drop it ("Pleural effusion (SAE)" -> "serious adverse event of pleural effusion"; drop "n (%)", "(N=155)", "(LY)"). Keep standard abbreviations ({standard_abbreviations}) and identifiers ({identifiers}) as they are.
5. Numbers: write each value as natural text with its unit {numbers}, using only numbers that appear in the quote. A value shown as [REDACTED] is recorded as "[REDACTED]".
6. One fact per distinct value: never repeat a value for the same qualifiers. Skip form instructions, references, tables of contents, boilerplate, and general background on the disease.
7. At most {max_facts} facts, the most important first: headline results, base-case figures and key disagreements before minor details.
Return only the JSON object."""


def extract_rules(max_facts: int, fr: Framing) -> str:
    return EXTRACT_RULES.format(max_facts=max_facts, owners=fr.owners_clause(), extract_shorthand=fr.example("extract_shorthand"),
                                standard_abbreviations=fr.standard_abbreviations, identifiers=fr.example("identifiers"), numbers=fr.example("numbers"))


def build_extraction(group: Group, units: list[Unit], corpus: Corpus, doc_id: str, max_facts: int, max_unit_chars: int,
                     framing: Framing | None = None) -> list[dict]:
    fr = framing or Framing()
    doc = corpus.docs[doc_id]
    system = (f"You build a fact sheet for a question-answering benchmark over the documents of {fr.subject}: \"{corpus.topic}\".\n"
              f"You are given excerpts from one document ({doc.label}), each headed by its id in brackets, section and pages. "
              "Record the facts they state for the slots listed, as one JSON object: "
              '{"facts": [{"slot": str, "unit_id": str, "quote": str, "qualifiers": {str: str}, "value": str}, ...]} '
              '(with "values" or "positions" in place of "value" where the slot\'s shape says so).\n\n'
              f"{extract_rules(max_facts, fr)}\n(extract prompt version {EXTRACT_PROMPT_VERSION})")
    slots = "\n".join(slot_line(s) for s in group.slots.values())
    excerpts = "\n\n".join(corpus.render(u, max_unit_chars) for u in units)
    return [{"role": "system", "content": system},
            {"role": "user", "content": f"Group: {group.describe}\nSlots:\n{slots}\n\nExcerpts:\n{excerpts}"}]


SCHEMA_SHAPE = {
    "name": "document family name",
    "version": 1,
    "framing": {
        "subject": "what the documents are, as 'one ...' (e.g. 'one clinical trial report')",
        "askers": "who would ask questions about such documents (e.g. 'a clinician or trial statistician')",
        "purpose": "what the answers matter to (e.g. 'the trial's conclusions')",
        "owners": ["the parties whose analyses the documents report, when slots carry an 'owner' qualifier; else []"],
        "personas": ["4 to 6 people who would type questions about such documents into a chat box"],
    },
    "groups": {
        "<group_id>": {
            "weight": "share of the dataset, 0-1",
            "describe": "what the group covers",
            "roles": ["document roles it is read from"],
            "sections": ["regexes over section titles"],
            "keywords": ["search terms"],
            "slots": {"<slot_id>": {"kind": "number | text | stance", "per": "for number slots with one value per arm/comparator: a lower_snake_case id (optional)",
                                     "parties": ["for stance slots"], "describe": "one sentence",
                                     "qualifiers": ["lower_snake_case ids of what tells two facts apart (data_cut, study_arm)"]}},
        }
    },
    "links": [{"from": "group.slot whose value names ...", "to": "group.slot ... qualified by it", "via": "the name of a qualifier (or the `per` key) of `to`",
               "describe": "how a question refers to the bridge, with {qualifier} placeholders from `from`",
               "match": "optional: 'qualifier' when a qualifier of `from` (named in source_via) names the thing and the value of `from` describes it; "
                        "describe may then use {value}"}],
}


def build_schema_draft(topic: str, outlines: dict[str, list[str]], samples: dict[str, list[str]], roles: list[str]) -> list[dict]:
    parts = []
    for role, titles in outlines.items():
        parts.append(f"## Role: {role}\nSection titles:\n" + "\n".join(titles))
        if samples.get(role):
            parts.append("Sample passages:\n" + "\n---\n".join(samples[role]))
    return [
        {"role": "system", "content":
            "You design the fact-sheet schema for a question-answering benchmark over a family of documents. "
            "A schema lists groups of slots: the kinds of fact a reader of such documents asks about (results, design, costs, positions of the parties), "
            "the document roles each group is found in, and the qualifiers that tell two facts of one slot apart (trial, population, arm, timepoint, scenario, whose analysis). "
            "Slots must be generic to the document family, never specific to one disease or product: 'the result of a trial endpoint per arm', not 'median PFS'. "
            "Use 6 to 10 groups with weights summing to 1, favouring what decision makers care about most. "
            "Fill \"framing\" in plain words for this family: it is how the benchmark's prompts speak of the documents and their readers. "
            f"Use only these roles: {', '.join(roles)}. Return only a JSON object of this shape: {json.dumps(SCHEMA_SHAPE)}"},
        {"role": "user", "content": f"Example bundle: \"{topic}\".\n\n" + "\n\n".join(parts)},
    ]
