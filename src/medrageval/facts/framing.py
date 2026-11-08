"""How the prompts speak of a document family.

What the documents are, who asks about them, whose analyses they report,
who types the messy questions, and the examples the prompt rules give. A
schema sets these in its `framing:` block; anything it leaves out takes the
neutral default here, so a schema for one family never inherits another
family's wording. The NICE appraisal wording lives in that schema's file.
"""

from dataclasses import dataclass, field
from types import MappingProxyType

# The examples the prompt rules give, keyed by the rule they illustrate. Neutral by default; a schema may give its
# family's own (the NICE schema gives the TA1063 examples the prompts were written with).
EXAMPLES = MappingProxyType({
    # extraction: abbreviations to expand, identifiers to keep, numbers written with their unit
    "extract_shorthand": '("Drug A + B" -> "drug A plus drug B", "TFSC" -> "time to first subsequent chemotherapy or death")',
    "identifiers": "DCO1, NCT01234567",
    "numbers": '("7.3 months", "40.0%", "2.7 mL/min", "0.50 (95% CI 0.38 to 0.65)")',
    # generation rules: an abbreviation spelled out, a regimen named once
    "spell_out": "(\"the trial's primary efficacy outcome\", not \"the trial's PEO\")",
    "regimen": '("drug A plus drug B", not "Drug A + drug B (Drug A with drug B)")',
    # generation tasks: a description standing in for a name, a speaker's own voice, an open question, table shorthand
    "described": "(\"the population the authors enrolled for ...\", \"the authors' analysis that changes ...\")",
    "voice": '("our trial", "we found", "this report")',
    "summary": '("Which adverse events did the trial report with drug X?", not "How did the report present safety?")',
    "table_shorthand": '("Drug A + B" is drug A plus drug B)',
    # rewrite feedback: an answer key without its row label
    "key_label": '("0", not "Placebo: 0")',
})

# Messy-question habits whose wording names things of one family; the other habits are the same for every family.
HABITS = MappingProxyType({
    "shorthand": "clinical and texting shorthand where a colleague would use it (pts, tx, w/, vs, b/c, abt, HR, AEs, pbo)",
    "statement": "a request or statement instead of a question (\"need the response rate for ...\", \"looking for the authors' figure on ...\")",
    "vague": "one specific detail blurred into something vaguer (\"the new drug\", \"that trial\", \"the authors or whoever it was\"), "
             "still enough for someone who knows the documents to tell what is meant",
    "misspell_names": "drug, trial and party names spelled as they sound or half-remembered (atorvastaton, metformine, the flourish trial), never corrected",
})

STRINGS = ("subject", "bundle", "kind", "askers", "askers_short", "purpose", "abbreviations", "standard_abbreviations")
LISTS = ("owners", "personas")
MAPPINGS = {"examples": EXAMPLES, "habits": HABITS}


@dataclass(frozen=True)
class Framing:
    subject: str = "one medical topic"                    # "the documents of {subject}"
    bundle: str | None = None                             # "a question-answering system over {bundle}"; default: "the documents of {subject}"
    kind: str | None = None                               # "questions about {kind}"; default: the subject with "one" made "a"
    askers: str = "a clinician, researcher or other reader of these documents"   # "Ask as {askers} would"
    askers_short: str | None = None                       # the same people in the rewrite feedback; default: askers
    purpose: str = "the topic"                            # "the answer matters to {purpose}", "what matters in {purpose}"
    abbreviations: str = "PFS, OS, HR, CI"                # abbreviations a question may use unexpanded
    standard_abbreviations: str = "PFS, OS, HR, CI"       # abbreviations extraction keeps as they are
    owners: tuple[str, ...] = ()                          # whose analysis an "owner" qualifier names ("company", "EAG")
    personas: tuple[str, ...] = ()                        # who types the messy questions; empty: the config's personas
    examples: dict[str, str] = field(default_factory=dict)
    habits: dict[str, str] = field(default_factory=dict)

    @property
    def bundle_text(self) -> str:
        return self.bundle or f"the documents of {self.subject}"

    @property
    def kind_text(self) -> str:
        if self.kind:
            return self.kind
        return "a " + self.subject[4:] if self.subject.startswith("one ") else self.subject

    @property
    def askers_short_text(self) -> str:
        return self.askers_short or self.askers

    def example(self, key: str) -> str:
        return self.examples.get(key) or EXAMPLES[key]

    def habit(self, key: str, default: str) -> str:
        return self.habits.get(key) or HABITS.get(key) or default

    def owners_clause(self) -> str:
        """' ("company" or "EAG")' after "whose analysis it is", or nothing when the schema names no owners."""
        return " (" + " or ".join(f'"{o}"' for o in self.owners) + ")" if self.owners else ""


def parse_framing(data, error) -> Framing:
    """A Framing from a schema's `framing:` mapping; `error(message)` builds the exception to raise."""
    if data is None:
        return Framing()
    if not isinstance(data, dict):
        raise error("framing: expected a mapping")
    known = set(STRINGS) | set(LISTS) | set(MAPPINGS)
    if unknown := sorted(set(data) - known):
        raise error(f"framing: unknown keys {unknown}; known: {sorted(known)}")
    kw = {}
    for k in STRINGS:
        if k in data and data[k] is not None:
            if not isinstance(data[k], str) or not data[k].strip():
                raise error(f"framing.{k}: expected a non-empty string")
            kw[k] = " ".join(data[k].split())
    for k in LISTS:
        if k in data and data[k] is not None:
            if not isinstance(data[k], list) or not all(isinstance(v, str) and v.strip() for v in data[k]):
                raise error(f"framing.{k}: expected a list of strings")
            kw[k] = tuple(" ".join(v.split()) for v in data[k])
    for k, defaults in MAPPINGS.items():
        if k in data and data[k] is not None:
            m = data[k]
            if not isinstance(m, dict) or not all(isinstance(v, str) and v.strip() for v in m.values()):
                raise error(f"framing.{k}: expected a mapping of strings")
            if unknown := sorted(set(m) - set(defaults)):
                raise error(f"framing.{k}: unknown keys {unknown}; known: {sorted(defaults)}")
            kw[k] = {key: " ".join(v.split()) for key, v in m.items()}
    return Framing(**kw)
