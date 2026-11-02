"""Prompts for the RAG's answers and for the judge that grades them.

The answer prompt is the same for every question: a deployed RAG does not
know an item's type, so the one under evaluation is not told either. It asks
for the exact figure with its unit, citations, and one fixed sentence when the
passages do not answer, so abstention can be scored. It warns about
[REDACTED] values because the bundle is full of them (2,356 in TA1063) and a
small model otherwise reads a neighbouring cell instead.

The judge sees what the RAG saw. It grades correctness against the reference
answer and its keys, and groundedness against the passages, so a wrong
answer can be put down to retrieval (the evidence was not shown) or to the
reader (it was, and the model got it wrong).

PROMPT_VERSION is in both system messages, so a change invalidates the
response cache and is recorded in the run report.
"""

from medrageval.rag.chunks import Chunk

PROMPT_VERSION = "1"
ABSTAIN = "Not found in the documents."


def build_answer(question: str, passages: list[Chunk], topic: str) -> list[dict]:
    system = (f"You answer questions about one NICE health technology appraisal (\"{topic}\") using only the numbered passages "
              "from its committee papers. Each passage starts with its document, section and pages.\n"
              "- Answer in one or two sentences. Give figures exactly as the passages print them, with their units.\n"
              "- If the question asks for an overview, answer in a short paragraph that covers each relevant point.\n"
              "- Say whose position or analysis it is (the company, the EAG, the committee, an expert) when the passage says.\n"
              "- Cite the passages you used, like [2].\n"
              "- Values printed as [REDACTED] are confidential: never guess them or take a neighbouring value instead.\n"
              f"- If the passages do not answer the question, reply exactly: {ABSTAIN}\n"
              "- Do not use knowledge from outside the passages.\n"
              f"(prompt version {PROMPT_VERSION})")
    context = "\n\n".join(f"[{i}] {c.text}" for i, c in enumerate(passages, start=1))
    return [{"role": "system", "content": system},
            {"role": "user", "content": f"Passages:\n\n{context}\n\nQuestion: {question}"}]


JUDGE_SYSTEM = """You grade the answer of a question-answering system over the committee papers of one NICE health technology appraisal ("{topic}").
You get the question, a reference answer written from the documents with the keys a correct answer must state, the system's answer, and the passages the system was shown.
Return only a JSON object:
{{"verdict": "correct" | "partial" | "incorrect" | "abstained", "keypoints": [{{"n": int, "status": "covered" | "contradicted" | "missing"}}], "grounded": true | false | null, "unsupported_claims": [str], "reason": str}}

verdict, for a question the documents answer:
- "correct": the answer states every answer key, or an equivalent (the same number at the printed precision with the same unit; a paraphrase of a phrase), for the subject the question asks about. Extra true detail is fine.
- "partial": some keys are stated and none is contradicted, or every key is stated but alongside a wrong claim about the asked subject.
- "incorrect": a key is missing and something else is given in its place, a key is contradicted, or the answer is about a different subject (another arm, data cut, population, scenario or party).
- "abstained": the answer declines or says the documents do not give it, and states no answer.
For a comparison, the answer must also say which side is higher or lower, as the reference does.
For a "which party" question the key is the party: naming it as the holder of the view is correct; naming a different party as the holder is incorrect.
For a summary question the keypoints decide: "correct" when all are covered, "partial" when at least half are covered, "incorrect" otherwise.
For a question the documents do not answer: "correct" when the answer declines, says the information is not reported, or says it is redacted; "incorrect" when it states a specific answer to the question.

keypoints: one entry per numbered keypoint when keypoints are given, else [].
grounded: true when every factual claim in the answer is supported by the passages, false when any is not, null when the answer only declines. Judge this against the passages, not the reference.
unsupported_claims: the answer's claims the passages do not support, each in a few words.
reason: one sentence.
(prompt version {version})"""


def _reference(item: dict) -> str:
    if not item.get("answerable", True):
        return "Reference: the documents do not answer this question (the value is redacted or not reported)."
    lines = [f"Reference answer: {item.get('answer') or '-'}"]
    if item.get("answer_keys"):
        lines.append("Answer keys: " + "; ".join(f'"{k}"' for k in item["answer_keys"]))
    if item.get("answer_aliases"):
        lines.append("Accepted names for the party: " + "; ".join(item["answer_aliases"]))
    if item.get("keypoints"):
        lines.append("Keypoints:\n" + "\n".join(f"{n}. {kp.get('point')}" for n, kp in enumerate(item["keypoints"], start=1)))
    return "\n".join(lines)


def build_judge(item: dict, response: str, passages: list[Chunk], topic: str, max_chars: int) -> list[dict]:
    def cut(text: str) -> str:
        return text if len(text) <= max_chars else text[:max_chars] + "\n[...]"
    shown = "\n\n".join(f"[{i}] {cut(c.text)}" for i, c in enumerate(passages, start=1)) or "(none)"
    user = (f"Question: {item['question']}\n\n{_reference(item)}\n\n"
            f"System answer:\n{response.strip() or '(empty)'}\n\n"
            f"Passages shown to the system:\n{shown}")
    return [{"role": "system", "content": JUDGE_SYSTEM.format(topic=topic, version=PROMPT_VERSION)},
            {"role": "user", "content": user}]
