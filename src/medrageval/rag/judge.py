"""The judge prompt: a frontier model grades the RAG's answer against the item, seeing what the RAG saw.

It grades correctness against the reference answer and its keys, and
groundedness against the passages, so a wrong answer can be put down to
retrieval (the evidence was not shown) or to the reader (it was, and the
model got it wrong). It also lists the reply's factual claims, each marked
supported or not, from which the faithfulness score (supported share) is
computed. JUDGE_VERSION is in the system message, so a change
invalidates the verdict cache and is recorded in the run report.
"""

from medrag.chunk import Chunk

JUDGE_VERSION = "4"

JUDGE_SYSTEM = """You grade the answer of a question-answering system over the documents of {topic}.
You get the question, a reference answer written from the documents with the keys a correct answer must state, the system's answer, and the passages the system was shown.
Return only a JSON object:
{{"verdict": "correct" | "partial" | "incorrect" | "abstained", "keypoints": [{{"n": int, "status": "covered" | "contradicted" | "missing"}}], "claims": [{{"claim": str, "supported": true | false}}], "grounded": true | false | null, "reason": str}}

verdict, for a question the documents answer:
- "correct": the answer states every answer key, or an equivalent (the same number at the printed precision with the same unit; a paraphrase of a phrase), for the subject the question asks about. Extra true detail is fine.
- "partial": some keys are stated and none is contradicted, or every key is stated but alongside a wrong claim about the asked subject.
- "incorrect": a key is missing and something else is given in its place, a key is contradicted, or the answer is about a different subject (another arm, data cut, population, scenario or party).
- "abstained": the answer declines or says the documents do not give it, and states no answer.
For a comparison, the answer must also say which side is higher or lower, as the reference does.
For a "which party" question the key is the party: naming it as the holder of the view is correct; naming a different party as the holder is incorrect.
For a summary question the keypoints decide: "correct" when all are covered, "partial" when at least half are covered, "incorrect" otherwise.
For a question the passages do not answer: "correct" when the answer declines, or says the information is not given, not reported or redacted; "incorrect" when it states a specific answer to the question, whether or not it happens to be right. The verdict turns on whether a specific answer is stated, not on whether the explanation for declining is accurate: a wrong explanation goes under claims.

keypoints: one entry per numbered keypoint when keypoints are given, else [].
claims: the factual claims the answer makes about the documents' content (a figure, a comparison, an attribution, a stated fact), at most 12, each in a few words with "supported": true when a passage shown states it and false when no passage does. A decline, or a statement that the passages or documents do not give, report or contain something, is not a claim: [] when the answer only declines. Judge this against the passages, not the reference: a true claim that no passage states is unsupported.
grounded: true when every claim is supported, false when any is not, null when there are no claims.
reason: one sentence.
(judge version {version})"""


def _reference(item: dict) -> str:
    if not item.get("answerable", True):
        return ("Reference: the passages shown do not answer this question (its evidence was withheld from the system, "
                "or the documents do not give it). The correct response declines.")
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
    return [{"role": "system", "content": JUDGE_SYSTEM.format(topic=topic, version=JUDGE_VERSION)},
            {"role": "user", "content": user}]
