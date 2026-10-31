"""A small in-memory bundle for the generate-stage tests: five documents, real TA1063 wording.

d01 cover page, d02 company submission, d03 EAG report, d04 clinical expert
statement (one long form table), d05 clarification questions and responses.
"""

import copy
import json
from pathlib import Path

from medrageval.config import load_config
from medrageval.generate.corpus import Corpus
from medrageval.text import to_plain

TABLE7 = (
    "<table><tr><th>Progression or death</th><th>Capivasertib + fulvestrant (N=155)</th><th>Placebo + fulvestrant (N=134)</th></tr>"
    "<tr><td>Median PFS (months)</td><td>7.3</td><td>3.1</td></tr>"
    "<tr><td>Events, n (%)</td><td>121 (78.1)</td><td>108 (80.6)</td></tr>"
    "<tr><td>PFS at 12 months (%)</td><td>31.2</td><td>12.4</td></tr>"
    "<tr><td>Hazard ratio (95% CI)</td><td>0.50 (0.38, 0.65)</td><td></td></tr></table>"
)

FORM_ROWS = [("1. Your name", "Charlotte Moss"), ("2. Name of organisation", "Royal Marsden NHS Foundation Trust")] + [
    (f"{n}. Question {n} about the technology and current practice?",
     f"Answer {n}: capivasertib plus fulvestrant would be used after an aromatase inhibitor in patients whose disease has a PI3K/AKT pathway alteration, "
     "which is a clear unmet need because current options such as everolimus plus exemestane are poorly tolerated in frail patients.")
    for n in range(3, 40)]
FORM = "<table>" + "".join(f"<tr><td>{q}</td><td>{a}</td></tr>" for q, a in FORM_ROWS) + "</table>"


def _el(doc, n, text, etype="paragraph", section=None, path=(), caption=None, page=1, source="native"):
    plain = to_plain(f"{caption}\n{text}" if caption else text)
    e = {"type": etype, "text": text, "page_start": page, "page_end": page, "bbox": [0, 0, 0, 0],
         "section_id": section, "section_path": list(path), "source": source, "redactions": plain.count("[REDACTED]"),
         "element_id": f"{doc}-e{n:04d}", "doc_id": doc, "plain": plain}
    if caption:
        e["caption"] = caption
    return e


ELEMENTS = [
    _el("d01", 1, "Committee Papers", path=["Committee Papers"], section="d01-s001"),
    _el("d01", 2, "Contents ........ 3", path=["Committee Papers"], section="d01-s001"),

    _el("d02", 1, "CAPItello-291 was a phase 3, randomised, double-blind, placebo-controlled trial of capivasertib plus fulvestrant in adults "
        "with HR-positive HER2-negative advanced breast cancer whose disease had progressed on an aromatase inhibitor.",
        section="d02-s001", path=["B.2 Clinical effectiveness", "B.2.3 Trial design"], page=10),
    _el("d02", 2, "Progression-free survival results by investigator assessment in the PI3K/AKT pathway-altered population are presented in Table 7, "
        "with a 50% reduction in the risk of progression or death (hazard ratio 0.50; 95% CI 0.38 to 0.65).",
        section="d02-s002", path=["B.2 Clinical effectiveness", "B.2.6 Clinical effectiveness results"], page=20),
    _el("d02", 3, TABLE7, "table", section="d02-s002", path=["B.2 Clinical effectiveness", "B.2.6 Clinical effectiveness results"],
        caption="**Table 7. PFS by investigator assessment in the PI3K/AKT pathway-altered population**", page=21),
    _el("d02", 4, "The list price of capivasertib is £[REDACTED] per pack of 64 tablets (excluding VAT), and the company estimates a "
        "total monthly drug cost of £[REDACTED] at the proposed dose.", section="d02-s003", path=["B.3 Cost effectiveness", "B.3.5 Cost and healthcare resource use"], page=40),
    _el("d02", 5, "1. Turner NC, Oliveira M, Howell SJ, et al. Capivasertib in hormone receptor-positive advanced breast cancer. N Engl J Med 2023;388:2058-70.",
        section="d02-s004", path=["B.4 References"], page=60),
    _el("d02", 6, "Page 3 of 146", section="d02-s001", path=["B.2 Clinical effectiveness"], page=3),
    _el("d02", 8, "The company's base-case ICER for capivasertib plus fulvestrant versus everolimus plus exemestane was £24,567 per QALY gained, "
        "with incremental QALYs of 0.61 and incremental costs of £14,986.", section="d02-s005", path=["B.3 Cost effectiveness", "B.3.10 Base-case results"], page=50),
    _el("d02", 9, "The proportional hazards assumption was not met for overall survival, which the company acknowledges underestimates uncertainty "
        "in the base-case ICER.", section="d02-s005", path=["B.3 Cost effectiveness", "B.3.10 Base-case results"], page=51),
    _el("d02", 10, "we are committed to meeting the requirements of copyright", section="d02-s001", path=["B.2 Clinical effectiveness"], page=2),
    _el("d02", 11, "In CAPItello-291, median overall survival was 26.0 months with capivasertib plus fulvestrant versus 22.3 months with placebo plus "
        "fulvestrant (hazard ratio 0.71), at 34% data maturity.", section="d02-s002", path=["B.2 Clinical effectiveness", "B.2.6 Clinical effectiveness results"], page=22),

    _el("d03", 1, "The EAG considers that the company's base-case ICER of £24,567 per QALY gained for capivasertib plus fulvestrant versus "
        "everolimus plus exemestane underestimates uncertainty, because the proportional hazards assumption was not met.",
        section="d03-s001", path=["5 Cost effectiveness", "5.2 EAG critique of the economic model"], page=90),
    _el("d03", 2, "The company stated that the rate of discontinuation of capivasertib due to adverse events was relatively low, at 13% in the "
        "capivasertib arm of CAPItello-291, which the EAG accepts.", section="d03-s002", path=["3 Clinical effectiveness", "3.2.5 Adverse events"], page=70),
    _el("d03", 3, "The company's base-case ICER for capivasertib plus fulvestrant versus everolimus plus exemestane was £24,567 per QALY gained, "
        "with incremental QALYs of 0.61 and incremental costs of £14,986.", section="d03-s001", path=["5 Cost effectiveness", "5.2 EAG critique of the economic model"], page=91),
    _el("d03", 4, "Treatment effect estimates are summarised in CS Table 7, and the EAG notes that median progression-free survival was "
        "7.3 months with capivasertib plus fulvestrant versus 3.1 months with placebo plus fulvestrant.",
        section="d03-s003", path=["3 Clinical effectiveness", "3.2.4 Efficacy results"], page=65),

    _el("d04", 1, FORM, "table", section="d04-s001", path=["Clinical expert statement"], page=120),

    _el("d05", 1, "A1. Priority question. Please provide the fulvestrant dosing schedule used in CAPItello-291?",
        section="d05-s001", path=["Clarification questions", "Company response:"], page=130),
    _el("d05", 2, "Patients in CAPItello-291 received fulvestrant 500 mg intramuscularly on days 1 and 15 of cycle 1 and then every 28 days, "
        "in line with UK clinical practice and the summary of product characteristics.",
        section="d05-s001", path=["Clarification questions", "Company response:"], page=130),
]


def _table(rows, header=None):
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    head = "<tr>" + "".join(f"<th>{c}</th>" for c in header) + "</tr>" if header else ""
    return f"<table>{head}{body}</table>"


AIC_CAPTION = "**Table {n}: AIC and BIC values for the parametric survival models fitted to the TTD capivasertib plus fulvestrant data ({arm} arm) CAPItello-291 (PI3K/AKT pathway-altered population, DCO1)**"
AIC_ROWS = [("Exponential", "608.6", "611.3"), ("Weibull", "610.5", "616.0"), ("Log-normal", "616.9", "622.4"), ("Gompertz", "610.2", "615.6")]
GROUPED = (
    "<table><tr><th>Study</th><th>Treatment arm</th><th>Sample size</th><th>Age</th></tr>"
    '<tr><td rowspan="2">FAKTION</td><td>Capivasertib + fulvestrant 500</td><td>69</td><td>62</td></tr>'
    "<tr><td>Fulvestrant 500</td><td>71</td><td>61</td></tr>"
    '<tr><td rowspan="2">BOLERO-2</td><td>Everolimus + exemestane</td><td>485</td><td>62</td></tr>'
    "<tr><td>Exemestane</td><td>239</td><td>61</td></tr>"
    "<tr><td>CONFIRM</td><td>Fulvestrant 500</td><td>362</td><td>61</td></tr></table>"
)
SUBGROUPS = _table([("All patients", "121/155 (78.1%)", "115/134 (85.8%)", "0.50 (0.38, 0.65)"), ("&lt;65 yéárs", "89/110 (80.9%)", "79/89 (88.8%)", "0.58 (0.43, 0.79)"),
                    ("≥65 years", "32/45 (71.1%)", "36/45 (80.0%)", "0.53 (0.33, 0.86)"), ("Asian", "35/48 (72.9%)", "30/35 (85.7%)", "0.59 (0.36, 0.96)")],
                   header=("", "Capivas ertib-fulvestrant", "Pl ace bo-fulvestrant", "Hazard ratio (95% CI)"))
_ECON = ["B.3 Cost effectiveness", "B.3.2 Economic analysis"]

# Elements for the selection rules added after the first smoke run: twin tables, a grouped table, OCR text, a figure stored
# as a table and a redacted value printed elsewhere. Kept apart from ELEMENTS so the seeded expectations above do not move.
EXTRA_ELEMENTS = [
    _el("d02", 20, _table(AIC_ROWS, ("Model", "AIC", "BIC")), "table", section="d02-s006", path=_ECON, caption=AIC_CAPTION.format(n=35, arm="capivasertib"), page=45),
    _el("d02", 21, _table(AIC_ROWS, ("Model", "AIC", "BIC")), "table", section="d02-s006", path=_ECON, caption=AIC_CAPTION.format(n=36, arm="fulvestrant"), page=46),
    _el("d02", 22, _table([(m, "0.9", "1.1") for m, _, _ in AIC_ROWS], ("Model", "Low", "High")), "table", section="d02-s006", path=_ECON, caption="**Table 40. Scenario results**", page=47),
    _el("d02", 23, _table([(m, "0.8", "1.2") for m, _, _ in AIC_ROWS], ("Model", "Low", "High")), "table", section="d02-s006", path=_ECON, caption="**Table 40. Scenario results**", page=48),
    _el("d02", 24, GROUPED, "table", section="d02-s006", path=_ECON, caption="**Table 16: Baseline data for the studies in the NMA**", page=49),
    _el("d02", 25, SUBGROUPS, "table", section="d02-s002", path=["B.2 Clinical effectiveness", "B.2.7 Subgroup analysis"], caption="**Table 10. Subgroup analyses of PFS**", page=25, source="ocr"),
    _el("d02", 26, SUBGROUPS.replace("yéárs", "years"), "table", section="d02-s002", path=["B.2 Clinical effectiveness", "B.2.7 Subgroup analysis"], caption="**Figure 10. Subgroup analyses of PFS**", page=26),
    _el("d02", 27, _table([("Median age; years (range)", "", "59.0 (34-90)"), ("Sex, n (%) (female)", "", "287 (99.3%)"), ("Body surface area (m2)", "Male", "[REDACTED]"), ("", "Female", "[REDACTED]")],
                          ("Characteristic", "", "PI3K/AKT pathway-altered population")), "table", section="d02-s007", path=["B.3 Cost effectiveness", "B.3.2.1 Patient population"],
        caption="**Table 14: Key demographics and baseline patient characteristics in the economic model**", page=78),
    _el("d02", 28, _table([("Age", "59.3 years", "48.5 – 71.7 (log-normal)"), ("Proportion female", "99.3%", "Fixed"), ("Body surface area female", "1.70", "1.39 – 2.06 (log-normal)"),
                           ("Body surface area male", "1.82", "1.49 – 2.02 (log-normal)")], ("Variable", "Value", "Distribution")), "table", section="d02-s007",
        path=["B.3 Cost effectiveness", "B.3.9.1 Summary of base-case analysis inputs"], caption="**Table 42. Key model variables for base case analysis**", page=115),
]

SECTIONS = [
    {"section_id": "d01-s001", "doc_id": "d01", "parent_id": None, "depth": 1, "title": "Committee Papers", "page": 1},
    {"section_id": "d02-s006", "doc_id": "d02", "parent_id": None, "depth": 1, "title": "B.3.2 Economic analysis", "page": 45},
    {"section_id": "d02-s007", "doc_id": "d02", "parent_id": None, "depth": 1, "title": "B.3.2.1 Patient population", "page": 78},
    {"section_id": "d02-s001", "doc_id": "d02", "parent_id": None, "depth": 1, "title": "B.2.3 Trial design", "page": 10},
    {"section_id": "d02-s002", "doc_id": "d02", "parent_id": None, "depth": 1, "title": "B.2.6 Clinical effectiveness results", "page": 20},
    {"section_id": "d02-s003", "doc_id": "d02", "parent_id": None, "depth": 1, "title": "B.3.5 Cost and healthcare resource use", "page": 40},
    {"section_id": "d02-s004", "doc_id": "d02", "parent_id": None, "depth": 1, "title": "B.4 References", "page": 60},
    {"section_id": "d02-s005", "doc_id": "d02", "parent_id": None, "depth": 1, "title": "B.3.10 Base-case results", "page": 50},
    {"section_id": "d03-s001", "doc_id": "d03", "parent_id": None, "depth": 1, "title": "5.2 EAG critique of the economic model", "page": 90},
    {"section_id": "d03-s002", "doc_id": "d03", "parent_id": None, "depth": 1, "title": "3.2.5 Adverse events", "page": 70},
    {"section_id": "d03-s003", "doc_id": "d03", "parent_id": None, "depth": 1, "title": "3.2.4 Efficacy results", "page": 65},
    {"section_id": "d04-s001", "doc_id": "d04", "parent_id": None, "depth": 1, "title": "Clinical expert statement", "page": 120},
    {"section_id": "d05-s001", "doc_id": "d05", "parent_id": None, "depth": 2, "title": "Company response:", "page": 130},
]

DOCUMENTS = [
    {"doc_id": "d01", "start_page": 1, "end_page": 2, "n_pages": 2, "signals": [], "headings": ["Committee Papers"], "running_text": None},
    {"doc_id": "d02", "start_page": 3, "end_page": 60, "n_pages": 58, "signals": [], "running_text": "Company evidence submission",
     "headings": ["NATIONAL INSTITUTE FOR HEALTH AND CARE EXCELLENCE", "Capivasertib with fulvestrant for treating hormone receptor-positive HER2-negative advanced breast cancer after endocrine treatment [ID6370]", "Document B"]},
    {"doc_id": "d03", "start_page": 61, "end_page": 100, "n_pages": 40, "signals": [], "running_text": None,
     "headings": ["Capivasertib in combination with fulvestrant for second-line treatment [ID6370]", "Produced by", "Rider on responsibility for report"]},
    {"doc_id": "d04", "start_page": 120, "end_page": 125, "n_pages": 6, "signals": [], "running_text": None,
     "headings": ["Capivasertib with fulvestrant for treating hormone receptor-positive HER2-negative advanced breast cancer after endocrine treatment [ID6370] Clinical expert statement", "Clinical expert statement"]},
    {"doc_id": "d05", "start_page": 130, "end_page": 140, "n_pages": 11, "signals": [], "running_text": None, "headings": ["Clarification questions"]},
]


def config() -> dict:
    cfg = copy.deepcopy(load_config()["generate"])
    cfg["attribution"]["rare_df_max"] = 0.3   # 20 units: "rare" must still allow a token in a handful of them
    cfg["unanswerable"]["stated_rare_df_max"] = 0.3
    cfg["max_workers"] = 2
    return cfg


def build_corpus(cfg: dict | None = None, elements=None) -> Corpus:
    return Corpus(Path("synthetic"), copy.deepcopy(DOCUMENTS), copy.deepcopy(SECTIONS), copy.deepcopy(elements or ELEMENTS), cfg or config())


def write_interim(path: Path, elements=None) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    with open(path / "elements.jsonl", "w") as f:
        for e in elements or ELEMENTS:
            f.write(json.dumps(e) + "\n")
    (path / "sections.json").write_text(json.dumps(SECTIONS))
    (path / "documents.json").write_text(json.dumps(DOCUMENTS))
    return path
