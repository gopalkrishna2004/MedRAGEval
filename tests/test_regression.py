"""End-to-end checks against facts verified by hand in the two real PDFs.

Run with: uv run pytest -m regression
The first run converts both PDFs (a few minutes); later runs reuse the cache.
"""

import json
from pathlib import Path

import pytest

from medrageval.parse.pipeline import parse_pdf
from medrageval.text import contains_quote

pytestmark = pytest.mark.regression
ROOT = Path(__file__).resolve().parents[1]


def _parsed(pdf_name: str, out_name: str) -> dict:
    pdf = ROOT / "data/raw" / pdf_name
    if not pdf.exists():
        pytest.skip(f"{pdf} not present")
    out = ROOT / "data/interim" / out_name
    report = parse_pdf(pdf, out)
    return {
        "report": report,
        "docs": json.loads((out / "documents.json").read_text()),
        "sections": json.loads((out / "sections.json").read_text()),
        "elements": [json.loads(line) for line in (out / "elements.jsonl").open()],
    }


@pytest.fixture(scope="module")
def ta1063():
    return _parsed("source_document.pdf", "ta1063")


@pytest.fixture(scope="module")
def ta1074():
    return _parsed("document2.pdf", "document2")


def _find(parsed, quote):
    return [e for e in parsed["elements"] if contains_quote(quote, e["plain"])]


# --- TA1063 (capivasertib) -------------------------------------------------------------

def test_ta1063_documents(ta1063):
    assert [d["start_page"] for d in ta1063["docs"]] == [1, 4, 150, 175, 333, 344, 356, 367, 381, 399, 411, 418, 542]


def test_ta1063_table_values_and_section(ta1063):
    hits = [e for e in _find(ta1063, "Median PFS (months) 7.3 3.1") if e["page_start"] == 42]
    assert len(hits) == 1 and hits[0]["type"] == "table"
    assert hits[0]["caption"].startswith("**Table 7.")
    assert any(s.startswith("B.2.6.") for s in hits[0]["section_path"])


def test_ta1063_merged_cell_kept(ta1063):
    (table,) = [e for e in ta1063["elements"] if e["page_start"] == 180 and e["type"] == "table"]
    assert '<td rowspan="9">63</td>' in table["text"]


def test_ta1063_ocr_sentence(ta1063):
    (hit,) = _find(ta1063, "When I challenged this, I was sent for psychological counselling")
    assert hit["page_start"] == 384 and hit["source"] == "ocr"


def test_ta1063_redactions(ta1063):
    (figure,) = [e for e in ta1063["elements"] if e["page_start"] == 44 and e["type"] == "figure"]
    assert figure["text"] == "[REDACTED]" and figure["caption"].startswith("Figure 3.")
    assert _find(ta1063, "N= [REDACTED] at 18 months")
    assert not any("****" in e["plain"] or "xxxxxxx" in e["plain"] for e in ta1063["elements"])


def test_ta1063_no_running_footers(ta1063):
    footers = ["Company evidence submission: Capivasertib with fulvestrant", "AstraZeneca UK Ltd (2024). All rights reserved",
               "CONFIDENTIAL UNTIL PUBLISHED", "Page 39 of 146"]
    for footer in footers:
        assert not _find(ta1063, footer), footer


def test_ta1063_citation_not_merged_into_number(ta1063):
    assert _find(ta1063, "Turner et al 2023 [6]")
    assert not _find(ta1063, "Turner et al 20236")


# --- TA1074 (sparsentan) ---------------------------------------------------------------

def test_ta1074_documents(ta1074):
    # p181 (patient summary) has no page numbers or title page: found because p180 is "178 of 178".
    assert [d["start_page"] for d in ta1074["docs"]] == [1, 3, 181, 201, 271, 282, 297, 307, 461, 467, 470, 481, 488]


def test_ta1074_deep_numbering(ta1074):
    by_id = {s["section_id"]: s for s in ta1074["sections"]}
    (leaf,) = [s for s in ta1074["sections"] if s["title"].startswith("B.3.3.1.2.1 ")]
    chain = []
    while leaf:
        chain.append(leaf["title"].split()[0])
        leaf = by_id.get(leaf["parent_id"])
    assert chain[:5] == ["B.3.3.1.2.1", "B.3.3.1.2", "B.3.3.1", "B.3.3", "B.3"]


def test_ta1074_no_running_footers(ta1074):
    for footer in ["BUSINESS USE", "Company evidence submission template for sparsentan_ID6308", "© CSL Vifor (2024)"]:
        assert not _find(ta1074, footer), footer


def test_reports_have_no_unexpected_warnings(ta1063, ta1074):
    assert ta1063["report"]["segmentation_vs_bookmarks"]["recall"] == 1.0
    assert ta1074["report"]["segmentation_vs_bookmarks"]["recall"] == 1.0
    assert all("no headings" not in w for w in ta1063["report"]["warnings"] + ta1074["report"]["warnings"])


def test_parse_report_names_the_file_not_its_path(ta1063):
    # The fixture passes an absolute path; committed outputs must not carry it.
    assert ta1063["report"]["pdf"] == "source_document.pdf"
