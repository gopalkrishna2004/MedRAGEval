import pytest

from medrageval.parse.clean import REDACTED, clean_pages, clean_text

HEADER_FOOTER = {"margin": 0.25, "min_pages": 3, "max_gap": 3, "min_in_margin": 0.8, "fuzzy_ratio": 0.85}


@pytest.mark.parametrize(
    "raw, expected",
    [
        # reference numbers must not merge into a preceding number
        ("Turner et al 2023<sup>6</sup>", "Turner et al 2023[6]"),
        ("CAPItello-291<sup>6,57</sup>", "CAPItello-291[6,57]"),
        ("study report<sup>_61_</sup>", "study report[61]"),
        ("17<sup>th</sup> July", "17th July"),
        ("Median PFS (months)<sup>†</sup>", "Median PFS (months)†"),
        ("CAELYX<sup>TM</sup>", "CAELYX TM"),
        ("© <mark>AstraZeneca UK Ltd</mark> <u>2024</u>", "© AstraZeneca UK Ltd 2024"),
    ],
)
def test_inline_markup(raw, expected):
    assert clean_text(raw) == expected


@pytest.mark.parametrize(
    "raw, expected",
    [
        # word-by-word redacted passage collapses to one marker
        ("Furthermore,<br/>**** *** *** ****** *****<br/>**** ** * ***", f"Furthermore,<br/>{REDACTED}"),
        # value with confidence interval keeps its shape
        ("HR **** (****, *****)", f"HR {REDACTED} ({REDACTED}, {REDACTED})"),
        ("<td>*******</td>", f"<td>{REDACTED}</td>"),
        ("£X,XXX.XX per pack", f"£{REDACTED} per pack"),
        ("discontinued in <mark>xxx</mark> patients", f"discontinued in {REDACTED} patients"),
        ("Lower bound  xxxxxxx xxxxxxx", f"Lower bound  {REDACTED}"),
        ("plus exemestan exxxxxxxxx) . Total", f"plus exemestan e {REDACTED}) . Total"),
        # inside Markdown italics
        ("[95% CI:_ _**********] versus_ _*** months", f"[95% CI:_ _{REDACTED}] versus_ _{REDACTED} months"),
    ],
)
def test_redactions(raw, expected):
    assert clean_text(raw) == expected


@pytest.mark.parametrize(
    "text",
    [
        "occurring in >5% in pivotal trials***",
        "base-case results (Scenario 1)***",
        "not publicly available; ***Cheapest cost per mg",
        "**Table 7. PFS by investigator assessment**",
        "30 x 2.5 mg tablets",
        "a complex reflex",
        "*Does not include RECIST progression events",
    ],
)
def test_not_redactions(text):
    assert clean_text(text) == text


def test_placeholder_under_black_box_is_redacted():
    assert clean_text("N= xx at 18 months, N= xx at 20 months", ["xx", "xx"]) == (
        f"N= {REDACTED} at 18 months, N= {REDACTED} at 20 months"
    )


def _box(text, y0, y1, cls="text", **extra):
    return {"class": cls, "bbox": [70, y0, 520, y1], "text": text, "source": "native",
            "redaction_boxes": 0, "redacted_tokens": [], **extra}


def _pages(n=5):
    pages = []
    for i in range(1, n + 1):
        pages.append({
            "page": i, "width": 595, "height": 842,
            "boxes": [
                _box("## NICE National Institute for Health and Care Excellence", 40, 70),
                _box(f"Body paragraph {i} about progression-free survival.", 100, 300),
                # repeated on every page, but in the body: must survive
                _box("**Source:** Data on file", 400, 410),
                _box(f"Company evidence submission [ID6370] Page {i} of {n}", 760, 780),
            ],
        })
    return pages


def test_running_headers_and_footers_removed_body_notes_kept():
    cleaned = clean_pages(_pages(), {"header_footer": HEADER_FOOTER})
    for p in cleaned:
        texts = [b["text"] for b in p["boxes"]]
        assert texts == [f"Body paragraph {p['page']} about progression-free survival.", "**Source:** Data on file"]
        assert p["page_label"] == f"{p['page']} of 5"
        assert len(p["removed"]) == 2


def test_template_heading_repeated_once_per_document_kept():
    # Same form heading at the top of a page in three separate documents.
    pages = _pages(12)
    for i in (0, 4, 8):
        pages[i]["boxes"].insert(1, _box("###### **About you**", 120, 135))
    cleaned = clean_pages(pages, {"header_footer": HEADER_FOOTER})
    assert [i for i, p in enumerate(cleaned) if "###### **About you**" in [b["text"] for b in p["boxes"]]] == [0, 4, 8]


def test_footer_glued_into_last_paragraph_removed():
    pages = _pages()
    pages[2]["boxes"][1] = _box("Treatments aiming to slow decline. Company evidence submission [ID6370] Page 3 of 5",
                                700, 790)
    pages[3]["boxes"][1] = _box("Company evidence submission [ID6370] is discussed in the body.", 300, 400)
    # a title at the top of a page that reads like the footer stays
    pages[4]["boxes"].insert(1, _box("Overview. Company evidence submission [ID6370]", 80, 100, cls="title"))
    cleaned = clean_pages(pages, {"header_footer": HEADER_FOOTER})
    assert cleaned[2]["boxes"][0]["text"] == "Treatments aiming to slow decline."
    assert cleaned[2]["page_label"] == "3 of 5"
    # same words in a body box away from the margin are left alone
    assert cleaned[3]["boxes"][0]["text"] == "Company evidence submission [ID6370] is discussed in the body."
    assert cleaned[4]["boxes"][0]["text"] == "Overview. Company evidence submission [ID6370]"


def test_footer_split_into_table_rows_removed():
    pages = _pages()
    # Rows of numbers, redactions or a single footer word at the edge are data and stay.
    rows = ("<tr><td>Table 42. Key variables</td><td>112</td></tr>"
            "<tr><td>Total</td><td>155</td></tr>"
            "<tr><td>3</td><td>-0.5 (-1.2, 0.3)</td></tr><tr><td>[REDACTED]</td></tr>"
            "<tr><td>Company evidence</td></tr><tr><td>submission [ID6370]</td></tr><tr><td>Page 3 of 5</td></tr>")
    pages[2]["boxes"][1] = _box(f"<table>{rows}</table>", 500, 790, cls="table")
    one_word = "<table><tr><td>Arm</td></tr><tr><td>Company</td></tr></table>"
    pages[3]["boxes"][1] = _box(one_word, 500, 790, cls="table")
    cleaned = clean_pages(pages, {"header_footer": HEADER_FOOTER})
    assert cleaned[2]["boxes"][0]["text"] == (
        "<table><tr><td>Table 42. Key variables</td><td>112</td></tr><tr><td>Total</td><td>155</td></tr>"
        "<tr><td>3</td><td>-0.5 (-1.2, 0.3)</td></tr><tr><td>[REDACTED]</td></tr></table>")
    assert cleaned[3]["boxes"][0]["text"] == one_word


def test_ocr_variant_of_known_footer_removed():
    pages = _pages()
    pages[4]["boxes"][3]["text"] = "Cornpany evidence subrnission [ID6370] Page 5 of 5"
    cleaned = clean_pages(pages, {"header_footer": HEADER_FOOTER})
    assert not any("subrnission" in b["text"] for b in cleaned[4]["boxes"])


def test_title_resembling_footer_far_from_its_run_kept():
    pages = _pages(12)
    for p in pages[6:]:
        p["boxes"][3]["text"] = f"Capivasertib with fulvestrant for treating breast cancer [ID6370] {p['page']} of 12"
    pages[0]["boxes"].insert(1, _box("**Capivasertib with fulvestrant for treating breast cancer [ID6370]**", 140, 160))
    cleaned = clean_pages(pages, {"header_footer": HEADER_FOOTER})
    assert "**Capivasertib with fulvestrant for treating breast cancer [ID6370]**" in [b["text"] for b in cleaned[0]["boxes"]]


def test_blacked_out_figure_becomes_redacted_box():
    pages = _pages()
    pages[0]["boxes"].append(_box("\n\n", 450, 700, cls="picture", redaction_boxes=1))
    cleaned = clean_pages(pages, {"header_footer": HEADER_FOOTER})
    figure = cleaned[0]["boxes"][-1]
    assert (figure["class"], figure["text"], figure["redactions"]) == ("picture", REDACTED, 1)


@pytest.mark.parametrize("raw, expected", [
    ("were sourced from the latest NHS reference costs (202122).", "were sourced from the latest NHS reference costs (2021–22)."),
    ("Prices from 201920 and 202324 tariffs", "Prices from 2019–20 and 2023–24 tariffs"),
    ("Currency code 202199 and 2021220 patients", "Currency code 202199 and 2021220 patients"),   # not the next year; seven digits
    ("a total of 202,122 and 2021.22", "a total of 202,122 and 2021.22"),
])
def test_year_range_missing_its_dash(raw, expected):
    assert clean_text(raw) == expected
