from medrageval.parse.structure import structure_document

DOC = {"doc_id": "d01"}


def _h(text, size=12.0, bold=True, y=100):
    return {"class": "section-header", "text": f"###### **{text}**", "bbox": [70, y, 520, y + 14],
            "font_size": size, "bold": bold, "source": "native", "redactions": 0}


def _p(text, cls="text", y=200):
    return {"class": cls, "text": text, "bbox": [70, y, 520, y + 40],
            "font_size": 11.0, "bold": False, "source": "native", "redactions": 0}


def _tree(boxes_by_page):
    # Space boxes out down the page unless a test sets positions itself.
    for boxes in boxes_by_page:
        if len({b["bbox"][1] for b in boxes}) == 1 and len(boxes) > 1:
            for i, b in enumerate(boxes):
                b["bbox"] = [70, 80 + 60 * i, 520, 94 + 60 * i]
    pages = [{"page": i, "boxes": boxes} for i, boxes in enumerate(boxes_by_page, start=1)]
    sections, elements = structure_document(DOC, pages)
    titles = {s["section_id"]: s["title"] for s in sections}
    return {s["title"]: titles.get(s["parent_id"]) for s in sections}, elements


def test_numbered_headings_nest_by_prefix_not_font():
    parents, _ = _tree([[_h("B.2. Clinical", 16), _h("B.2.6 Results", 12), _h("B.2.6.1 PFS", 12),
                         _h("B.2.7 Subgroups", 12), _h("B.3. Cost", 16)]])
    assert parents == {"B.2. Clinical": None, "B.2.6 Results": "B.2. Clinical", "B.2.6.1 PFS": "B.2.6 Results",
                       "B.2.7 Subgroups": "B.2. Clinical", "B.3. Cost": None}


def test_named_levels_contain_numbered_and_are_siblings():
    parents, _ = _tree([[_h("SECTION 2: Landscape"), _h("2a) The condition"), _h("2b) Diagnosis"),
                         _h("SECTION 3: The treatment"), _h("3a) How it works")]])
    assert parents["2b) Diagnosis"] == "SECTION 2: Landscape"
    assert parents["SECTION 3: The treatment"] is None
    assert parents["3a) How it works"] == "SECTION 3: The treatment"


def test_unnumbered_headings_nest_by_typography():
    parents, _ = _tree([[_h("Living with the condition", 14), _h("Symptoms", 12), _h("Diagnosis", 12),
                         _h("Treatment", 14)]])
    assert parents == {"Living with the condition": None, "Symptoms": "Living with the condition",
                       "Diagnosis": "Living with the condition", "Treatment": None}


def test_split_heading_joined_and_caption_not_a_section():
    parents, elements = _tree([[
        _h("Part 1: Treating breast cancer after", y=100), _h("endocrine treatment", y=115),
        _h("Table 3. Baseline characteristics", y=150), _p("<table><tr><th>a</th></tr></table>", "table", y=170),
    ]])
    assert list(parents) == ["Part 1: Treating breast cancer after endocrine treatment"]
    assert elements[0]["type"] == "table" and elements[0]["caption"] == "Table 3. Baseline characteristics"


def test_short_heading_followed_by_another_heading_not_joined():
    short = _h("5.5.2 Exploratory analysis results", y=100)
    short["bbox"] = [70, 100, 300, 114]
    parents, _ = _tree([[short, _h("Results of the EAG's preferred analysis", y=118), _p("Body.", y=150)]])
    assert list(parents) == ["5.5.2 Exploratory analysis results", "Results of the EAG's preferred analysis"]


def test_table_continuing_on_next_page_is_merged():
    head = "<tr><th>Arm</th><th>N</th></tr>"
    _, elements = _tree([
        [_h("B.1 Results"), _p(f"<table>{head}<tr><td>A</td><td>1</td></tr></table>", "table", y=600)],
        [_p(f"<table>{head}<tr><td>B</td><td>2</td></tr></table>", "table", y=80), _p("After the table.")],
    ])
    table = elements[0]
    assert (table["page_start"], table["page_end"]) == (1, 2)
    assert table["text"].count(head) == 1 and "<td>B</td>" in table["text"]
    assert elements[1]["section_path"] == ["B.1 Results"]
