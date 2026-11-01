import pytest
from synthetic import ELEMENTS, EXTRA_ELEMENTS, TABLE7, _el, build_corpus, config

from medrageval.generate.corpus import BM25, corpus_topic, detect_roles, extract_entities, is_noise, make_units, mentions, significant_numbers, template_elements, tokenize
from medrageval.text import contains_quote

CFG = config()


@pytest.mark.parametrize(
    "element_id, reason",
    [
        ("d02-e0005", "section:^references\\b"),   # reference list
        ("d02-e0006", "short"),                    # page label fragment
        ("d02-e0010", "text:committed to meeting the requirements of copyright"),
        ("d01-e0002", "text:\\.{6,}"),             # contents line with dot leaders
        ("d02-e0001", None),
    ],
)
def test_noise_reasons(element_id, reason):
    el = next(e for e in ELEMENTS if e["element_id"] == element_id)
    assert is_noise(el, CFG) == reason


def test_figures_are_noise():
    el = dict(next(e for e in ELEMENTS if e["element_id"] == "d02-e0001"), type="figure")
    assert is_noise(el, CFG) == "figure"
    chart = next(e for e in EXTRA_ELEMENTS if e["element_id"] == "d02-e0026")   # an OCR'd forest plot stored as a table
    assert chart["type"] == "table" and is_noise(chart, CFG) == "figure"


def test_units_carry_source():
    ocr = next(e for e in EXTRA_ELEMENTS if e["element_id"] == "d02-e0025")
    assert make_units(ocr, CFG)[0].source == "ocr"
    assert make_units(next(e for e in ELEMENTS if e["element_id"] == "d02-e0001"), CFG)[0].source == "native"


def test_twin_tables_and_their_discriminating_caption_words():
    corpus = build_corpus(CFG, ELEMENTS + EXTRA_ELEMENTS)
    assert corpus.twins["d02-e0020"] == ["d02-e0021"] and corpus.twins["d02-e0021"] == ["d02-e0020"]
    assert corpus.discriminator(corpus.units["d02-e0021"]) == ["fulvestrant arm"]   # "fulvestrant" alone is in both captions
    assert corpus.discriminator(corpus.units["d02-e0020"]) == ["capivasertib arm"]
    assert corpus.discriminator(corpus.units["d02-e0022"]) == [""]                  # identical captions: nothing tells them apart
    assert corpus.discriminator(corpus.units["d02-e0003"]) is None                   # no twin
    assert "d02-e0024" not in corpus.twins and "d02-e0027" not in corpus.twins


def test_twins_across_documents_and_copies():
    from synthetic import DOCUMENTS, MORE_DOCUMENTS, MORE_ELEMENTS
    corpus = build_corpus(CFG, ELEMENTS + MORE_ELEMENTS, DOCUMENTS + MORE_DOCUMENTS)
    assert corpus.twins["d02-e0003"] == ["d05-e0020"] and corpus.copies["d02-e0003"] == ["d03-e0020"]   # same cells in the EAG report
    assert corpus.twins["d03-e0020"] == ["d05-e0020"] and corpus.copies["d03-e0020"] == ["d02-e0003"]
    assert corpus.discriminator(corpus.units["d02-e0003"]) == ["company submission"]                       # captions identical: the document tells
    assert corpus.discriminator(corpus.units["d05-e0020"]) == ["clarification"]
    assert corpus.discriminator(corpus.units["d03-e0020"]) == ["EAG"]


def test_long_table_splits_into_row_units_with_header():
    form = next(e for e in ELEMENTS if e["element_id"] == "d04-e0001")
    assert len(form["plain"]) > CFG["limits"]["max_table_chars"]
    units = make_units(form, CFG)
    assert len(units) > 3 and all(u.unit_id.startswith("d04-e0001#r") for u in units)
    assert all(u.element_id == "d04-e0001" and u.row_index is not None for u in units)
    header = "1. Your name | Charlotte Moss"
    assert all(u.plain.startswith(header) for u in units)
    # every row line of a row unit is a line of the parent element, so a quote from it is a quote from the element
    for u in units:
        for line in u.plain.splitlines()[1:]:
            assert contains_quote(line, form["plain"])
    assert all(len(u.plain) <= CFG["limits"]["row_group_chars"] + 400 for u in units)


def test_short_table_stays_whole():
    table = next(e for e in ELEMENTS if e["element_id"] == "d02-e0003")
    (u,) = make_units(table, CFG)
    assert u.unit_id == "d02-e0003" and u.row_index is None and u.evidence


def test_roles_from_headings_and_body():
    c = build_corpus()
    assert {d: c.docs[d].role for d in c.docs} == {
        "d01": "cover", "d02": "company_submission", "d03": "eag_report", "d04": "clinical_expert", "d05": "clarification"}
    assert c.docs["d03"].label == "EAG report"


def test_role_override_and_other():
    cfg = config()
    cfg["roles"]["override"] = {"d03": "guidance"}
    c = build_corpus(cfg)
    assert c.docs["d03"].role == "guidance"
    docs = [{"doc_id": "d09", "headings": ["Something else entirely"] * 3, "running_text": None}]
    els = [dict(ELEMENTS[2], doc_id="d09", element_id=f"d09-e{i:04d}") for i in range(40)]
    assert detect_roles(docs, els, CFG)["d09"][0] == "other"


def test_topic_is_the_common_prefix_of_the_most_shared_title():
    titles = [
        "Capivasertib with fulvestrant for treating advanced breast cancer [ID6370] Patient expert statement",
        "Capivasertib with fulvestrant for treating advanced breast cancer [ID6370] Clinical expert statement",
        "Patient Organisation Submission", "Patient Organisation Submission",
        "NATIONAL INSTITUTE FOR HEALTH AND CARE EXCELLENCE",
    ]
    assert corpus_topic(titles) == "Capivasertib with fulvestrant for treating advanced breast cancer [ID6370]"


def test_cross_reference_resolves_within_document_only():
    c = build_corpus()
    assert c.resolve_references(c.units["d02-e0002"]) == [("table", "7", "d02-e0003")]
    assert c.resolve_references(c.units["d03-e0004"]) == []   # "CS Table 7" is in d02, not d03


def test_entities_learned_from_trial_context():
    c = build_corpus()
    assert c.entity_names == {"CAPITELLO291": "CAPItello-291"}
    assert c.units["d02-e0001"].unit_id in c.entities["CAPITELLO291"]
    assert extract_entities("the phase 3 trial and the PROTECT study and the ITT study", CFG) == {"PROTECT": "PROTECT"}
    assert extract_entities("SOLAR-1 and BOLERO-2 data; HER2-negative; EQ-5D; SF-36; COVID-19", CFG) == {"SOLAR1": "SOLAR-1", "BOLERO2": "BOLERO-2"}
    assert mentions("results of CAPItello291 were", "CAPITELLO291")


def test_significant_numbers():
    assert significant_numbers("£24,567 per QALY in 2023 with 0.61 QALYs and 50% of 155 patients, n=12") == {"24567", "0.61", "50", "155"}
    assert significant_numbers("Around 40-50% of people (aged 34-90) pay £1,000–2,000") == {"40-50", "1000-2000"}   # a range is one value


def test_bm25_ranks_rare_number_first_and_respects_filters():
    c = build_corpus()
    hits = c.bm25.search(tokenize("£24,567"), k=3)         # "24,567" tokenises as "24", "567"
    assert {h for h, _ in hits} == {"d02-e0008", "d03-e0001", "d03-e0003"}
    only = c.bm25.search(["567"], k=3, only={"d03-e0001"})
    assert [h for h, _ in only] == ["d03-e0001"]
    assert "d03-e0001" not in [h for h, _ in c.bm25.search(["567"], k=5, exclude={"d03-e0001"})]
    assert BM25({"a": ["x"], "b": ["y"]}).search(["z"]) == []


def test_render_has_context_header_and_flat_section_titles():
    c = build_corpus()
    text = c.render(c.units["d02-e0002"])
    head, body = text.split("\n", 1)
    assert head.startswith("[d02-e0002] Document: Company evidence submission (d02) | Section: B.2 Clinical effectiveness > B.2.6")
    assert "Pages: 20-20" in head and body.startswith("Progression-free survival")


def test_form_text_repeated_across_documents_of_one_role_is_noise():
    note = "You do not have to answer every question – they are prompts to guide you. The text boxes will expand as you type."
    els = [_el("d05", 1, note), _el("d06", 1, note), _el("d07", 1, "Our members told us that side effects matter most."), _el("d02", 1, note)]
    roles = {"d05": "patient_org", "d06": "patient_org", "d07": "patient_org", "d02": "company_submission"}
    assert template_elements(els, roles, CFG) == {"d05-e0001", "d06-e0001"}   # one company copy is not a form template


def test_reprint_with_shortened_labels_is_a_copy():
    reprint = _el("d03", 30, TABLE7.replace("Hazard ratio (95% CI)", "HR (95% CI)").replace("Events, n (%)", "Events n (%)"), "table",
                  section="d03-s003", path=["3 Clinical effectiveness"], caption="**Table 3.8: PFS by investigator assessment**", page=67)
    corpus = build_corpus(CFG, ELEMENTS + [reprint])
    assert corpus.copies["d02-e0003"] == ["d03-e0030"] and corpus.copies["d03-e0030"] == ["d02-e0003"]
    assert "d02-e0003" not in corpus.twins
