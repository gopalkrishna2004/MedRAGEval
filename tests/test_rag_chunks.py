import copy

from synthetic import ELEMENTS, EXTRA_ELEMENTS, _el, build_corpus, config

from medrageval.config import load_config
from medrageval.rag.chunks import build_chunks, split_text

RAG = load_config()["rag"]
CORPUS = build_corpus(config(), elements=ELEMENTS + EXTRA_ELEMENTS)
CHUNKS = build_chunks(CORPUS, RAG)


def _chunk_of(unit_id: str):
    return next(c for c in CHUNKS if unit_id in c.unit_ids)


def test_every_unit_is_in_exactly_one_chunk():
    seen = [u for c in CHUNKS for u in c.unit_ids]
    assert set(seen) == set(CORPUS.units)
    assert len(seen) == len(set(seen))   # nothing in the synthetic bundle is long enough to be split


def test_chunks_never_cross_a_section_or_document():
    for c in CHUNKS:
        units = [CORPUS.units[u] for u in c.unit_ids]
        assert len({u.section_id for u in units}) == 1 and len({u.doc_id for u in units}) == 1


def test_consecutive_paragraphs_of_a_section_are_packed():
    icer = _chunk_of("d02-e0008")
    assert icer.unit_ids == ["d02-e0008", "d02-e0009"]   # the ICER and the caveat after it, same section
    assert icer.page_start == 50 and icer.page_end == 51


def test_a_table_is_a_chunk_of_its_own_with_its_caption():
    t7 = _chunk_of("d02-e0003")
    assert t7.unit_ids == ["d02-e0003"]
    assert "Table 7. PFS by investigator assessment" in t7.text and "Median PFS (months) | 7.3 | 3.1" in t7.text
    form = [c for c in CHUNKS if c.doc_id == "d04"]   # a long form table: one chunk per row group
    assert len(form) > 1 and all(len(c.unit_ids) == 1 and "#r" in c.unit_ids[0] for c in form)


def test_header_names_the_document_section_and_pages():
    head = _chunk_of("d03-e0001").text.split("\n", 1)[0]
    assert head == "EAG report | 5 Cost effectiveness > 5.2 EAG critique of the economic model | p. 90"
    assert _chunk_of("d02-e0008").text.startswith("Company evidence submission | B.3 Cost effectiveness > B.3.10 Base-case results | pp. 50-51")


def test_noise_is_not_chunked():
    assert not any("e0005" in u or "e0010" in u for c in CHUNKS for u in c.unit_ids)   # references, copyright line


def test_long_paragraph_is_split_into_numbered_pieces():
    sentences = [f"Sentence {n} says capivasertib plus fulvestrant improved progression-free survival in cohort {n}." for n in range(60)]
    long = _el("d02", 50, " ".join(sentences), section="d02-s001", path=["B.2 Clinical effectiveness", "B.2.3 Trial design"], page=13)
    cfg = copy.deepcopy(RAG)
    chunks = build_chunks(build_corpus(config(), elements=ELEMENTS + [long]), cfg)
    pieces = [c for c in chunks if "d02-e0050" in c.unit_ids]
    assert len(pieces) > 1 and [c.part for c in pieces] == list(range(len(pieces)))
    assert all(len(c.text.split("\n", 1)[1]) <= cfg["chunking"]["max_chars"] for c in pieces)
    body = " ".join(c.text.split("\n", 1)[1].replace("\n", " ") for c in pieces)
    assert all(s in body for s in sentences)


def test_split_text_cuts_at_lines_then_sentences():
    assert split_text("short", 100) == ["short"]
    assert split_text("a" * 30 + "\n" + "b" * 30, 40) == ["a" * 30, "b" * 30]
    out = split_text("First sentence here. Second sentence here. Third one.", 25)
    assert out == ["First sentence here.", "Second sentence here.", "Third one."]
    assert all(len(p) <= 10 for p in split_text("x" * 35, 10))
