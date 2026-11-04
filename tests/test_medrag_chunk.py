from medrag.chunk import Chunk, TokenCounter, build_chunks, header, split_long, split_sentences, table_lines
from medrag.elements import Element

CFG = {"target_tokens": 40, "max_tokens": 60, "overlap_sentences": 1, "max_table_tokens": 60, "min_body_chars": 15}
COUNT = TokenCounter(None)     # ~4 chars per token


def para(text, page=5, path=("B.2 Clinical effectiveness",), title="Company submission"):
    return Element("text", text, page, page, list(path), title)


def test_split_sentences_and_split_long():
    assert split_sentences("One sentence. Another one! A third? (Fourth) 5th.") == ["One sentence.", "Another one!", "A third?", "(Fourth) 5th.", ]
    long = " ".join(f"Sentence number {i} is here." for i in range(12))
    pieces = split_long(long, 20, COUNT.count)
    assert len(pieces) > 1 and all(COUNT.count(p) <= 20 for p in pieces) and " ".join(pieces) == long
    one = "word " * 200
    assert all(COUNT.count(p) <= 20 for p in split_long(one.strip(), 20, COUNT.count))


def test_prose_is_packed_per_section_with_an_overlap_sentence():
    els = [Element("heading", "B.2 Clinical effectiveness", 5, 5, ["B.2 Clinical effectiveness"], "Company submission", level=1)]
    els += [para(f"Paragraph {i} states a fact about the trial. It has two sentences.") for i in range(6)]
    els += [Element("heading", "B.3 Costs", 6, 6, ["B.3 Costs"], "Company submission", level=1), para("Costs are described here in one line.", 6, ("B.3 Costs",))]
    chunks = build_chunks(els, CFG, COUNT)
    assert all(c.tokens <= CFG["max_tokens"] + 12 for c in chunks)          # header adds a little
    assert chunks[0].text.startswith("Company submission | B.2 Clinical effectiveness | p. 5\n")
    assert chunks[0].body.startswith("Paragraph 0") and "Paragraph 1" in chunks[0].body
    assert chunks[1].body.startswith("It has two sentences.")                # the last sentence of the previous chunk opens the next
    assert chunks[-1].heading_path == ["B.3 Costs"] and chunks[-1].page_start == 6
    assert all(len({c.heading_path[0]}) == 1 for c in chunks) and [c.chunk_id for c in chunks] == [f"c{i:05d}" for i in range(1, len(chunks) + 1)]


def test_a_table_stays_whole_with_caption_and_notes_or_splits_by_rows_with_the_header_repeated():
    md = "| Arm | Median PFS | HR |\n|---|---|---|\n| Capivasertib | 7.3 | 0.60 |\n| Placebo | 3.1 | - |"
    small = Element("table", md, 21, 21, ["B.2.6 Results"], "Company submission", caption="Table 12. PFS results", notes=["* investigator assessed"])
    [c] = build_chunks([small], CFG, COUNT)
    assert c.kind == "table" and c.body == f"Table 12. PFS results\n{md}\n* investigator assessed" and c.part is None and c.metadata == {"caption": "Table 12. PFS results"}
    rows = "\n".join(f"| Scenario {i} | £{20000 + i * 1000:,} | {1 + i / 10:.2f} |" for i in range(12))
    big = Element("table", f"| Scenario | ICER | QALYs |\n|---|---|---|\n{rows}", 30, 31, ["B.3.10 Scenarios"], "Company submission", caption="Table 50. Scenarios")
    parts = build_chunks([big], {**CFG, "max_table_tokens": 70}, COUNT)
    assert len(parts) > 1 and [p.part for p in parts] == list(range(len(parts)))
    assert all(p.body.startswith("Table 50. Scenarios\n| Scenario | ICER | QALYs |\n|---|---|---|\n") for p in parts)
    assert "".join(p.body for p in parts).count("| Scenario 11 |") == 1 and parts[0].metadata["parts"] == len(parts)
    assert all(p.page_start == 30 and p.page_end == 31 for p in parts)


def test_table_lines_and_header():
    head, rows = table_lines("| a | b |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |")
    assert head == ["| a | b |", "|---|---|"] and rows == ["| 1 | 2 |", "| 3 | 4 |"]
    assert header("EAG report", ["6", "6.3 EAG's preferred assumptions", "6.3.1", "Detail"], 537, 537) == "EAG report | 6.3 EAG's preferred assumptions > 6.3.1 > Detail | p. 537"
    assert header(None, [], 1, 3) == "Document | - | pp. 1-3"


def test_tiny_fragments_are_dropped_and_json_round_trips():
    chunks = build_chunks([para("12"), para("A real paragraph with enough text to keep.")], CFG, COUNT)
    assert len(chunks) == 1 and chunks[0].body == "A real paragraph with enough text to keep."
    assert Chunk.from_json(chunks[0].to_json()) == chunks[0]
