from medrag.elements import Element, attach_table_context, compress_markdown_table, merge_continuations


def text(t, page, label="text"):
    return Element("text", t, page, page, ["6 Results"], "EAG report", label=label)


def table(md, p0, p1=None, caption=None):
    return Element("table", md, p0, p1 or p0, ["6 Results"], "EAG report", caption=caption)


MD = "| Technology | Total costs (£) |\n|---|---|\n| Alpelisib | £52,709 |"


def test_compress_markdown_table_removes_padding_and_shortens_separators():
    padded = "| Technology          | Total costs (£)     |\n|---------------------|---------------------|\n| Alpelisib           | £52,709             |"
    assert compress_markdown_table(padded) == MD


def test_caption_above_is_preferred_and_a_caption_below_is_used_when_nothing_is_above():
    els = [text("Table 6.1: Company base case", 530, "caption"), table(MD, 530), text("Some prose.", 530),
           table(MD, 531), text("* footnote one", 531, "footnote"), text("Table 6.2: EAG base case", 531, "caption")]
    out = attach_table_context(els)
    tables = [e for e in out if e.kind == "table"]
    assert [t.caption for t in tables] == ["Table 6.1: Company base case", "Table 6.2: EAG base case"]
    assert tables[1].notes == ["* footnote one"]
    assert [e.text for e in out if e.kind == "text"] == ["Some prose."]       # captions and footnotes were absorbed


def test_a_caption_between_two_tables_belongs_to_the_one_below_it():
    els = [table(MD, 10), text("Table 2. Second table", 10, "caption"), table(MD, 10)]
    tables = [e for e in attach_table_context(els) if e.kind == "table"]
    assert [t.caption for t in tables] == [None, "Table 2. Second table"]


def test_duplicate_caption_element_is_dropped_and_prose_captions_are_recognised():
    els = [table(MD, 534, caption="Table 6.3: Scenarios"), text("Table 6.3: Scenarios", 534, "caption"), text("**Table 7 Costs**", 540), table(MD, 540)]
    out = attach_table_context(els)
    assert [e.text for e in out if e.kind == "text"] == []
    assert [e.caption for e in out if e.kind == "table"] == ["Table 6.3: Scenarios", "Table 7 Costs"]


def test_continuation_tables_merge_when_the_header_repeats_on_the_next_page():
    a = table(MD + "\n| Everolimus | £26,065 |", 530)
    b = table(MD, 531)
    c = table("| Other | Header |\n|---|---|\n| x | y |", 532)
    out = merge_continuations([a, b, c])
    assert len(out) == 2 and out[0].page_end == 531 and out[0].text.count("| Alpelisib |") == 2 and out[0].text.count("| Technology |") == 1
    assert merge_continuations([a, table(MD, 535)]) == [a, table(MD, 535)]    # not on the next page
