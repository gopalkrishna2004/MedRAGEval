from medrageval.parse.segment import infer_page_numbers, segment

SETTINGS = {
    "use_outline": True,
    "weights": {"page_reset": 2, "title_page": 1, "footer_change": 1, "outline": 1},
    "threshold": 2,
    "min_numbered_pages": 0.5,
}


def _page(n, label=None, title=False, footer=None):
    boxes = [{"class": "text", "text": f"body of page {n}"}]
    if title:
        boxes.insert(0, {"class": "title", "text": "###### **Report title**"})
    removed = [{"text": footer}] if footer else []
    return {"page": n, "page_label": label, "boxes": boxes, "removed": removed}


def _starts(pages, outline=(), **overrides):
    return [d["start_page"] for d in segment(pages, list(outline), {**SETTINGS, **overrides})]


def test_unnumbered_cover_before_page_two_is_page_one():
    pages = [_page(1, "1 of 2"), _page(2, "2 of 2"), _page(3), _page(4, "2"), _page(5, "3")]
    assert infer_page_numbers(pages)[2] == (1, None)
    assert _starts(pages) == [1, 3]


def test_reset_by_changed_total():
    pages = [_page(1, "1 of 2"), _page(2, "2 of 2"), _page(3, "2 of 5"), _page(4, "3 of 5")]
    assert _starts(pages) == [1, 3]


def test_page_after_last_numbered_page_starts_document():
    # The next document has no page numbers or title page at all.
    pages = [_page(1, "1 of 2"), _page(2, "2 of 2"), _page(3), _page(4)]
    assert _starts(pages) == [1, 3]


def test_weak_signal_alone_does_not_split():
    pages = [_page(i, f"{i} of 4") for i in range(1, 5)]
    pages[2] = _page(3, "3 of 4", title=True)
    assert _starts(pages) == [1]
    assert _starts(pages, outline=[{"level": 1, "page": 3, "title": "x"}]) == [1, 3]


def test_title_page_decides_when_pages_are_unnumbered():
    pages = [_page(1, title=True), _page(2), _page(3, title=True), _page(4)]
    assert _starts(pages) == [1, 3]


def test_footer_change_with_title_splits_without_page_reset():
    # Continuous numbering, so only the two weak signals together can split.
    pages = [_page(1, "1", footer="Company submission"), _page(2, "2", footer="Company submission"),
             _page(3, "3", title=True, footer="Assessment report"), _page(4, "4", footer="Assessment report")]
    assert _starts(pages) == [1, 3]
    pages[2] = _page(3, "3", footer="Assessment report")  # footer change alone
    assert _starts(pages) == [1]
