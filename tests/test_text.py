import pytest

from medrageval.text import contains_quote, normalize, to_plain

TABLE = (
    "<table><tr><th>Progression or death</th><th>Capivasertib<br/>(N=155)</th></tr>"
    "<tr><td>Median PFS (months)†</td><td>7.3</td></tr>"
    "<tr><td>Grade ≥3 AEs</td><td>&lt;1</td></tr></table>"
)


def test_table_to_plain_rows():
    assert to_plain(TABLE) == (
        "Progression or death | Capivasertib (N=155)\nMedian PFS (months)† | 7.3\nGrade ≥3 AEs | <1"
    )


def test_markdown_and_tags_removed_but_template_text_kept():
    raw = "###### **Table 7. PFS**\n\n- _Source:_ CSR[6,57] on **<insert deadline>.**<br/>Next"
    assert to_plain(raw) == "Table 7. PFS\n\nSource: CSR[6,57] on <insert deadline>.\nNext"


@pytest.mark.parametrize(
    "quote",
    [
        "HR 0.50; 95% CI 0.38–0.65",              # dash style and spacing differ
        "hazard ratio 0.50 (95% CI: 0.38, 0.65)",  # punctuation differs
        "Median PFS (months) 7.3",                # table row without the cell separator
        "“capivasertib plus fulvestrant”",        # curly quotes
    ],
)
def test_quotes_match_despite_formatting(quote):
    source = (
        "a 50% reduction (hazard ratio 0.50; 95% CI 0.38-0.65) with 'capivasertib plus fulvestrant'[6] "
        "HR 0.50; 95%CI 0.38, 0.65 " + TABLE
    )
    assert contains_quote(quote, source)


@pytest.mark.parametrize("quote", ["Median PFS (months) 73", "HR 0.5 95% CI", "progression free survival 7.3"])
def test_quotes_with_different_content_do_not_match(quote):
    assert not contains_quote(quote, "HR 0.50; 95% CI 0.38-0.65. " + TABLE)


def test_normalize_keeps_decimals_and_symbols():
    assert normalize("Grade ≥3: 7.3% (n<5)") == "grade ≥ 3 7.3 % n < 5"
