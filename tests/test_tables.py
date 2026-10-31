"""Table repairs on HTML as pymupdf4llm writes it, with rows taken from TA1063."""

from medrageval.parse.tables import repair_table
from medrageval.text import table_rows, to_plain


def rows(html):
    return table_rows(repair_table(html))


def test_header_split_over_two_rows_becomes_one_and_the_spacer_column_goes():
    html = ('<table><tr><th>Model</th><th colspan="3">PI3K/AKT pathway-altered, post-CDK4/6i</th></tr>'
            '<tr><th></th><th>AIC</th><th></th><th>BIC</th></tr>'
            '<tr><td>Exponential</td><td>324.2</td><td></td><td>326.9</td></tr>'
            '<tr><td>Log-logistic</td><td>322.8</td><td></td><td>328.3</td></tr></table>')
    assert rows(html) == [["Model", "PI3K/AKT pathway-altered, post-CDK4/6i AIC", "PI3K/AKT pathway-altered, post-CDK4/6i BIC"],
                          ["Exponential", "324.2", "326.9"], ["Log-logistic", "322.8", "328.3"]]


def test_group_header_over_a_td_row_of_column_names():
    html = ('<table><tr><th></th><th colspan="2">Evidence of non-PH<br/>(strong = 3, weak = 1)</th></tr>'
            '<tr><td>Trial</td><td>PFS</td><td>OS</td></tr>'
            '<tr><td>BOLERO-2</td><td>Weak</td><td>Moderate</td></tr>'
            '<tr><td>SOLAR-1</td><td>Moderate</td><td>Weak</td></tr>'
            '<tr><td>PFS = progression-free survival; PH =</td><td>proportional hazards; OS = overall</td><td>survival</td></tr></table>')
    out = rows(html)
    assert out[0] == ["Trial", "Evidence of non-PH (strong = 3, weak = 1) PFS", "Evidence of non-PH (strong = 3, weak = 1) OS"]
    assert out[1] == ["BOLERO-2", "Weak", "Moderate"]                      # text values are not swallowed into the header
    assert out[-1] == ["PFS = progression-free survival; PH = proportional hazards; OS = overall survival", "", ""]   # one note cell


def test_label_wrapped_onto_its_own_line_is_joined():
    html = ('<table><tr><th></th><th>CAPItello-291</th><th>SOLAR-1</th></tr>'
            '<tr><td>Patients starting</td><td>[REDACTED]</td><td>116 (68.6%)</td></tr>'
            '<tr><td>subsequent medication</td><td></td><td></td></tr>'
            '<tr><td colspan="3">Of those that received subsequent therapy:</td></tr>'
            '<tr><td>Chemotherapy</td><td>[REDACTED]</td><td>38 + 20 (50.0%)*</td></tr></table>')
    out = rows(html)
    assert out[1] == ["Patients starting subsequent medication", "[REDACTED]", "116 (68.6%)"]
    assert out[2] == ["Of those that received subsequent therapy:", "", ""] and out[3][0] == "Chemotherapy"


def test_label_wrapped_across_value_rows_spans_them_and_blank_labels_are_covered():
    html = ('<table><tr><th></th><th>Endpoint</th><th>Observed</th><th>Modelled</th></tr>'
            '<tr><th>Capivasertib plus fulvestrant<br/>(PI3K/AKT+prior CDK4/6i</th><th>OS</th><th>[REDACTED]</th><th>[REDACTED]</th></tr>'
            '<tr><th>pop from CAPItello-291)</th><th>PFS</th><th>[REDACTED]</th><th>[REDACTED]</th></tr>'
            '<tr><td>Everolimus plus exemestane<br/>(BOLERO-2)</td><td>OS</td><td>19.8</td><td>16.2</td></tr>'
            '<tr><td></td><td>PFS</td><td>9.5</td><td>6.3</td></tr></table>')
    fixed = repair_table(html)
    assert 'rowspan="2"' in fixed and fixed.count("<tr>") == 5
    out = rows(html)
    assert out[1][0] == out[2][0] == "Capivasertib plus fulvestrant (PI3K/AKT+prior CDK4/6i pop from CAPItello-291)"
    assert out[3] == ["Everolimus plus exemestane (BOLERO-2)", "OS", "19.8", "16.2"]
    assert out[4] == ["Everolimus plus exemestane (BOLERO-2)", "PFS", "9.5", "6.3"]
    assert "Everolimus plus exemestane (BOLERO-2) | PFS | 9.5 | 6.3" in to_plain(fixed)


def test_repeated_header_row_and_existing_spans_survive():
    html = ('<table><tr><th>Study</th><th>Arm</th><th>N</th></tr>'
            '<tr><td rowspan="2">FAKTION</td><td>Capivasertib</td><td>69</td></tr><tr><td>Fulvestrant</td><td>71</td></tr>'
            '<tr><td>Study</td><td>Arm</td><td>N</td></tr>'
            '<tr><td>CONFIRM</td><td>Fulvestrant</td><td>362</td></tr></table>')
    assert rows(html) == [["Study", "Arm", "N"], ["FAKTION", "Capivasertib", "69"], ["FAKTION", "Fulvestrant", "71"], ["CONFIRM", "Fulvestrant", "362"]]
    assert 'rowspan="2"' in repair_table(html)


def test_ordinary_tables_are_unchanged():
    html = ('<table><tr><th>Progression or death</th><th>Capivasertib + fulvestrant (N=155)</th><th>Placebo + fulvestrant (N=134)</th></tr>'
            '<tr><td>Median PFS (months)</td><td>7.3</td><td>3.1</td></tr><tr><td>Hazard ratio (95% CI)</td><td>0.50 (0.38, 0.65)</td><td></td></tr></table>')
    assert repair_table(html) == html
    assert repair_table("<table><tr><td>one cell</td></tr></table>") == "<table><tr><td>one cell</td></tr></table>"


def test_lowercase_labels_with_complete_values_stay_separate_rows():
    html = ('<table><tr><th>Model terms</th><th>Converged?</th><th>AIC</th><th>BIC</th></tr>'
            '<tr><td>trt (model 1)</td><td>Yes</td><td>-5874.3</td><td>-5662.0</td></tr>'
            '<tr><td>PFSFLAG (model 2)</td><td>Yes</td><td>-5903.7</td><td>-5691.5</td></tr>'
            '<tr><td>trt + PFSFLAG (model 3)</td><td>Yes</td><td>-5897.4</td><td>-5678.6</td></tr></table>')
    assert repair_table(html) == html


def test_header_wrapped_over_three_rows():
    html = ('<table><tr><th>HR applied to PFS for TTD for:</th><th></th><th>ICER* (capivasertib plus</th><th>% change from</th></tr>'
            '<tr><th>Capivasertib plus</th><th>Alpelisib plus</th><th>fulvestrant vs. alpelisib</th><th>base case</th></tr>'
            '<tr><td>fulvestrant arm</td><td>fulvestrant</td><td>plus fulvestrant)</td><td></td></tr>'
            '<tr><td>1.15</td><td>1.20</td><td>[REDACTED]</td><td>+5.4%</td></tr></table>')
    assert rows(html) == [["HR applied to PFS for TTD for: Capivasertib plus fulvestrant arm", "Alpelisib plus fulvestrant",
                           "ICER* (capivasertib plus fulvestrant vs. alpelisib plus fulvestrant)", "% change from base case"],
                          ["1.15", "1.20", "[REDACTED]", "+5.4%"]]


def test_wrapped_prose_row_is_joined_cell_by_cell():
    html = ('<table><tr><th>Item</th><th>Detail</th></tr>'
            '<tr><td>Patient access</td><td>A Patient Access Scheme is available for sparsentan</td></tr>'
            '<tr><td>scheme (if</td><td>[REDACTED].</td></tr>'
            '<tr><td>applicable)</td><td></td></tr>'
            '<tr><td>Marketing authorisation</td><td>Granted in 2024.</td></tr></table>')
    assert rows(html) == [["Item", "Detail"], ["Patient access scheme (if applicable)", "A Patient Access Scheme is available for sparsentan [REDACTED]."],
                          ["Marketing authorisation", "Granted in 2024."]]
