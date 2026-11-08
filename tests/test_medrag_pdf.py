from medrag.pdf import clean_title, title_for_page


def test_clean_title_strips_file_name_noise():
    assert clean_title("1a. ID6370 Capivasertib_CAPItello291 Document B v0.1 23Jul24 [CIC_REDACTED]") == "Capivasertib CAPItello291 Document B"
    assert clean_title("\ufeffEffectiveness of a sleep intervention") == "Effectiveness of a sleep intervention"
    assert clean_title("5. ID6370 Capivasertib for breast cancer - EAG Report [redacted] - postFAC_18102024 (1)") == "Capivasertib for breast cancer - EAG Report - postFAC 18102024"
    assert clean_title("4b. ID6370 Patient expert statement KS Redacted") == "Patient expert statement KS"
    assert clean_title("[x]") == "[x]"


def test_title_for_page():
    docs = [{"title": "A", "page_start": 1, "page_end": 3}, {"title": "B", "page_start": 4, "page_end": 10}]
    assert title_for_page(docs, 3) == "A" and title_for_page(docs, 4) == "B" and title_for_page(docs, 11) is None
