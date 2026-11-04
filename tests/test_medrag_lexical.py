from medrag.lexical import BM25, tokenize


def test_tokenize_keeps_numbers_whole_and_splits_hyphens():
    assert tokenize("The ICER was £52,681 per QALY in CAPItello-291 (HR 0.60).") == ["the", "icer", "was", "52,681", "per", "qaly", "in", "capitello", "291", "hr", "0.60"]


def test_bm25_prefers_rare_terms_and_ignores_unknown_ones():
    docs = {"a": tokenize("the company base case ICER was £24,567 per QALY"),
            "b": tokenize("the EAG base case ICER was £52,681 per QALY"),
            "c": tokenize("the committee discussed the company and the EAG")}
    bm = BM25(docs)
    assert bm.search(tokenize("£52,681"), 3)[0][0] == "b"
    assert bm.search(tokenize("zzz unknown"), 3) == []
    top = [cid for cid, _ in bm.search(tokenize("company ICER"), 3)]
    assert top[0] == "a" and set(top) == {"a", "b", "c"}
    assert BM25({}).search(["x"], 3) == []
