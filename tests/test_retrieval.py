from rank_bm25 import BM25Okapi

from trustdesk.retrieval.bm25 import Bm25Index, tokenize
from trustdesk.retrieval.search import rrf


def test_tokenize_keeps_technical_terms_and_drops_stopwords() -> None:
    assert tokenize("Is data encrypted with AES-256 and TLS 1.2?") == [
        "data",
        "encrypted",
        "aes-256",
        "tls",
        "1.2",
    ]


def test_rrf_rewards_agreement_between_lists() -> None:
    fused = rrf([["a", "b", "c"], ["b", "c", "d"]], k=60)
    ids = [d for d, _ in fused]
    assert ids[0] == "b"  # ranked 2nd and 1st beats 1st and absent
    assert set(ids) == {"a", "b", "c", "d"}
    assert fused[0][1] == 1 / 62 + 1 / 61


def test_rrf_single_list_keeps_order() -> None:
    assert [d for d, _ in rrf([["x", "y", "z"]])] == ["x", "y", "z"]


def test_bm25_finds_exact_terms() -> None:
    docs = [
        "All laptops use full disk encryption.",
        "Connections require TLS 1.2 or higher; TLS 1.0 is disabled.",
        "Employees complete security awareness training every year.",
    ]
    idx = Bm25Index(
        ids=["d0", "d1", "d2"],
        docs=docs,
        metas=[{}, {}, {}],
        bm25=BM25Okapi([tokenize(d) for d in docs]),
    )
    top = idx.top("Which TLS version is required?", 2)
    assert top[0][0] == 1
    assert all(score > 0 for _, score in top)
