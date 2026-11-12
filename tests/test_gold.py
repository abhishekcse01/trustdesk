import csv
import json
from pathlib import Path

from trustdesk.evals.gold import FIELDS, RANKINGS, pages_in_order, score_sheet
from trustdesk.retrieval.dense import ChunkHit


def hit(url: str) -> ChunkHit:
    return ChunkHit(chunk_id=url, text="t", url=url, title="T", heading_path="H", score=1.0)


def test_pages_in_order_dedupes_chunks_to_pages() -> None:
    assert pages_in_order([hit("a"), hit("b"), hit("a"), hit("c")]) == ["a", "b", "c"]


def test_score_sheet_uses_only_hand_marked_gold(tmp_path: Path) -> None:
    rankings = {
        "Q1": {
            "dense": ["p1", "p2", "p3"],
            "dense + rerank": ["p2", "p1", "p3"],
            "hybrid (RRF)": ["p4", "p5", "p2"],
            "hybrid + rerank": ["p5", "p6", "p7"],
        },
        "Q2": {
            name: ["x"] for name in ["dense", "dense + rerank", "hybrid (RRF)", "hybrid + rerank"]
        },
    }
    (tmp_path / RANKINGS).write_text(json.dumps(rankings), encoding="utf-8")
    sheet = tmp_path / "s.csv"
    rows = [
        {"qid": "Q1", "page_url": "p2", "is_gold": "y"},
        {"qid": "Q1", "page_url": "p1", "is_gold": "n"},
        {"qid": "Q2", "page_url": "x", "is_gold": "n"},  # labelled, but no gold page exists
    ]
    with sheet.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows([{k: r.get(k, "") for k in FIELDS} for r in rows])
    res = {r["config"]: r for r in score_sheet(sheet)}
    assert res["dense"]["questions"] == 1  # Q2 has no gold page, so it is not scored
    assert res["dense"]["hit_at_1"] == 0.0 and res["dense"]["hit_at_3"] == 1.0
    assert res["dense + rerank"]["hit_at_1"] == 1.0
    assert res["hybrid (RRF)"]["mrr_at_10"] == round(1 / 3, 3)
    assert res["hybrid + rerank"]["hit_at_3"] == 0.0
    assert res["_coverage"]["no_gold_found"] == ["Q2"]
