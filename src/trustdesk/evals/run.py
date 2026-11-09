"""Run the golden set through TrustDesk, judge it, and write JSONL + a Markdown summary."""

from __future__ import annotations

import csv
import json
import random
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from trustdesk.agents.answer import PROMPT_VERSION, answer
from trustdesk.config import Settings, api_model, path
from trustdesk.evals.golden import load_references
from trustdesk.evals.judge import JUDGE_VERSION, judge
from trustdesk.evals.metrics import cohen_kappa, summarize
from trustdesk.telemetry import flush


def run_eval(
    cfg: Settings,
    mode: str = "local",
    limit: int | None = None,
    ids: set[str] | None = None,
    use_judge: bool = True,
    verified_only: bool = False,
    log: Callable[[str], None] = print,
) -> tuple[dict[str, Any], Path]:
    refs = [r for r in load_references().values() if (not ids or r.id in ids)]
    if verified_only:
        refs = [r for r in refs if r.verified]
    refs = refs[:limit] if limit else refs
    if not refs:
        raise SystemExit("No references. Run `trustdesk golden build` first.")
    drafter = api_model("drafter") if mode == "api" else cfg.llm.drafter
    judge_model = api_model("judge") if use_judge else ""
    stamp = f"{datetime.now():%Y%m%d-%H%M%S}"
    out = path("reports/runs") / f"eval-{stamp}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    with out.open("w", encoding="utf-8") as f:
        for i, ref in enumerate(refs, 1):
            res = answer(ref.question, cfg, drafter=drafter)
            row: dict[str, Any] = {
                "id": ref.id,
                "domain": ref.domain,
                "label": ref.label,
                "verified": ref.verified,
                "question": ref.question,
                "answer": res.draft.model_dump() if res.draft else None,
                "error": res.error,
                "abstained": bool(res.draft and res.draft.abstain),
                "retrieved_urls": [h.url for h in res.hits],
                "expected_urls": ref.supporting_urls,
                "citations_verified": sum(c.in_context and c.quote_found for c in res.checks),
                "citations_total": len(res.checks),
                "seconds": res.seconds,
                "trace_id": res.trace_id,
            }
            if use_judge:
                verdict, err = judge(ref, res, cfg, judge_model)
                row["judge"] = verdict.model_dump() if verdict else None
                row["judge_error"] = err
            rows.append(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            j = row.get("judge") or {}
            log(
                f"  [{i:>3}/{len(refs)}] {ref.id} {ref.label:<12} -> "
                f"{'abstain' if row['abstained'] else 'answer':<7} {j.get('correctness', '-'):<17} "
                f"{row['seconds']:.0f}s"
            )
    flush()
    summary = summarize(rows)
    summary.update(
        {
            "run": stamp,
            "mode": mode,
            "drafter": drafter,
            "judge": judge_model,
            "prompt_version": PROMPT_VERSION,
            "judge_version": JUDGE_VERSION,
            "verified_references": sum(r.verified for r in refs),
        }
    )
    report = write_report(summary, rows, stamp)
    if use_judge:
        export_judge_sheet(rows, stamp)
    return summary, report


def write_report(summary: dict[str, Any], rows: list[dict[str, Any]], stamp: str) -> Path:
    def pct(v: Any) -> str:
        return "n/a" if v is None else f"{v:.0%}" if isinstance(v, float) and v <= 1 else str(v)

    lines = [
        f"# Eval run {stamp}",
        "",
        f"Drafter `{summary['drafter']}` · judge `{summary['judge'] or 'none'}` · prompt "
        f"`{summary['prompt_version']}` · {summary['questions']} questions "
        f"({summary['verified_references']} with human-verified references)",
        "",
        "| Metric | Value |",
        "| --- | --- |",
    ]
    names = {
        "correctness": "Correctness (judge, 1 / 0.5 / 0)",
        "faithfulness": "Faithfulness (claims supported by context)",
        "citation_accuracy": "Citation accuracy (verbatim)",
        "retrieval_hit_at_3": "Retrieval hit@3 (expected page in top 3)",
        "retrieval_mrr": "Retrieval MRR",
        "correct_abstention_rate": "Correct abstention (unanswerable)",
        "false_abstention_rate": "False abstention (answerable/partial)",
        "valid_output_rate": "Valid structured output",
        "median_seconds": "Median seconds per question",
        "judge_failures": "Judge parse failures (counted as missing)",
    }
    for key, label in names.items():
        if key in summary:
            v = summary[key]
            lines.append(
                f"| {label} | {v if key in ('median_seconds', 'judge_failures') else pct(v)} |"
            )
    if summary.get("correctness_by_label"):
        lines += ["", "| Label | Correctness |", "| --- | --- |"]
        lines += [f"| {k} | {pct(v)} |" for k, v in summary["correctness_by_label"].items()]
    worst = [r for r in rows if (r.get("judge") or {}).get("correctness") == "incorrect"]
    if worst:
        lines += [
            "",
            "## Incorrect answers",
            "",
            "| ID | Label | Judge rationale |",
            "| --- | --- | --- |",
        ]
        lines += [f"| {r['id']} | {r['label']} | {r['judge']['rationale']} |" for r in worst]
    file = path("reports/evals") / f"eval-{stamp}.md"
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (file.with_suffix(".json")).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return file


def export_judge_sheet(rows: list[dict[str, Any]], stamp: str, n: int = 50) -> Path:
    """E6: a CSV of up to n judged answers with an empty human_correctness column."""
    judged = [r for r in rows if r.get("judge")]
    sample = random.Random(42).sample(judged, min(n, len(judged)))
    file = path("evals/judge_validation") / f"sheet-{stamp}.csv"
    file.parent.mkdir(parents=True, exist_ok=True)
    with file.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            ["id", "label", "question", "candidate", "judge_correctness", "human_correctness"]
        )
        for r in sorted(sample, key=lambda x: x["id"]):
            a = r["answer"] or {}
            cand = (
                f"ABSTAINED: {a.get('abstain_reason')}"
                if a.get("abstain")
                else a.get("answer_text", "")
            )
            w.writerow([r["id"], r["label"], r["question"], cand, r["judge"]["correctness"], ""])
    return file


def judge_agreement(sheet: Path) -> dict[str, Any]:
    with sheet.open(encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["human_correctness"].strip()]
    judge_l = [r["judge_correctness"] for r in rows]
    human_l = [r["human_correctness"].strip() for r in rows]
    agree = sum(a == b for a, b in zip(judge_l, human_l, strict=True))
    return {
        "labelled": len(rows),
        "agreement": round(agree / len(rows), 3) if rows else None,
        "cohen_kappa": cohen_kappa(judge_l, human_l) if rows else None,
    }
