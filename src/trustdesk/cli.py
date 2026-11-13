"""Command line: fetch -> index -> ask / batch."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import typer
import yaml
from rich.console import Console
from rich.table import Table

from trustdesk import config
from trustdesk.agents.answer import AnswerResult, answer
from trustdesk.ingest.fetch import crawl_source
from trustdesk.ingest.index import build_index
from trustdesk.retrieval.search import search

app = typer.Typer(add_completion=False, help="TrustDesk: cited answers to security questionnaires.")
golden_app = typer.Typer(help="Golden set: reference answers for evaluation.")
app.add_typer(golden_app, name="golden")
console = Console()


def _ids(ids: str | None) -> set[str] | None:
    return {i.strip() for i in ids.split(",")} if ids else None


@golden_app.command("build")
def golden_build(
    ids: str | None = typer.Option(None, help="Comma-separated IDs, e.g. G001,G002"),
) -> None:
    """Draft reference answers with the API reference model (skips verified entries)."""
    from trustdesk.evals.golden import build_references

    refs = build_references(config.settings(), _ids(ids), log=console.print)
    console.print(f"[green]{len(refs)} references in evals/golden/references.yaml[/]")


@golden_app.command("stats")
def golden_stats() -> None:
    """Label counts, expected-vs-drafted agreement and verification progress."""
    from collections import Counter

    from trustdesk.evals.golden import load_questions, load_references

    qs = {q.id: q for q in load_questions()}
    refs = load_references()
    labels = Counter(r.label for r in refs.values())
    same = sum(1 for r in refs.values() if qs[r.id].expect == r.label)
    console.print(
        f"questions {len(qs)} · references {len(refs)} · verified "
        f"{sum(r.verified for r in refs.values())}"
    )
    console.print(f"labels {dict(labels)} · drafted label matches our prior {same}/{len(refs)}")


@app.command("eval")
def eval_cmd(
    mode: str = typer.Option("local", help="local = Ollama drafter, api = API drafter"),
    limit: int | None = typer.Option(None, help="Only the first N questions"),
    ids: str | None = typer.Option(None, help="Comma-separated question IDs"),
    no_judge: bool = typer.Option(False, help="Skip the LLM judge"),
    verified_only: bool = typer.Option(False, help="Only human-verified references"),
) -> None:
    """Run the golden set, judge it, and write reports/evals/eval-<time>.md."""
    from trustdesk.evals.run import run_eval

    summary, report = run_eval(
        config.settings(), mode, limit, _ids(ids), not no_judge, verified_only, log=console.print
    )
    console.print_json(json.dumps(summary))
    console.print(f"[green]Report: {report}[/]")


@app.command("report")
def report_cmd(runs: list[Path]) -> None:
    """Combine eval JSONL files from reports/runs/ into one report (e.g. after a resumed run)."""
    from trustdesk.evals.run import report_from_runs

    cfg = config.settings()
    summary, report = report_from_runs(runs, cfg.llm.drafter, config.api_model("judge"))
    console.print_json(json.dumps(summary))
    console.print(f"[green]Report: {report}[/]")


@app.command("run")
def run_cmd(
    question: str,
    decision: str | None = typer.Option(None, help="approve | edit | reject (skips the prompt)"),
    text: str = typer.Option("", help="Replacement answer when decision is edit"),
    reviewer: str = typer.Option("reviewer", help="Name recorded in the audit log"),
) -> None:
    """Answer one question through the phase 4 graph, pausing for your review."""
    import uuid

    from langgraph.types import Command

    from trustdesk.graph.runtime import open_graph

    cfg = config.settings()
    graph, conn = open_graph(cfg)
    thread = {"configurable": {"thread_id": f"q-{uuid.uuid4().hex[:8]}"}}
    try:
        state = graph.invoke({"question_id": "cli", "question": question}, thread)
        payload = state["__interrupt__"][0].value
        d, v = payload["draft"] or {}, payload["verification"]
        console.print(
            f"[bold]Draft[/] ({'abstained' if d.get('abstain') else d.get('answer_choice')})"
        )
        console.print(d.get("abstain_reason") if d.get("abstain") else d.get("answer_text"))
        for c in d.get("citations", []):
            console.print(f'  [dim]"{c["quote"]}"[/]')
        console.print(
            f"[dim]verifier: route={v.get('route')} evidence={v.get('evidence')} "
            f"citations={v.get('citation_ratio')} support={v.get('support')} "
            f"confidence={v.get('confidence')} | {v.get('reason')}[/]"
        )
        if payload["guard_flags"]:
            console.print(f"[yellow]Guard flags: {payload['guard_flags']}[/]")
        choice = decision or typer.prompt("Decision [approve/edit/reject]", default="approve")
        if choice == "edit" and not text:
            text = typer.prompt("Your answer")
        final = graph.invoke(
            Command(resume={"decision": choice, "text": text, "reviewer": reviewer}), thread
        )
        console.print(f"[green]{final['status']}[/] and recorded in the audit log.")
    finally:
        conn.close()


@app.command("audit")
def audit_cmd(limit: int = 10) -> None:
    """Show the most recent reviewed answers from the audit log."""
    from trustdesk.store import audit

    table = Table("when", "question", "decision", "reviewer", "answer")
    for r in audit.recent(config.path(config.settings().graph.audit_db), limit):
        table.add_row(
            r["decided_at"], r["question"][:40], r["decision"], r["reviewer"], r["answer_text"][:50]
        )
    console.print(table)


@app.command("calibrate")
def calibrate_cmd() -> None:
    """Pick the Verifier's abstain threshold from the golden set (local only, no API calls)."""
    from trustdesk.graph.calibrate import calibrate

    console.print_json(json.dumps(calibrate(config.settings(), log=console.print)))


gold_app = typer.Typer(help="Unbiased retrieval benchmark with hand-marked gold pages.")
app.add_typer(gold_app, name="gold")


@gold_app.command("sheet")
def gold_sheet(n: int = 30, depth: int = 5) -> None:
    """Pool the top pages from every retrieval setup into a sheet for hand-marking."""
    from trustdesk.evals.gold import make_sheet

    sheet = make_sheet(config.settings(), n, depth, log=console.print)
    console.print(f"[green]Mark is_gold (y/n) in {sheet}[/]")


@gold_app.command("score")
def gold_score(sheet: Path | None = None) -> None:
    """Score every retrieval setup against the hand-marked gold pages."""
    from trustdesk.evals.gold import score_sheet

    results = score_sheet(sheet)
    table = Table("config", "questions", "hit@1", "hit@3", "MRR@10")
    for r in results:
        if r["config"] != "_coverage" and r["questions"]:
            table.add_row(
                r["config"],
                str(r["questions"]),
                f"{r['hit_at_1']:.0%}",
                f"{r['hit_at_3']:.0%}",
                f"{r['mrr_at_10']:.2f}",
            )
    console.print(table)
    console.print_json(json.dumps(results[-1]))


@app.command("grid")
def grid_cmd(
    configs: str | None = typer.Option(
        None, help='Comma-separated names, e.g. "dense,hybrid (RRF)"'
    ),
) -> None:
    """Compare retrieval configs on the golden set (hit@1, hit@3, MRR). No API calls."""
    from trustdesk.evals.grid import run_grid

    names = [c.strip() for c in configs.split(",")] if configs else None
    _, report = run_grid(config.settings(), names, log=console.print)
    console.print(f"[green]Report: {report}[/]")


@app.command("models")
def models_cmd() -> None:
    """List the Gemini models your GEMINI_API_KEY can call (names only)."""
    import os

    import httpx

    config.load_env()
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        raise SystemExit("GEMINI_API_KEY is empty in .env")
    r = httpx.get(
        "https://generativelanguage.googleapis.com/v1beta/models",
        headers={"x-goog-api-key": key},  # header, never the URL, so the key stays out of logs
        params={"pageSize": 1000},
        timeout=30,
    )
    if r.status_code != 200:
        raise SystemExit(f"Gemini API returned {r.status_code}: check the key in .env")
    names = sorted(
        m["name"].removeprefix("models/")
        for m in r.json().get("models", [])
        if "generateContent" in m.get("supportedGenerationMethods", [])
    )
    for n in names:
        console.print(f"gemini/{n}")
    console.print(f"[dim]{len(names)} models. Current choice: {config.api_model('judge')}[/]")


@app.command("agreement")
def agreement_cmd(sheet: Path) -> None:
    """Judge-vs-human agreement and Cohen's kappa from a filled judge_validation sheet (E6)."""
    from trustdesk.evals.run import judge_agreement

    console.print_json(json.dumps(judge_agreement(sheet)))


@app.command()
def fetch() -> None:
    """Crawl the public policy sources in config/sources.yaml into data/raw/."""
    cfg, src_cfg = config.settings(), config.sources()
    out = config.path(cfg.raw_dir)
    for src in src_cfg.sources:
        console.print(f"[bold]Fetching {src.name}[/] (max {src_cfg.max_pages} pages)")
        saved = crawl_source(src, src_cfg, out, log=console.print)
        console.print(f"Saved {len(saved)} pages to {out}")
    console.print(f"[dim]{src_cfg.attribution}[/]")


@app.command()
def index() -> None:
    """Chunk, embed and index data/raw/ into ChromaDB (full rebuild)."""
    n = build_index(config.settings(), log=console.print)
    console.print(f"[green]Indexed {n} chunks.[/]")


@app.command("search")
def search_cmd(query: str, k: int = 5) -> None:
    """Show the top-k chunks for a query (retrieval debugging)."""
    for h in search(query, config.settings(), k):
        console.print(f"[bold]{h.score:.3f}[/] {h.heading_path}\n  [dim]{h.url}[/]")


def _print_result(r: AnswerResult) -> None:
    if r.draft is None:
        console.print(f"[red]Invalid model output:[/] {r.error}")
        return
    d = r.draft
    if d.abstain:
        console.print(f"[yellow]Needs human input:[/] {d.abstain_reason}")
    else:
        console.print(f"[bold]{d.answer_choice or ''}[/] {d.answer_text}")
    for c, chk in zip(d.citations, r.checks, strict=True):
        hit = next((h for h in r.hits if h.chunk_id == c.chunk_id), None)
        mark = "[green]ok[/]" if chk.in_context and chk.quote_found else "[red]unsupported[/]"
        where = f"{hit.heading_path} ({hit.url})" if hit else "unknown chunk"
        console.print(f'  {mark} "{c.quote}"\n     [dim]{where}[/]')
    console.print(
        f"[dim]{r.model} | {r.seconds}s | {r.prompt_tokens} prompt + "
        f"{r.output_tokens} output tokens | {r.output_tok_s} tok/s[/]"
    )


@app.command()
def ask(question: str) -> None:
    """Answer one question with citations."""
    _print_result(answer(question, config.settings()))


@app.command()
def batch(
    questions: Path = typer.Argument(Path("evals/phase1/questions.yaml")),
    out_dir: Path = typer.Option(Path("reports/runs")),
) -> None:
    """Answer every question in a YAML file; write JSONL and print a summary."""
    cfg = config.settings()
    items = yaml.safe_load(questions.read_text(encoding="utf-8"))["questions"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"run-{datetime.now():%Y%m%d-%H%M%S}.jsonl"
    table = Table("id", "expect", "result", "cites ok", "secs")
    with out.open("w", encoding="utf-8") as f:
        for item in items:
            console.print(f"[bold]{item['id']}[/] {item['question']}")
            r = answer(item["question"], cfg)
            row = {"id": item["id"], "expect": item.get("expect"), **r.model_dump(exclude={"hits"})}
            row["retrieved"] = [
                h.model_dump(include={"chunk_id", "url", "heading_path", "score"}) for h in r.hits
            ]
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            result = (
                "error"
                if r.draft is None
                else ("abstain" if r.draft.abstain else (r.draft.answer_choice or "text"))
            )
            n_ok = sum(c.in_context and c.quote_found for c in r.checks)
            table.add_row(
                item["id"],
                str(item.get("expect")),
                result,
                f"{n_ok}/{len(r.checks)}",
                f"{r.seconds:.0f}",
            )
    console.print(table)
    console.print(f"Wrote {out}")


if __name__ == "__main__":
    app()
