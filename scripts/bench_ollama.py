"""Measure local Ollama speed on this laptop (roadblock R1).

Usage: uv run python scripts/bench_ollama.py qwen3:4b [qwen3:8b ...]
Sends a ~2,500-token prompt (the size of one questionnaire question with 3 chunks)
and reports prompt-processing and generation speed from Ollama's own counters.
"""

from __future__ import annotations

import sys
import time

import httpx

URL = "http://localhost:11434/api/generate"
FILLER = (
    "Access to production systems requires multi-factor authentication and is reviewed "
    "quarterly by the system owner. Customer data is encrypted at rest and in transit. "
)
PROMPT = FILLER * 75 + "\n\nIn one sentence of about 120 words, summarise the policy above."


def bench(model: str) -> None:
    # Warm-up load so the timing excludes reading the model from disk.
    httpx.post(
        URL,
        json={
            "model": model,
            "prompt": "hi",
            "stream": False,
            "think": False,
            "options": {"num_predict": 1},
        },
        timeout=600,
    )
    t0 = time.perf_counter()
    r = httpx.post(
        URL,
        json={
            "model": model,
            "prompt": PROMPT,
            "stream": False,
            "think": False,
            "options": {"num_ctx": 8192, "num_predict": 160, "temperature": 0},
        },
        timeout=1200,
    )
    r.raise_for_status()
    d = r.json()
    wall = time.perf_counter() - t0
    pp = d["prompt_eval_count"] / (d["prompt_eval_duration"] / 1e9)
    tg = d["eval_count"] / (d["eval_duration"] / 1e9)
    print(
        f"{model:<12} prompt {d['prompt_eval_count']:>5} tok @ {pp:6.1f} tok/s | "
        f"output {d['eval_count']:>4} tok @ {tg:5.1f} tok/s | wall {wall:5.1f}s"
    )


if __name__ == "__main__":
    for m in sys.argv[1:] or ["qwen3:4b"]:
        bench(m)
