"""Load settings.yaml and sources.yaml into typed models."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel


class ChunkingCfg(BaseModel):
    target_tokens: int = 300
    overlap_tokens: int = 50
    min_tokens: int = 40


class EmbeddingCfg(BaseModel):
    model: str = "BAAI/bge-small-en-v1.5"
    batch_size: int = 64


class RetrievalCfg(BaseModel):
    top_k: int = 3
    mode: Literal["dense", "hybrid"] = "dense"
    rerank: bool = False
    candidates: int = 20  # pool size per retriever before fusion / reranking
    rrf_k: int = 60
    reranker_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"


class LlmCfg(BaseModel):
    drafter: str = "ollama/qwen3:4b"
    ollama_url: str = "http://localhost:11434"
    temperature: float = 0.1
    num_ctx: int = 8192
    timeout_s: float = 600


class EvalCfg(BaseModel):
    # Empty string = pick automatically from whichever API key is in .env.
    reference_model: str = ""
    judge_model: str = ""
    api_drafter: str = ""
    reference_top_k: int = 4
    # Token budget for API calls (free tier): cap each reply and each context chunk.
    max_output_tokens: int = 400
    judge_chunk_chars: int = 800
    # Seconds between API calls; 0 = automatic (4.5 s for Gemini's free tier, 15 req/min).
    min_interval_s: float = 0.0
    max_retries: int = 4


class GraphCfg(BaseModel):
    abstain_threshold: float = 0.5  # evidence below this -> abstain (calibrate with `calibrate`)
    support_threshold: float = 0.3
    max_retries: int = 1
    retry_extra_k: int = 2  # widen context by this many chunks on retry
    checkpoint_db: str = "data/checkpoints.sqlite"
    audit_db: str = "data/trustdesk.sqlite"


class TracingCfg(BaseModel):
    service_name: str = "trustdesk"
    jsonl_dir: str = "reports/traces"


class Settings(BaseModel):
    data_dir: str = "data"
    raw_dir: str = "data/raw"
    chroma_dir: str = "data/chroma"
    collection: str = "policies"
    chunking: ChunkingCfg = ChunkingCfg()
    embedding: EmbeddingCfg = EmbeddingCfg()
    retrieval: RetrievalCfg = RetrievalCfg()
    llm: LlmCfg = LlmCfg()
    eval: EvalCfg = EvalCfg()
    graph: GraphCfg = GraphCfg()
    tracing: TracingCfg = TracingCfg()


# Defaults per provider: (reference drafter, judge, API-mode drafter).
PROVIDER_DEFAULTS = {
    "ANTHROPIC_API_KEY": (
        "anthropic/claude-sonnet-5-5",
        "anthropic/claude-haiku-5-5",
        "anthropic/claude-haiku-5-5",
    ),
    # Free tier: Flash-Lite allows ~500 requests/day; Flash only ~20/day (Nov 2026 reports).
    # Run `trustdesk models` to see the exact names your key can use.
    "GEMINI_API_KEY": (
        "gemini/gemini-3.5-flash-lite",
        "gemini/gemini-3.5-flash-lite",
        "gemini/gemini-3.5-flash-lite",
    ),
}


def min_interval(model: str) -> float:
    """Pause between API calls, to stay under free-tier rate limits."""
    cfg = settings().eval
    if cfg.min_interval_s:
        return cfg.min_interval_s
    return 4.5 if model.startswith("gemini/") else 0.0


def api_model(kind: str) -> str:
    """Resolve the model for kind in {"reference", "judge", "drafter"}.

    An explicit value in settings.yaml wins; otherwise the provider whose key is set.
    """
    load_env()
    cfg = settings().eval
    explicit = {
        "reference": cfg.reference_model,
        "judge": cfg.judge_model,
        "drafter": cfg.api_drafter,
    }
    if explicit[kind]:
        return explicit[kind]
    idx = {"reference": 0, "judge": 1, "drafter": 2}[kind]
    for key, models in PROVIDER_DEFAULTS.items():
        if os.environ.get(key):
            return models[idx]
    if os.environ.get("OPENAI_API_KEY"):
        raise SystemExit(
            f"OPENAI_API_KEY found: set eval.{kind}_model (e.g. openai/<model>) "
            "in config/settings.yaml"
        )
    raise SystemExit(
        "No API key found. Add GEMINI_API_KEY, ANTHROPIC_API_KEY or OPENAI_API_KEY to .env"
    )


@lru_cache(maxsize=1)
def load_env() -> bool:
    """Load .env from the repo root (keys stay in the environment, never in settings)."""
    from dotenv import load_dotenv

    return load_dotenv(repo_root() / ".env")


class Source(BaseModel):
    name: str
    allow_prefix: str
    seeds: list[str]
    deny_prefixes: list[str] = []
    priority_keywords: list[str] = []


class SourcesCfg(BaseModel):
    attribution: str
    user_agent: str
    delay_s: float = 1.0
    max_pages: int = 250
    sources: list[Source]


def repo_root() -> Path:
    """Repo root: $TRUSTDESK_ROOT, else the nearest parent of cwd holding pyproject.toml."""
    env = os.environ.get("TRUSTDESK_ROOT")
    if env:
        return Path(env).resolve()
    here = Path.cwd().resolve()
    for p in (here, *here.parents):
        if (p / "pyproject.toml").exists() and (p / "config").is_dir():
            return p
    return here


def _load_yaml(name: str) -> dict[str, object]:
    path = repo_root() / "config" / name
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a mapping")
    return data


@lru_cache(maxsize=1)
def settings() -> Settings:
    return Settings.model_validate(_load_yaml("settings.yaml"))


@lru_cache(maxsize=1)
def sources() -> SourcesCfg:
    return SourcesCfg.model_validate(_load_yaml("sources.yaml"))


def path(rel: str) -> Path:
    """Resolve a settings path relative to the repo root."""
    return repo_root() / rel
