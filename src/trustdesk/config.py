"""Load settings.yaml and sources.yaml into typed models."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

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


class LlmCfg(BaseModel):
    drafter: str = "ollama/qwen3:4b"
    ollama_url: str = "http://localhost:11434"
    temperature: float = 0.1
    num_ctx: int = 8192
    timeout_s: float = 600


class Settings(BaseModel):
    data_dir: str = "data"
    raw_dir: str = "data/raw"
    chroma_dir: str = "data/chroma"
    collection: str = "policies"
    chunking: ChunkingCfg = ChunkingCfg()
    embedding: EmbeddingCfg = EmbeddingCfg()
    retrieval: RetrievalCfg = RetrievalCfg()
    llm: LlmCfg = LlmCfg()


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
