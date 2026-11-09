"""Model access with JSON-schema structured output.

`ollama/<tag>` models are called on the local Ollama server directly, because its native API
supports a JSON schema in `format` and `think: false` for Qwen3 reliably. Any other model
name goes through LiteLLM (install the `api` extra), so agents can switch by config alone.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import httpx

from trustdesk.config import LlmCfg


@dataclass
class LlmResult:
    content: str
    model: str
    seconds: float
    prompt_tokens: int = 0
    output_tokens: int = 0
    prompt_tok_s: float = 0.0
    output_tok_s: float = 0.0


def _ollama(
    model: str, messages: list[dict[str, str]], schema: dict[str, Any], cfg: LlmCfg
) -> LlmResult:
    body = {
        "model": model,
        "messages": messages,
        "format": schema,
        "stream": False,
        "think": False,
        "options": {"temperature": cfg.temperature, "num_ctx": cfg.num_ctx},
    }
    t0 = time.perf_counter()
    r = httpx.post(f"{cfg.ollama_url}/api/chat", json=body, timeout=cfg.timeout_s)
    r.raise_for_status()
    data = r.json()
    secs = time.perf_counter() - t0
    pe, pd = data.get("prompt_eval_count", 0), data.get("prompt_eval_duration", 0)
    ec, ed = data.get("eval_count", 0), data.get("eval_duration", 0)
    return LlmResult(
        content=data["message"]["content"],
        model=f"ollama/{model}",
        seconds=secs,
        prompt_tokens=pe,
        output_tokens=ec,
        prompt_tok_s=pe / (pd / 1e9) if pd else 0.0,
        output_tok_s=ec / (ed / 1e9) if ed else 0.0,
    )


def _litellm(
    model: str, messages: list[dict[str, str]], schema: dict[str, Any], cfg: LlmCfg
) -> LlmResult:
    try:
        import litellm
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise SystemExit("API models need the extra: uv sync --extra api") from exc
    t0 = time.perf_counter()
    resp = litellm.completion(
        model=model,
        messages=messages,
        temperature=cfg.temperature,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "answer", "schema": schema},
        },
        timeout=cfg.timeout_s,
    )
    usage = getattr(resp, "usage", None)
    return LlmResult(
        content=resp.choices[0].message.content or "",
        model=model,
        seconds=time.perf_counter() - t0,
        prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
        output_tokens=getattr(usage, "completion_tokens", 0) or 0,
    )


def complete_json(
    model: str, system: str, user: str, schema: dict[str, Any], cfg: LlmCfg
) -> LlmResult:
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    if model.startswith("ollama/"):
        return _ollama(model.removeprefix("ollama/"), messages, schema, cfg)
    return _litellm(model, messages, schema, cfg)
