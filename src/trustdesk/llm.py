"""Model access with JSON-schema structured output.

`ollama/<tag>` models are called on the local Ollama server directly, because its native API
supports a JSON schema in `format` and `think: false` for Qwen3 reliably. Any other model
name goes through LiteLLM (install the `api` extra), so agents can switch by config alone.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

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


_last_api_call = 0.0  # monotonic time of the last API call, for the free-tier throttle


def _litellm(
    model: str, messages: list[dict[str, str]], schema: dict[str, Any], cfg: LlmCfg
) -> LlmResult:
    try:
        import litellm
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise SystemExit("API models need the extra: uv sync --extra api") from exc
    from litellm.exceptions import (
        APIConnectionError,
        RateLimitError,
        ServiceUnavailableError,
        Timeout,
    )

    from trustdesk.config import load_env, min_interval, settings

    load_env()
    litellm.suppress_debug_info = True
    global _last_api_call
    wait = min_interval(model) - (time.monotonic() - _last_api_call)
    if wait > 0:
        time.sleep(wait)
    t0 = time.perf_counter()
    retries = settings().eval.max_retries
    for attempt in range(retries + 1):
        try:
            _last_api_call = time.monotonic()
            # Gemini 3+ deprecates sampling parameters; other providers still get temperature.
            extra: dict[str, Any] = {"max_tokens": settings().eval.max_output_tokens}
            if not model.startswith("gemini/gemini-3"):
                extra["temperature"] = cfg.temperature
            resp = litellm.completion(
                model=model,
                messages=messages,
                **extra,
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": "answer", "schema": schema},
                },
                timeout=cfg.timeout_s,
            )
            break
        except (RateLimitError, ServiceUnavailableError, Timeout, APIConnectionError):
            if attempt == retries:
                raise
            time.sleep(min(60.0, 5.0 * 2**attempt))  # 5, 10, 20, 40 s
    usage = getattr(resp, "usage", None)
    return LlmResult(
        content=resp.choices[0].message.content or "",
        model=model,
        seconds=time.perf_counter() - t0,
        prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
        output_tokens=getattr(usage, "completion_tokens", 0) or 0,
    )


def inline_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Resolve local $ref/$defs so every provider accepts the schema."""
    defs = schema.get("$defs", {})

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                return walk(defs[node["$ref"].split("/")[-1]])
            return {k: walk(v) for k, v in node.items() if k != "$defs"}
        if isinstance(node, list):
            return [walk(v) for v in node]
        return node

    result: dict[str, Any] = walk(schema)
    return result


def complete_json(
    model: str, system: str, user: str, schema: dict[str, Any], cfg: LlmCfg, name: str = "llm"
) -> LlmResult:
    from trustdesk.telemetry import INPUT, MODEL, OUTPUT, TOK_COMPLETION, TOK_PROMPT, span

    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    schema = inline_schema(schema)
    with span(name, "LLM", **{MODEL: model, INPUT: user[:4000]}) as s:
        if model.startswith("ollama/"):
            res = _ollama(model.removeprefix("ollama/"), messages, schema, cfg)
        else:
            res = _litellm(model, messages, schema, cfg)
        s.set_attribute(OUTPUT, res.content[:4000])
        s.set_attribute(TOK_PROMPT, res.prompt_tokens)
        s.set_attribute(TOK_COMPLETION, res.output_tokens)
        return res


T = TypeVar("T", bound=BaseModel)


def parse_model[T: BaseModel](content: str, cls: type[T]) -> T:
    """Parse JSON (optionally fenced) into a Pydantic model."""
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?|```$", "", content, flags=re.M).strip()
    return cls.model_validate(json.loads(content))


def structured[T: BaseModel](
    model: str, system: str, user: str, cls: type[T], cfg: LlmCfg, name: str = "llm"
) -> tuple[T | None, str | None, LlmResult, float]:
    """Call a model for JSON matching cls, with one repair retry (R10).

    Returns (parsed or None, last error, last raw result, total seconds).
    """
    schema = cls.model_json_schema()
    total = 0.0
    error: str | None = None
    res: LlmResult | None = None
    for attempt in range(2):
        prompt = user
        if attempt:
            prompt += f"\n\nYour previous reply was invalid: {error}. Reply with valid JSON only."
        res = complete_json(model, system, prompt, schema, cfg, name=name)
        total += res.seconds
        try:
            return parse_model(res.content, cls), None, res, total
        except (json.JSONDecodeError, ValidationError) as exc:
            error = str(exc).splitlines()[0][:200]
    assert res is not None
    return None, error, res, total
