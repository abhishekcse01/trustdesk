# ADR 0001: Qwen3 4B on Ollama by default, called through Ollama's native API

Date: 2026-11-09 · Status: accepted

## Context

The development laptop has no discrete GPU (AMD Ryzen 5 150, integrated Radeon, 15.2 GB RAM).
Measured with `scripts/bench_ollama.py` on a 2,053-token prompt and 160 output tokens:

| Setup | Prompt processing | Generation | Wall time |
| --- | --- | --- | --- |
| Qwen3 4B, CPU only | 30.7 tok/s | 5.3 tok/s | 107 s |
| Qwen3 4B, integrated Radeon via Vulkan (`OLLAMA_IGPU_ENABLE=1`) | 53.9 tok/s | 5.0 tok/s | 81 s |

An 8B model would roughly halve these numbers.

## Decision

- Default drafter is `ollama/qwen3:4b`, with the integrated GPU enabled through Vulkan.
- Local models are called through Ollama's native `/api/chat`, which accepts a JSON schema in
  `format` and `think: false` (Qwen3 otherwise spends output tokens on reasoning). Other model
  names go through LiteLLM, so switching to an API model is a config change.
- Keep prompts short: top 3 chunks of ~300 tokens.

## Consequences

- About 1 minute per question locally: fine for development and small runs, too slow for the full
  golden set. Phase 2 runs bulk evals and the LLM judge on an API model.
- Generation speed, not prompt processing, is the bottleneck, so output length is capped.
