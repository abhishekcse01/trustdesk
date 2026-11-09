"""Heading-aware Markdown chunker (D2 steps 4-5: chunk, tag metadata).

Each chunk keeps the heading path it came from ("Encryption Policy > Data at rest"),
so citations can point a reviewer to the exact section of a policy page.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
TOKENS_PER_WORD = 1.33


def approx_tokens(text: str) -> int:
    return int(len(text.split()) * TOKENS_PER_WORD)


@dataclass
class Chunk:
    id: str
    text: str
    url: str
    title: str
    heading_path: str
    content_hash: str
    fetched_at: str
    position: int
    metadata: dict[str, str | int] = field(default_factory=dict)

    def as_metadata(self) -> dict[str, str | int]:
        return {
            "url": self.url,
            "title": self.title,
            "heading_path": self.heading_path,
            "content_hash": self.content_hash,
            "fetched_at": self.fetched_at,
            "position": self.position,
            **self.metadata,
        }


def read_page(path: Path) -> tuple[dict[str, str], str]:
    """Split a saved page into (front matter, markdown body)."""
    raw = path.read_text(encoding="utf-8")
    if raw.startswith("---\n"):
        end = raw.find("\n---\n", 4)
        if end != -1:
            meta = yaml.safe_load(raw[4:end]) or {}
            return {k: str(v) for k, v in meta.items()}, raw[end + 5 :].strip()
    return {}, raw


def split_sections(markdown: str, title: str) -> list[tuple[str, str]]:
    """Return (heading_path, body) pairs, one per heading section."""
    sections: list[tuple[str, str]] = []
    stack: list[tuple[int, str]] = []
    buf: list[str] = []

    def flush() -> None:
        body = "\n".join(buf).strip()
        if body:
            names = [name for _, name in stack] or [title]
            sections.append((" > ".join(names), body))
        buf.clear()

    for line in markdown.splitlines():
        m = HEADING.match(line)
        if m:
            flush()
            level, name = len(m.group(1)), m.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, name))
        else:
            buf.append(line)
    flush()
    return sections


def window(text: str, target: int, overlap: int) -> list[str]:
    """Split text into word windows of ~target tokens with ~overlap tokens shared."""
    words = text.split()
    size = max(1, int(target / TOKENS_PER_WORD))
    step = max(1, size - int(overlap / TOKENS_PER_WORD))
    if len(words) <= size:
        return [" ".join(words)]
    out = []
    for start in range(0, len(words), step):
        out.append(" ".join(words[start : start + size]))
        if start + size >= len(words):
            break
    return out


def chunk_page(
    meta: dict[str, str], markdown: str, target: int = 300, overlap: int = 50, min_tokens: int = 40
) -> list[Chunk]:
    title = meta.get("title", "")
    url = meta.get("url", "")
    sections = split_sections(markdown, title)

    # Merge tiny sections forward so a lone heading with one line is not its own chunk.
    merged: list[tuple[str, str]] = []
    carry: tuple[str, str] | None = None
    for path, body in sections:
        if carry:
            path, body = carry[0], f"{carry[1]}\n\n{body}"
            carry = None
        if approx_tokens(body) < min_tokens:
            carry = (path, body)
        else:
            merged.append((path, body))
    if carry:
        merged.append(carry)

    chunks: list[Chunk] = []
    for path, body in merged:
        for piece in window(body, target, overlap):
            pos = len(chunks)
            cid = hashlib.sha1(f"{url}|{path}|{pos}|{piece[:64]}".encode()).hexdigest()[:16]
            chunks.append(
                Chunk(
                    id=cid,
                    text=piece,
                    url=url,
                    title=title,
                    heading_path=path,
                    content_hash=meta.get("content_hash", ""),
                    fetched_at=meta.get("fetched_at", ""),
                    position=pos,
                )
            )
    return chunks
