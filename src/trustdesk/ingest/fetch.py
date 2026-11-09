"""Polite crawler for public policy pages (D2 steps 1-3: source list, fetch, clean).

Stays inside each source's allow_prefix, honours robots.txt, waits between requests,
and saves each page as Markdown with front matter (url, title, fetched_at, content_hash).
"""

from __future__ import annotations

import hashlib
import re
import time
import urllib.robotparser
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urldefrag, urljoin, urlparse

import httpx
import yaml
from bs4 import BeautifulSoup
from markdownify import markdownify

from trustdesk.config import Source, SourcesCfg

# Docsy/Hugo content containers first, then generic fallbacks.
CONTENT_SELECTORS = ("div.td-content", "main article", "article", "main", "div#content")
SKIP_EXT = re.compile(r"\.(png|jpe?g|gif|svg|pdf|zip|xml|json|ics|mp4|webp)$", re.I)


@dataclass
class Page:
    url: str
    title: str
    markdown: str
    links: list[str]


def normalize_url(base: str, href: str) -> str | None:
    """Absolute URL without fragment or query; None for non-http links."""
    if not href or href.startswith(("mailto:", "tel:", "javascript:")):
        return None
    url, _ = urldefrag(urljoin(base, href))
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return None
    path = parsed.path or "/"
    if not SKIP_EXT.search(path) and not path.endswith("/") and "." not in path.rsplit("/", 1)[-1]:
        path += "/"
    return f"{parsed.scheme}://{parsed.netloc}{path}"


def in_scope(url: str, prefix: str, deny: Sequence[str] = ()) -> bool:
    return (
        url.startswith(prefix)
        and not url.startswith(tuple(deny))
        and not SKIP_EXT.search(urlparse(url).path)
    )


def is_priority(url: str, keywords: Sequence[str]) -> bool:
    path = urlparse(url).path.lower()
    return any(k in path for k in keywords)


def parse_page(url: str, html: str) -> Page:
    soup = BeautifulSoup(html, "html.parser")
    title_tag = soup.find("h1") or soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else url
    links = [u for a in soup.find_all("a", href=True) if (u := normalize_url(url, str(a["href"])))]
    node = None
    for sel in CONTENT_SELECTORS:
        node = soup.select_one(sel)
        if node:
            break
    node = node or soup.body or soup
    for tag in node.select("script, style, nav, footer, aside, .td-page-meta, .feedback--title"):
        tag.decompose()
    md = markdownify(str(node), heading_style="ATX", strip=["img"])
    md = re.sub(r"\n{3,}", "\n\n", md).strip()
    return Page(url=url, title=title, markdown=md, links=links)


def slug_for(url: str) -> str:
    path = urlparse(url).path.strip("/").replace("/", "__") or "index"
    return re.sub(r"[^A-Za-z0-9_.-]", "-", path)[:150]


def save_page(page: Page, out_dir: Path, source_name: str) -> Path:
    digest = hashlib.sha256(page.markdown.encode("utf-8")).hexdigest()
    meta = {
        "url": page.url,
        "title": page.title,
        "source": source_name,
        "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "content_hash": digest,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{slug_for(page.url)}.md"
    front = yaml.safe_dump(meta, sort_keys=False, allow_unicode=True)
    path.write_text(f"---\n{front}---\n\n{page.markdown}\n", encoding="utf-8")
    return path


def _robots_for(client: httpx.Client, url: str, agent: str) -> urllib.robotparser.RobotFileParser:
    p = urlparse(url)
    rp = urllib.robotparser.RobotFileParser()
    try:
        r = client.get(f"{p.scheme}://{p.netloc}/robots.txt")
        rp.parse(r.text.splitlines() if r.status_code == 200 else [])
    except httpx.HTTPError:
        rp.parse([])
    return rp


def crawl_source(
    src: Source, cfg: SourcesCfg, out_dir: Path, log: Callable[[str], None] = print
) -> list[Path]:
    """Breadth-first crawl of one source; policy-like URLs jump the queue.

    Returns saved file paths.
    """
    saved: list[Path] = []
    headers = {"User-Agent": cfg.user_agent}
    with httpx.Client(headers=headers, timeout=30, follow_redirects=True) as client:
        robots = _robots_for(client, src.allow_prefix, cfg.user_agent)
        queue: deque[str] = deque(u for s in src.seeds if (u := normalize_url(s, s)))
        seen: set[str] = set(queue)
        while queue and len(saved) < cfg.max_pages:
            url = queue.popleft()
            if not robots.can_fetch(cfg.user_agent, url):
                continue
            try:
                resp = client.get(url)
            except httpx.HTTPError as exc:
                log(f"  skip {url}: {exc}")
                continue
            final = normalize_url(str(resp.url), str(resp.url)) or url
            ctype = resp.headers.get("content-type", "")
            if (
                resp.status_code != 200
                or "text/html" not in ctype
                or not in_scope(final, src.allow_prefix, src.deny_prefixes)
            ):
                continue
            page = parse_page(final, resp.text)
            if len(page.markdown) > 200:
                saved.append(save_page(page, out_dir, src.name))
                log(f"  [{len(saved):>3}] {final}")
            for link in page.links:
                if link not in seen and in_scope(link, src.allow_prefix, src.deny_prefixes):
                    seen.add(link)
                    if is_priority(link, src.priority_keywords):
                        queue.appendleft(link)
                    else:
                        queue.append(link)
            time.sleep(cfg.delay_s)
    return saved
