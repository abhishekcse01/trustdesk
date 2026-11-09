from trustdesk.ingest.chunk import approx_tokens, chunk_page, split_sections, window

PAGE = """# Encryption Policy

Intro text about the policy scope and owners of this document for all systems.

## Data at rest

All customer data stored in production databases is encrypted at rest using AES-256.
Keys are managed in a cloud KMS and rotated every year by the infrastructure team.

## Data in transit

### TLS

Connections use TLS 1.2 or higher. Older protocol versions are disabled on all endpoints.
"""

META = {"url": "https://example.com/enc/", "title": "Encryption Policy", "content_hash": "abc"}


def test_split_sections_keeps_heading_path() -> None:
    paths = [p for p, _ in split_sections(PAGE, "Encryption Policy")]
    assert paths == [
        "Encryption Policy",
        "Encryption Policy > Data at rest",
        "Encryption Policy > Data in transit > TLS",
    ]


def test_small_sections_merge_forward() -> None:
    chunks = chunk_page(META, PAGE, target=300, overlap=50, min_tokens=20)
    # The 14-word intro is merged into the next section instead of standing alone.
    assert chunks[0].heading_path == "Encryption Policy"
    assert "AES-256" in chunks[0].text
    assert all(c.url == META["url"] for c in chunks)


def test_window_respects_size_and_overlap() -> None:
    text = " ".join(f"w{i}" for i in range(1000))
    pieces = window(text, target=133, overlap=27)  # ~100 words, ~20 overlap
    assert all(len(p.split()) <= 100 for p in pieces)
    first, second = pieces[0].split(), pieces[1].split()
    assert first[-20:] == second[:20]
    assert pieces[-1].split()[-1] == "w999"


def test_chunk_ids_are_stable_and_unique() -> None:
    a = chunk_page(META, PAGE, min_tokens=5)
    b = chunk_page(META, PAGE, min_tokens=5)
    assert [c.id for c in a] == [c.id for c in b]
    assert len({c.id for c in a}) == len(a)


def test_approx_tokens() -> None:
    assert approx_tokens("one two three") == 3
