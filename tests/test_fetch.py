from trustdesk.ingest.fetch import in_scope, is_priority, normalize_url, parse_page, slug_for

PREFIX = "https://handbook.gitlab.com/handbook/security/"


def test_normalize_url_drops_fragment_and_adds_slash() -> None:
    assert normalize_url(PREFIX, "policies#top") == PREFIX + "policies/"
    assert normalize_url(PREFIX, "mailto:a@b.c") is None


def test_in_scope_filters_prefix_and_assets() -> None:
    assert in_scope(PREFIX + "encryption-policy/", PREFIX)
    assert not in_scope("https://handbook.gitlab.com/handbook/marketing/", PREFIX)
    assert not in_scope(PREFIX + "diagram.png", PREFIX)


def test_deny_prefixes_and_priority() -> None:
    deny = [PREFIX + "corporate/"]
    assert not in_scope(PREFIX + "corporate/systems/linux/", PREFIX, deny)
    assert in_scope(PREFIX + "encryption-policy/", PREFIX, deny)
    assert is_priority(PREFIX + "encryption-policy/", ["policy"])
    assert not is_priority(PREFIX + "labs/", ["policy"])


def test_parse_page_extracts_main_content() -> None:
    html = """<html><head><title>T</title></head><body><nav>menu</nav>
    <div class="td-content"><h1>Encryption Policy</h1><h2>At rest</h2><p>AES-256.</p>
    <a href="../other/">x</a></div><footer>foot</footer></body></html>"""
    page = parse_page(PREFIX + "enc/", html)
    assert page.title == "Encryption Policy"
    assert "## At rest" in page.markdown and "menu" not in page.markdown
    assert PREFIX + "other/" in page.links


def test_slug_is_filesystem_safe() -> None:
    assert slug_for(PREFIX + "a b/c?d") == "handbook__security__a-b__c"
