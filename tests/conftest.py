"""Import trustdesk first so its Windows Smart App Control shims apply before any test module
imports langgraph or chromadb (see trustdesk._compat)."""

import trustdesk  # noqa: F401
