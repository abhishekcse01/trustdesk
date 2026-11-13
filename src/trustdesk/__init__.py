"""TrustDesk: cited answers to security questionnaires from policy documents."""

from trustdesk._compat import apply_all as _apply_compat

# Must run before chromadb or langgraph are imported anywhere (see trustdesk._compat).
_apply_compat()

__version__ = "0.1.0"
