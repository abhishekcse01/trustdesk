"""Allow `python -m trustdesk ...` (Windows Smart App Control can block venv .exe launchers)."""

from trustdesk.cli import app

app()
