"""Support ``python -m llmverify`` alongside the installed ``llmverify`` script."""

from __future__ import annotations

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
