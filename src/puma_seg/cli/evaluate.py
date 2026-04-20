"""CLI entry point for `puma-eval`."""

from __future__ import annotations

from puma_seg.cli._evaluate_impl import main

__all__ = ["main"]


if __name__ == "__main__":
    main()
