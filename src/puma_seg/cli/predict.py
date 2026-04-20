"""CLI entry point for `puma-predict`."""

from __future__ import annotations

from puma_seg.cli._predict_impl import main

__all__ = ["main"]


if __name__ == "__main__":
    main()
