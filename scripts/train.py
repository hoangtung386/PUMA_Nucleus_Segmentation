"""Script entry point for training.

Requires package installation first:
    pip install -e ".[dev]"
"""

from __future__ import annotations

from puma_seg.cli._train_impl import main

if __name__ == "__main__":
    main()
