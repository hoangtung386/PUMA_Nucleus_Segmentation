"""Ensure output filenames and layout match Grand Challenge expectations."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("/output/images"))
    return parser.parse_args()


def _normalize_extensions(directory: Path, expected_ext: str) -> None:
    if not directory.exists():
        return
    for path in directory.iterdir():
        if not path.is_file():
            continue
        stem = path.stem
        target = directory / f"{stem}{expected_ext}"
        if path != target:
            path.rename(target)


def main() -> None:
    args = parse_args()
    _normalize_extensions(args.output_dir / "melanoma-cell-detection", ".json")
    _normalize_extensions(args.output_dir / "melanoma-tissue-mask-segmentation", ".tif")


if __name__ == "__main__":
    main()
