#!/usr/bin/env bash
set -euo pipefail

python scripts/challenge_inference.py \
  --input-dir /input/images \
  --output-dir /output/images \
  --track "${PUMA_TRACK:-1}" \
  --seg-model "${PUMA_SEG_MODEL:-cyto3}"

python scripts/output_rename.py --output-dir /output/images
