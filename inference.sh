#!/usr/bin/env bash
set -euo pipefail

exec python /opt/app/scripts/infer_wsi.py \
  --input "${SYMBIOPAN_INPUT:-/input/images/melanoma-whole-slide-image}" \
  --output "${SYMBIOPAN_OUTPUT:-/output}" \
  --cp "${SYMBIOPAN_CKPT:-/opt/app/checkpoints/best_model.pth}"
