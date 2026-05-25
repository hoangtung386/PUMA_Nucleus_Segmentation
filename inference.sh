#!/usr/bin/env bash
set -euo pipefail

exec python /opt/app/scripts/run_inference.py \
  --input /input/images/melanoma-whole-slide-image \
  --output /output \
  --cp /opt/app/checkpoints/best_model.pth
