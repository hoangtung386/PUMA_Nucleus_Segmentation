#!/usr/bin/env bash
set -euo pipefail

STAGE2_ARGS=()
if [ -f /opt/app/checkpoints/nuclei_refiner_residual_best.pth ]; then
  STAGE2_ARGS=(--stage2-cp /opt/app/checkpoints/nuclei_refiner_residual_best.pth)
fi

SITE_ARGS=()
if [ -f /opt/app/checkpoints/site_classifier_atto.pth ]; then
  SITE_ARGS=(--site-classifier-cp /opt/app/checkpoints/site_classifier_atto.pth)
fi

python /opt/app/scripts/run_inference.py \
  --input /input/images/melanoma-whole-slide-image \
  --output /output \
  --cp /opt/app/checkpoints/best_model.pth \
  --cellpose-mode auto \
  "${STAGE2_ARGS[@]}" \
  "${SITE_ARGS[@]}"
