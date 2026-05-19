#!/usr/bin/env bash
set -euo pipefail

STAGE2_ARGS=()
if [ -f /opt/app/checkpoint/nuclei_refiner_residual_best.pth ]; then
  STAGE2_ARGS=(--stage2-cp /opt/app/checkpoint/nuclei_refiner_residual_best.pth)
fi

SITE_ARGS=()
if [ -f /opt/app/checkpoint/site_classifier_atto.pth ]; then
  SITE_ARGS=(--site-classifier-cp /opt/app/checkpoint/site_classifier_atto.pth)
fi

python /opt/app/infer_wsi.py \
  --input /input/images/melanoma-whole-slide-image \
  --output /output \
  --cp /opt/app/checkpoint/best_model.pth \
  --cellpose-mode auto \
  "${STAGE2_ARGS[@]}" \
  "${SITE_ARGS[@]}"
