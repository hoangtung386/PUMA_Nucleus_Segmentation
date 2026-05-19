#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
TAG="puma-merged-v22-v4-track2"
INPUT_DIR="${SCRIPT_DIR}/test"
OUTPUT_DIR="${SCRIPT_DIR}/output"
mkdir -p "${OUTPUT_DIR}"

docker build "${SCRIPT_DIR}" --platform=linux/amd64 --tag "${TAG}"

docker run --rm \
  --shm-size=8g \
  --memory=32g \
  --platform=linux/amd64 \
  --network none \
  --gpus all \
  -v "${INPUT_DIR}/:/input/images/melanoma-whole-slide-image/" \
  -v "${OUTPUT_DIR}/:/output/" \
  "${TAG}"
