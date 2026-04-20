#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${1:-puma-algorithm:latest}"
ARCHIVE_NAME="${2:-puma-algorithm.tar.gz}"

tmp_tar="$(mktemp -u /tmp/puma-algo-XXXXXX.tar)"
docker save "${IMAGE_NAME}" -o "${tmp_tar}"
gzip -c "${tmp_tar}" > "${ARCHIVE_NAME}"
rm -f "${tmp_tar}"

echo "Saved ${IMAGE_NAME} to ${ARCHIVE_NAME}"
