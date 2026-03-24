#!/bin/bash
# Upload demo images to HPC (they are excluded from git by .gitignore).
#
# Usage (from project root):
#   bash data_preparation/demo_images/upload_demo_images.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REMOTE="tufts-login"
REMOTE_DIR="/cluster/tufts/c26sp1ee0141/pliu07/vis_inject/demos/demo_images"

echo "Uploading demo images to HPC..."
echo "  Local:  ${PROJECT_ROOT}/demos/demo_images/"
echo "  Remote: ${REMOTE}:${REMOTE_DIR}"
echo ""

ssh "${REMOTE}" "mkdir -p ${REMOTE_DIR}"
scp "${PROJECT_ROOT}"/demos/demo_images/ORIGIN_*.png "${REMOTE}:${REMOTE_DIR}/"

echo ""
echo "Done. Uploaded files:"
ssh "${REMOTE}" "ls -lh ${REMOTE_DIR}/ORIGIN_*.png"
