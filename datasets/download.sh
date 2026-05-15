#!/usr/bin/env bash
# Download any HuggingFace LeRobot dataset into ./raw/<basename>/.
#
# Usage:
#   bash datasets/download.sh                                   # default: LightwheelAI/leisaac-pick-orange
#   bash datasets/download.sh <ORG>/<DATASET>
#   REPO_ID=foo/bar  bash datasets/download.sh                  # equivalent
#   LOCAL_DIR=/abs/path  bash datasets/download.sh ORG/DATA     # override local dir
#
# Notes:
#   - Resumable: re-running skips already-downloaded files.
#   - raw/<basename>/ is gitignored.
#   - Run datasets/convert_to_v30.sh <basename> afterwards if the dataset is
#     LeRobot v2.1 (lerobot >= 0.5.x no longer reads v2.1 directly).

set -euo pipefail

REPO_ID="${1:-${REPO_ID:-LightwheelAI/leisaac-pick-orange}}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_DIR="${LOCAL_DIR:-${SCRIPT_DIR}/raw/$(basename "${REPO_ID}")}"

mkdir -p "${LOCAL_DIR}"

echo "[download] repo:       ${REPO_ID}"
echo "[download] local dir:  ${LOCAL_DIR}"

if command -v hf >/dev/null 2>&1; then
    hf download "${REPO_ID}" --repo-type dataset --local-dir "${LOCAL_DIR}"
elif command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli download "${REPO_ID}" --repo-type dataset --local-dir "${LOCAL_DIR}"
else
    echo "[download] no 'hf' / 'huggingface-cli' on PATH; falling back to python snapshot_download"
    python - <<PY
from huggingface_hub import snapshot_download
print(snapshot_download(
    repo_id="${REPO_ID}",
    repo_type="dataset",
    local_dir="${LOCAL_DIR}",
))
PY
fi

echo "[download] done: ${LOCAL_DIR}"
