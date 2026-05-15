#!/usr/bin/env bash
# Convert a downloaded LeRobot v2.1 dataset to v3.0 (in-place; v2.1 backed up
# to <basename>_old). lerobot >= 0.5.x rejects v2.1.
#
# Usage:
#   bash datasets/convert_to_v30.sh                                # default: leisaac-pick-orange under raw/
#   bash datasets/convert_to_v30.sh <BASENAME>                     # e.g. some-other-dataset
#   bash datasets/convert_to_v30.sh <ORG>/<DATASET>                # repo-id form (basename is auto-extracted)
#
# Idempotent: if dataset is already v3.0 the script no-ops (unless FORCE=1).

set -euo pipefail

ARG="${1:-LightwheelAI/leisaac-pick-orange}"
# Accept either bare basename or full repo-id.
case "${ARG}" in
    */*) REPO_ID="${ARG}"; BASENAME="$(basename "${ARG}")" ;;
    *)   REPO_ID="LightwheelAI/${ARG}"; BASENAME="${ARG}" ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-${SCRIPT_DIR}/raw/${BASENAME}}"

if [[ ! -f "${ROOT}/meta/info.json" ]]; then
    echo "[convert] dataset not found at ${ROOT}" >&2
    echo "[convert] hint: bash datasets/download.sh ${REPO_ID}" >&2
    exit 1
fi

CURRENT_VERSION="$(python -c "import json,sys; print(json.load(open(sys.argv[1])).get('codebase_version',''))" "${ROOT}/meta/info.json")"
echo "[convert] dataset:        ${ROOT}"
echo "[convert] current codebase_version: ${CURRENT_VERSION}"

if [[ "${CURRENT_VERSION}" == "v3.0" && "${FORCE:-0}" != "1" ]]; then
    echo "[convert] already v3.0, skipping (FORCE=1 to override)"
    exit 0
fi

CONDA_ENV="${CONDA_ENV:-lerobot}"
conda run -n "${CONDA_ENV}" --no-capture-output python -m lerobot.scripts.convert_dataset_v21_to_v30 \
    --repo-id="${REPO_ID}" \
    --root="${ROOT}" \
    --push-to-hub=False \
    --force-conversion

echo "[convert] done; v2.1 backup at ${ROOT}_old"
