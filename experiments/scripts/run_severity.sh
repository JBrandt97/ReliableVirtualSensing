#!/bin/bash
# Run the severity sweep for every (model, dataset) pair in severity.yaml.
# Writes per-severity CSVs and summary tables under severity_results/.
#
# Usage:  bash experiments/scripts/run_severity.sh [output_dir]

set -e

OUT_DIR="${1:-severity_results}"

python -m experiments.main severity \
    --config experiments/configs/severity.yaml \
    --output_dir "${OUT_DIR}"

echo "Severity evaluation written to: ${OUT_DIR}"
