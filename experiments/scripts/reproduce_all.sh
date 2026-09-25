#!/bin/bash
# End-to-end paper reproduction: train every model, then run severity evaluation.

set -e

echo "=== Step 1/2: Train all models ==="
bash experiments/scripts/train_models.sh

echo "=== Step 2/2: Severity evaluation ==="
bash experiments/scripts/run_severity.sh

echo "=== Reproduction complete ==="
