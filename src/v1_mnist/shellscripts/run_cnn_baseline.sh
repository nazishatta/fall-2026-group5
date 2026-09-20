#!/bin/bash
# Week 2: MNIST -> LeNet-5 -> fc2 embeddings

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

CONFIG="src/component/configs/week_2_baseline.yaml"
CHECKPOINT="outputs/week_2/checkpoints/lenet5_best.pth"

echo "Running Week 2 baseline..."

python src/component/pipeline/01_prepare_data.py --config "$CONFIG"
python src/component/pipeline/02_train_baseline.py --config "$CONFIG"
python src/component/pipeline/03_extract_embeddings.py     --config "$CONFIG"     --checkpoint "$CHECKPOINT"

echo ""
echo "Week 2 complete:"
echo "- MNIST prepared with 70/15/15 split"
echo "- LeNet-5 trained and evaluated"
echo "- 84-D fc2 embeddings saved"
echo "- Outputs: outputs/week_2/"
