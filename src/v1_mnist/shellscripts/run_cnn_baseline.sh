#!/bin/bash
# v1_mnist: MNIST -> LeNet-5 -> fc2 embeddings

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$PROJECT_ROOT"

CONFIG="src/v1_mnist/component/configs/cnn_baseline.yaml"
CHECKPOINT="outputs/v1_mnist/cnn_baseline/checkpoints/lenet5_best.pth"

echo "Running v1_mnist CNN baseline..."

python src/v1_mnist/component/pipeline/01_prepare_data.py --config "$CONFIG"
python src/v1_mnist/component/pipeline/02_train_baseline.py --config "$CONFIG"
python src/v1_mnist/component/pipeline/03_extract_embeddings.py \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT"

echo ""
echo "v1_mnist CNN baseline complete:"
echo "- MNIST prepared with 70/15/15 split"
echo "- LeNet-5 trained and evaluated"
echo "- 84-D fc2 embeddings saved"
echo "- Outputs: outputs/v1_mnist/cnn_baseline/"
