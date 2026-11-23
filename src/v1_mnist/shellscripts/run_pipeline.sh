#!/usr/bin/env bash
# Run the MNIST pipeline: the shared step runners in proposal order, with the settings in
# src/v1_mnist/component/configs/ (base.yaml names the dataset and the output folder).
#
# Usage, from anywhere:
#   bash src/v1_mnist/shellscripts/run_pipeline.sh [run_all options]
# Examples:
#   bash src/v1_mnist/shellscripts/run_pipeline.sh --dry-run
#   bash src/v1_mnist/shellscripts/run_pipeline.sh --from som_nnsom_training --to error_pattern_analysis
#
# MNIST's Phase 1 results are frozen, so the default starts at the SOM; pass --from data_pipeline
# only for a new run ID.
# Every option is passed on to run_all.py (see its --help). The Python of the active
# environment is used; activate it first (for example: source ~/dl-venv/bin/activate).
set -euo pipefail

CODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"   # folder holding src/ and outputs/
cd "$CODE_ROOT"

# Start at the default step unless a --from is given.
case " $* " in
  *" --from "*|*" --from="*) ;;
  *) set -- --from som_nnsom_training "$@" ;;
esac
exec python src/v1_mnist/component/pipeline/run_all.py "$@"
