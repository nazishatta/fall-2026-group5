#!/usr/bin/env bash
# Run the Fashion-MNIST pipeline: the shared step runners in proposal order, with the settings in
# src/v1_fashion_mnist/component/configs/ (base.yaml names the dataset and the output folder).
#
# Usage, from anywhere:
#   bash src/v1_fashion_mnist/shellscripts/run_pipeline.sh [run_all options]
# Examples:
#   bash src/v1_fashion_mnist/shellscripts/run_pipeline.sh --dry-run
#   bash src/v1_fashion_mnist/shellscripts/run_pipeline.sh --from data_pipeline --to error_pattern_analysis
#
# The default starts at the data pipeline; once Phase 1 has run, its runners refuse to
# overwrite it, so pass --from som_nnsom_training to continue from the SOM.
# Every option is passed on to run_all.py (see its --help). The Python of the active
# environment is used; activate it first (for example: source ~/dl-venv/bin/activate).
set -euo pipefail

CODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"   # folder holding src/ and outputs/
cd "$CODE_ROOT"

# Start at the default step unless a --from is given.
case " $* " in
  *" --from "*|*" --from="*) ;;
  *) set -- --from data_pipeline "$@" ;;
esac
exec python src/v1_fashion_mnist/component/pipeline/run_all.py "$@"
