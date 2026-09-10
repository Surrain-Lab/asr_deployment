#!/usr/bin/env bash
# Check that this machine can actually run the pipeline.
# Pass --suggest to print detected paths for config.yaml instead of checking.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
ENV_NAME="${ASRBENCH_ENV:-asrbench}"
CONDA="$(awk '/^[[:space:]]+conda:/ {print $2; exit}' config.yaml 2>/dev/null || true)"
if [ ! -x "${CONDA:-}" ]; then
  CONDA="$(command -v conda || echo "$HOME/miniconda3/bin/conda")"
fi
exec "$CONDA" run -n "$ENV_NAME" --no-capture-output python -m asrbench.doctor "$@"
