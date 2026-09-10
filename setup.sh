#!/usr/bin/env bash
# One-time setup on a new machine.
#
# Creates the small conda environment the web UI and scheduler run in, then
# checks that everything the pipeline needs is actually present. The pipeline
# itself keeps using the existing whisperx / pyannote environments - this
# repository wraps those scripts, it does not contain them.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

ENV_NAME="${ASRBENCH_ENV:-asrbench}"

# ── configuration ────────────────────────────────────────────────────
if [ ! -f config.yaml ]; then
  echo "No config.yaml yet - creating one from config.example.yaml"
  cp config.example.yaml config.yaml
  echo
  echo "  Edit config.yaml before going further. The paths in it point at the"
  echo "  machine this was developed on and will not be right here."
  echo
  NEEDS_EDIT=1
else
  NEEDS_EDIT=0
fi

CONDA="$(awk '/^[[:space:]]+conda:/ {print $2; exit}' config.yaml)"
if [ ! -x "$CONDA" ]; then
  echo "conda not found at '$CONDA' (paths.conda in config.yaml)."
  for c in "$HOME/miniconda3/bin/conda" "$HOME/anaconda3/bin/conda" \
           "$HOME/miniforge3/bin/conda" "$(command -v conda || true)"; do
    if [ -x "$c" ]; then echo "  Found one at: $c"; break; fi
  done
  exit 1
fi
echo "Using conda: $CONDA"

# ── environment ──────────────────────────────────────────────────────
if "$CONDA" env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Environment '$ENV_NAME' already exists."
else
  echo "Creating environment '$ENV_NAME' (python 3.11)..."
  "$CONDA" create -y -n "$ENV_NAME" python=3.11
fi

echo "Installing requirements..."
"$CONDA" run -n "$ENV_NAME" --no-capture-output pip install -q -r requirements.txt

# ── directories and database ─────────────────────────────────────────
mkdir -p logs run locks
RUNS="$(awk '/^[[:space:]]+runs_root:/ {print $2; exit}' config.yaml)"
mkdir -p "$RUNS" 2>/dev/null || echo "  (could not create $RUNS - check runs_root)"
"$CONDA" run -n "$ENV_NAME" python -c "from asrbench import db; db.init()" \
  && echo "Database ready."

# ── preflight ────────────────────────────────────────────────────────
echo
if [ "$NEEDS_EDIT" = "1" ]; then
  echo "Next: edit config.yaml, then run"
  echo "    $CONDA run -n $ENV_NAME python -m asrbench.doctor --suggest"
  echo "    bin/doctor.sh"
  exit 0
fi

echo "Running preflight check..."
echo
if "$CONDA" run -n "$ENV_NAME" --no-capture-output python -m asrbench.doctor; then
  echo
  echo "Setup complete. Start with:  bin/start.sh"
else
  echo
  echo "Setup finished, but the preflight found problems. Fix them, then run"
  echo "bin/doctor.sh again."
  exit 1
fi
