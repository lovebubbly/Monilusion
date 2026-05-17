#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -n "${PHASE53_PYTHON:-}" ]]; then
  PYTHON="$PHASE53_PYTHON"
elif [[ -x "$ROOT/.venv/bin/python" ]]; then
  PYTHON="$ROOT/.venv/bin/python"
elif [[ -x "$ROOT/venv/bin/python" ]]; then
  PYTHON="$ROOT/venv/bin/python"
else
  PYTHON="$(command -v python3)"
fi

cd "$ROOT"
"$PYTHON" "$ROOT/tools/run_phase53_active_shadow_cycle.py" --rebind-paths "$@"
