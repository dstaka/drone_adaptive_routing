#!/usr/bin/env bash
set -euo pipefail
# Launch this script inside a unique tmux session from the project root.
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
exec .venv-flight/bin/python -m adaptive_drone_patrol.jobs "$@"
