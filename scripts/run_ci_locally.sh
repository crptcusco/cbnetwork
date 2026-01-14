#!/usr/bin/env bash
set -euo pipefail

# Creates a clean virtualenv, installs requirements, and runs pytest with PYTHONPATH=src
VENV_DIR=".tmp_test_env"
python -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
if [ -f requirements.txt ]; then
  pip install -r requirements.txt
fi

export PYTHONPATH=src
pytest -q

deactivate
echo "Local CI run completed."
