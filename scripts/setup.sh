#!/usr/bin/env bash
# Installs only in this project's .venv. No sudo, driver or global package changes.
set -euo pipefail
export PIP_CACHE_DIR="$PWD/.cache/pip"
if [[ ! -f pyproject.toml || ! -f requirements.lock ]]; then
  printf 'Run this script from the project root.\n' >&2
  exit 1
fi
python3 -c 'import sys; assert sys.version_info[:3] == (3,14,4), "Use CPython 3.14.4 or the pinned Docker image"'
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock
.venv/bin/python -m pip install --no-deps --no-build-isolation -e .
make test
