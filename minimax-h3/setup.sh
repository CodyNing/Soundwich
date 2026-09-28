#!/usr/bin/env bash
# Create .venv with the pinned diffusers MiniMax-H3 implementation and inference dependencies.
# Idempotent. Optionally set PYTHON (default python3, >= 3.10).
set -euo pipefail
cd "$(dirname "$0")"
DIFFUSERS_COMMIT=8b3c707ebd3ec4881f4190cf42931da07eaf3b65
if [[ ! -d third_party/diffusers/.git ]]; then
  git clone https://github.com/huggingface/diffusers.git third_party/diffusers
fi
if [[ "$(git -C third_party/diffusers rev-parse HEAD)" != "$DIFFUSERS_COMMIT" ]]; then
  git -C third_party/diffusers fetch origin "$DIFFUSERS_COMMIT"
  git -C third_party/diffusers checkout --detach "$DIFFUSERS_COMMIT"
fi
if [[ ! -x .venv/bin/python ]]; then
  "${PYTHON:-python3}" -m venv .venv
fi
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r requirements.txt
.venv/bin/python -m pip install -e third_party/diffusers
.venv/bin/python -c "import diffusers, soundwich_h3.batch; print('diffusers', diffusers.__version__, 'OK')"
