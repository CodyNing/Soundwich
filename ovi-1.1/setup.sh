#!/usr/bin/env bash
# Clone upstream Ovi at the pinned commit and install its dependencies plus
# the extras this package needs. Idempotent: re-running reuses the existing
# checkout and virtualenv.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UPSTREAM_URL="https://github.com/character-ai/Ovi.git"
UPSTREAM_COMMIT="5b69b25a4b3115216e9ea53a37a04410be6ad39a"
UPSTREAM_DIR="${ROOT}/third_party/Ovi"
ENV_DIR="${ROOT}/.venv"

if [ -d "${UPSTREAM_DIR}/.git" ]; then
  echo "third_party/Ovi already present, skipping clone"
else
  mkdir -p "${ROOT}/third_party"
  git clone "${UPSTREAM_URL}" "${UPSTREAM_DIR}"
  git -C "${UPSTREAM_DIR}" checkout "${UPSTREAM_COMMIT}"
fi

if [ ! -d "${ENV_DIR}" ]; then
  python3 -m venv "${ENV_DIR}"
fi

"${ENV_DIR}/bin/pip" install --upgrade pip
"${ENV_DIR}/bin/pip" install -r "${UPSTREAM_DIR}/requirements.txt"
"${ENV_DIR}/bin/pip" install -r "${ROOT}/requirements.txt"

# flash-attn is required by upstream ovi/modules/attention.py and needs a
# CUDA toolchain matching your GPU; install it separately if the generic
# wheel below does not match your environment (see flash-attn's own
# installation docs for CUDA-version-specific builds).
"${ENV_DIR}/bin/pip" install flash-attn --no-build-isolation || \
  echo "flash-attn install failed; install a build matching your CUDA/GPU manually"

echo "Environment ready: ${ENV_DIR}"
echo "Download the 10-second model checkpoint with:"
echo "  ${ENV_DIR}/bin/python ${UPSTREAM_DIR}/download_weights.py --output-dir ${ROOT}/ckpts --models 960x960_10s"
