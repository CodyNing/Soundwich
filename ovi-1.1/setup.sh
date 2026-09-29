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

if [ ! -d "${UPSTREAM_DIR}/.git" ]; then
  mkdir -p "${ROOT}/third_party"
  git clone "${UPSTREAM_URL}" "${UPSTREAM_DIR}"
fi
if [ "$(git -C "${UPSTREAM_DIR}" rev-parse HEAD)" != "${UPSTREAM_COMMIT}" ]; then
  git -C "${UPSTREAM_DIR}" fetch origin "${UPSTREAM_COMMIT}"
  git -C "${UPSTREAM_DIR}" checkout --detach "${UPSTREAM_COMMIT}"
fi

# Versions below are the environment the paper results were reproduced with (Python 3.11, CUDA 12.8).
PYTHON="${PYTHON:-python3.11}"
if [ ! -d "${ENV_DIR}" ]; then
  "${PYTHON}" -c 'import sys; assert sys.version_info[:2] == (3, 11), "Python 3.11 is required"'
  "${PYTHON}" -m venv "${ENV_DIR}"
fi

"${ENV_DIR}/bin/pip" install --upgrade pip
"${ENV_DIR}/bin/pip" install --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0
"${ENV_DIR}/bin/pip" install -r "${UPSTREAM_DIR}/requirements.txt"
"${ENV_DIR}/bin/pip" install -r "${ROOT}/requirements.txt"

# Upstream Ovi's attention requires flash-attn. It compiles against your CUDA toolkit (CUDA_HOME) when no
# prebuilt wheel matches; set TORCH_CUDA_ARCH_LIST for your GPU to speed up the build.
if ! "${ENV_DIR}/bin/pip" install flash_attn==2.8.3.post1 --no-build-isolation; then
  echo "flash-attn failed to install; Ovi cannot run without it. Install a build matching your CUDA/GPU." >&2
  exit 1
fi

echo "Environment ready: ${ENV_DIR}"
echo "Download the 10-second model checkpoint with:"
echo "  ${ENV_DIR}/bin/python ${UPSTREAM_DIR}/download_weights.py --output-dir ${ROOT}/ckpts --models 960x960_10s"
