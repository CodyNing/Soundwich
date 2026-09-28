#!/usr/bin/env bash
# Clone LTX-2 at the pinned commit into third_party/, apply the Soundwich patch,
# and install everything into third_party/LTX-2/.venv. Safe to re-run.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LTX_REPO="https://github.com/Lightricks/LTX-2.git"
LTX_COMMIT="400fd31054597515f47125691032c04b1c3ee24e"
LTX_DIR="$HERE/third_party/LTX-2"
PATCH="$HERE/patches/ltx-2.patch"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required (https://docs.astral.sh/uv/getting-started/installation/)" >&2
    exit 1
fi

if [ ! -d "$LTX_DIR/.git" ]; then
    mkdir -p "$HERE/third_party"
    git clone "$LTX_REPO" "$LTX_DIR"
fi

if [ "$(git -C "$LTX_DIR" rev-parse HEAD)" != "$LTX_COMMIT" ]; then
    git -C "$LTX_DIR" cat-file -e "$LTX_COMMIT^{commit}" 2>/dev/null || git -C "$LTX_DIR" fetch origin
    git -C "$LTX_DIR" checkout --quiet "$LTX_COMMIT"
fi

if git -C "$LTX_DIR" apply --reverse --check "$PATCH" >/dev/null 2>&1; then
    echo "patches/ltx-2.patch is already applied"
else
    git -C "$LTX_DIR" apply "$PATCH"
    echo "applied patches/ltx-2.patch"
fi

(cd "$LTX_DIR" && uv sync --extra natten)

echo
echo "Done. Run from $HERE with:"
echo "  $LTX_DIR/.venv/bin/python -m soundwich_ltx.generate --scene examples/neon_biology_lab.yaml"
