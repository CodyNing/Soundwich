"""Soundwich multi-stem method for Ovi 1.1.

Runs one frozen Ovi checkpoint to generate one shared video and several
independently controlled, timeline-gated audio stems. See
``python -m soundwich_ovi.generate --help``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PINNED_UPSTREAM_COMMIT = "5b69b25a4b3115216e9ea53a37a04410be6ad39a"


def _find_upstream_repo_root() -> Path | None:
    """Return the upstream Ovi checkout root, if importable or locatable."""
    try:
        import ovi

        return Path(ovi.__file__).resolve().parent.parent
    except ImportError:
        pass

    candidates = []
    env_repo = os.environ.get("SOUNDWICH_OVI_REPO")
    if env_repo:
        candidates.append(Path(env_repo))
    candidates.append(Path(__file__).resolve().parents[1] / "third_party" / "Ovi")

    for candidate in candidates:
        if (candidate / "ovi" / "__init__.py").is_file():
            return candidate
    return None


def _ensure_upstream_importable() -> Path:
    """Make the upstream ``ovi`` package importable.

    ``setup.sh`` clones upstream Ovi into ``third_party/Ovi`` (gitignored)
    next to this package. If ``ovi`` is not already on ``sys.path`` (for
    example because the caller set PYTHONPATH or installed it directly), add
    that checkout.

    ``ovi.ovi_fusion_engine`` loads a default config from a path relative to
    the upstream repo root as a module-level side effect, so this also
    chdirs there for the duration of that one import. Returns the repo root.
    """
    repo_root = _find_upstream_repo_root()
    if repo_root is None:
        raise ImportError(
            "Cannot import the upstream 'ovi' package. Run ./setup.sh to "
            f"clone it into third_party/Ovi at commit {PINNED_UPSTREAM_COMMIT}, "
            "or set SOUNDWICH_OVI_REPO (or PYTHONPATH) to an existing checkout."
        )
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    previous_cwd = Path.cwd()
    try:
        os.chdir(repo_root)
        import ovi.ovi_fusion_engine  # noqa: F401
    finally:
        os.chdir(previous_cwd)
    return repo_root


# Upstream Ovi also loads its model configs relative to this directory.
UPSTREAM_REPO_ROOT = _ensure_upstream_importable()

# Attaches the multi-stem forward path to ovi.modules.fusion.FusionModel.
from . import fusion as _fusion  # noqa: E402,F401

__all__ = ["UPSTREAM_REPO_ROOT"]
