"""Import code from a third-party repo whose top-level package is also called `models`.

CED (`models.audiotransformer`) and EfficientAT (`models.mn.model`) both import their
own files as `models.<x>`, which collides with this project's `models` package. The
collision is silent - an import resolves to whichever `models` came first. So the
repo is put first on sys.path with this project's `models.*` entries lifted out of
sys.modules, the requested modules are imported, and everything is put back. The
imported modules keep references to their own submodules, so they keep working
after their `models` package is gone from sys.modules again.

The repo is cloned at a pinned commit on first use, so a fresh molab sandbox needs
nothing beyond network access.
"""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path

_CLASHING = ("models", "helpers")


def ensure_repo(repo_dir: str, url: str, commit: str) -> Path:
    path = Path(repo_dir)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "-q", url, str(path)], check=True)
        subprocess.run(["git", "-C", str(path), "checkout", "-q", commit], check=True)
        print(f"cloned {url} @ {commit} -> {path}", flush=True)
    return path


def _owned(name: str) -> bool:
    return any(name == root or name.startswith(root + ".") for root in _CLASHING)


def _shadows(entry: str) -> bool:
    """Does this sys.path entry hold a `models`/`helpers` of its own (ours, say)?"""
    root = Path(entry or ".")
    return any((root / name).is_dir() for name in _CLASHING)


def import_isolated(repo_dir: str, url: str, commit: str, *modules: str) -> list:
    """Import `modules` from the repo at `repo_dir` without touching ours.

    Putting the repo first on sys.path is not enough: EfficientAT's `models` has no
    __init__.py, and Python prefers a regular package anywhere on the path (ours)
    over a namespace package earlier on it. So every entry that carries its own
    `models` or `helpers` is left off the path for the duration of the import.
    The import also runs from the repo's root: EfficientAT's helpers open
    `metadata/...` relative to the working directory at import time.
    """
    path = str(ensure_repo(repo_dir, url, commit))
    ours = {name: sys.modules.pop(name) for name in list(sys.modules) if _owned(name)}
    saved_path, saved_cwd = list(sys.path), os.getcwd()
    sys.path[:] = [path] + [entry for entry in saved_path
                            if entry and not _shadows(entry)]
    importlib.invalidate_caches()
    os.chdir(path)
    try:
        return [importlib.import_module(name) for name in modules]
    finally:
        os.chdir(saved_cwd)
        sys.path[:] = saved_path
        for name in [name for name in sys.modules if _owned(name)]:
            del sys.modules[name]
        sys.modules.update(ours)
        importlib.invalidate_caches()
