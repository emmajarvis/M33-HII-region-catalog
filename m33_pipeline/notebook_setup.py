from __future__ import annotations

import os
import sys
from pathlib import Path


def repo_root_from_notebook(start: str | os.PathLike[str] = ".") -> Path:
    start_path = Path(start).resolve()
    for candidate in (start_path, *start_path.parents):
        if (candidate / "m33_pipeline").is_dir():
            return candidate
    raise RuntimeError(f"Could not locate repo root from: {start_path}")


def prepare_notebook(start: str | os.PathLike[str] = ".") -> Path:
    root = repo_root_from_notebook(start)
    root_str = str(root)
    if sys.path[0] != root_str:
        try:
            sys.path.remove(root_str)
        except ValueError:
            pass
        sys.path.insert(0, root_str)
    os.chdir(root)
    return root
