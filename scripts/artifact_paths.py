"""Locate saved artifacts across the current and earlier repository layouts."""
from pathlib import Path


DIRECTORY_MOVES = (
    ("research_v2/expert_review_app/", "apps/expert_review/"),
    ("research_v2/", "analysis/"),
    ("kaggle_laya/", "kaggle/laya/"),
    ("kaggle_clm/", "kaggle/clm/"),
)


def current_path(name: str) -> str:
    for old, new in DIRECTORY_MOVES:
        if name.startswith(old):
            return new + name[len(old):]
    return name


def artifact_path(root: Path, name: str) -> Path:
    """Find an archive without renaming it or changing its recorded hash."""
    candidates = [name, current_path(name)]
    for old, new in DIRECTORY_MOVES:
        if name.startswith(new):
            candidates.append(old + name[len(new):])
    for relative in candidates:
        path = root / relative
        if path.is_file():
            return path
    return root / name
