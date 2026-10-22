"""Old saved bundles remain usable after moving the source directories."""
import importlib.util
from pathlib import Path


def load_paths():
    source = Path(__file__).resolve().parents[1] / "scripts/artifact_paths.py"
    spec = importlib.util.spec_from_file_location("artifact_paths", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_legacy_bundle_and_relocated_archives_are_both_readable(tmp_path):
    paths = load_paths()
    old = "research_v2/evaluation/classification/summary.json.gz"
    new = "analysis/evaluation/classification/summary.json.gz"
    legacy = tmp_path / old
    legacy.parent.mkdir(parents=True)
    legacy.write_bytes(b"archive")
    assert paths.artifact_path(tmp_path, new) == legacy
    assert paths.current_path(old) == new
    relocated = tmp_path / new
    relocated.parent.mkdir(parents=True)
    legacy.rename(relocated)
    assert paths.artifact_path(tmp_path, old) == relocated
    assert paths.current_path("research_v2/expert_review_app/data/pairs.json") == "apps/expert_review/data/pairs.json"
