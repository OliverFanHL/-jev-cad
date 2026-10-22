"""Load the completed imported CLM evaluation for manuscript postprocessing."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMPORT = ROOT.parent / "outputs/study/clm/kaggle_import"
CLM_METHODS = (
    "clm_direct_choice", "clm_choice_probabilities", "clm_geometry_questions",
    "clm_all_questions", "geometry_plus_clm_choice_probabilities",
    "geometry_plus_clm_geometry_questions", "geometry_plus_clm", "descriptor_plus_clm",
)


def load_matched_summaries() -> tuple[dict, dict]:
    manifest = json.loads((IMPORT / "output_manifest.json").read_text(encoding="utf-8"))
    if (manifest["completed"], manifest["pending"]) != (9798, 0):
        raise ValueError("A completed 9,798-part CLM import is required")
    base = json.loads((ROOT / "evaluation/classification/summary.json").read_text(encoding="utf-8"))
    clm = json.loads((IMPORT / "evaluation/classification/summary.json").read_text(encoding="utf-8"))
    if base["cohort_hash"] != clm["cohort_hash"]:
        raise ValueError("CLM and manuscript use different frozen cohorts")
    if not set(CLM_METHODS).issubset(clm["methods"]):
        raise ValueError("The imported CLM classification summary is incomplete")
    if any(clm["methods"][name]["n"] != 1960 for name in CLM_METHODS):
        raise ValueError("CLM metrics do not use the complete 1,960-part test set")
    return base["methods"], clm["methods"]
