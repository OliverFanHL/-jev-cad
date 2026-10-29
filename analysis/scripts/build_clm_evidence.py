"""Build CLM comparison tables from saved evaluator predictions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
CLM_METHODS = (
    "clm_direct_choice", "clm_choice_probabilities", "clm_geometry_questions",
    "clm_all_questions", "geometry_plus_clm_choice_probabilities",
    "geometry_plus_clm_geometry_questions", "geometry_plus_clm", "descriptor_plus_clm",
)
REFERENCE_METHODS = ("descriptor_plus_geometry", "jev_direct_choice",
                     "jev_all_questions", "geometry_plus_jev", "laya_ft_direct_choice")


def _predictions(directory: Path, name: str, test_ids: set[str]) -> pd.DataFrame:
    frame = pd.read_csv(directory / f"{name}.csv", dtype={"sample_id": str})
    if len(frame) != len(test_ids) or frame["sample_id"].duplicated().any() or set(frame["sample_id"]) != test_ids:
        raise ValueError(f"{name} predictions do not cover the frozen test IDs exactly")
    if not frame["split"].eq("test").all():
        raise ValueError(f"{name} includes non-test predictions")
    return frame.set_index("sample_id").sort_index()


def build(clm_eval: Path, baseline_eval: Path, output: Path,
          *, cohort_path: Path = ROOT / "outputs/study/full/cohort.csv",
          audit_path: Path = ROOT / "outputs/study/full/evaluation/duplicate_audit.json",
          cases_path: Path = ROOT / "analysis/evidence/qualitative_examples.json") -> None:
    clm = clm_eval / "classification"
    base = baseline_eval / "classification"
    summary = json.loads((clm / "summary.json").read_text(encoding="utf-8"))["methods"]
    base_summary = json.loads((base / "summary.json").read_text(encoding="utf-8"))["methods"]
    if not set(CLM_METHODS).issubset(summary):
        raise ValueError("Aggregate CLM classification output is incomplete")
    source_manifest = base / "source_manifest.csv"
    cohort = pd.read_csv(source_manifest if source_manifest.exists() else cohort_path, dtype={"sample_id": str})
    if cohort["sample_id"].duplicated().any():
        raise ValueError("Frozen cohort has duplicate sample IDs")
    test_ids = set(cohort.loc[cohort["split"] == "test", "sample_id"])
    predictions = {name: _predictions(clm, name, test_ids) for name in CLM_METHODS}
    predictions.update({name: _predictions(base, name, test_ids) for name in REFERENCE_METHODS})
    for name in REFERENCE_METHODS:
        if (clm / f"{name}.csv").exists() and not _predictions(clm, name, test_ids)[
                ["label", "prediction"]].equals(predictions[name][["label", "prediction"]]):
            raise ValueError(f"{name} differs from the matched baseline")
    y = predictions["descriptor_plus_geometry"]["label"]
    if any(not p["label"].equals(y) for p in predictions.values()):
        raise ValueError("CLM predictions differ in test labels")
    output.mkdir(parents=True, exist_ok=True)
    fields = ("n", "accuracy", "macro_f1", "balanced_accuracy", "nll", "brier")
    combined = {**base_summary, **{name: summary[name] for name in CLM_METHODS}}
    pd.DataFrame([{"method": name, **{key: item[key] for key in fields}}
                  for name, item in combined.items()]).to_csv(output / "matched_classification.csv", index=False)
    main = ("descriptor_plus_geometry", "jev_direct_choice", "jev_all_questions",
            "geometry_plus_jev", "laya_ft_direct_choice", *CLM_METHODS)
    per_class = []
    for label in sorted(y.unique()):
        mask = y.eq(label)
        per_class.append({"class": label, "n": int(mask.sum()),
                          **{name: float(predictions[name].loc[mask, "prediction"].eq(label).mean())
                             for name in main}})
    pd.DataFrame(per_class).to_csv(output / "class_recall.csv", index=False)
    pairs = []
    for a, b in (("clm_direct_choice", "jev_direct_choice"),
                 ("clm_all_questions", "jev_all_questions"),
                 ("geometry_plus_clm", "geometry_plus_jev"),
                 ("geometry_plus_clm", "descriptor_plus_geometry"),
                 ("clm_direct_choice", "laya_ft_direct_choice")):
        ac = predictions[a]["prediction"].eq(y)
        bc = predictions[b]["prediction"].eq(y)
        pairs.append({"a": a, "b": b, "both_correct": int((ac & bc).sum()),
                      "a_only": int((ac & ~bc).sum()), "b_only": int((~ac & bc).sum()),
                      "both_wrong": int((~ac & ~bc).sum())})
    (output / "paired_outcomes.json").write_text(json.dumps(pairs, indent=2), encoding="utf-8")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    compromised = {str(sample["sample_id"])
                   for group in audit["groups"]
                   if "test" in {s["split"] for s in group["samples"]}
                   and {s["split"] for s in group["samples"]} & {"train", "val"}
                   for sample in group["samples"] if sample["split"] == "test"}
    clean_ids = y.index.difference(list(compromised))
    sensitivity = {"test_n": len(y), "cross_split_duplicate_test_n": len(compromised),
                   "remaining_n": len(clean_ids),
                   "methods": {name: float(predictions[name].loc[clean_ids, "prediction"].eq(y.loc[clean_ids]).mean())
                               for name in (*CLM_METHODS, *REFERENCE_METHODS)}}
    (output / "duplicate_sensitivity.json").write_text(json.dumps(sensitivity, indent=2), encoding="utf-8")
    if cases_path.exists():
        cases = json.loads(cases_path.read_text(encoding="utf-8"))
        case_ids = [str(case["sample_id"]) for case in cases]
        if not set(case_ids).issubset(test_ids):
            raise ValueError("Existing qualitative examples include non-test IDs")
        pd.DataFrame([{"sample_id": sid, "label": y.at[sid],
                       **{name: predictions[name].at[sid, "prediction"] for name in CLM_METHODS}}
                      for sid in case_ids]).to_csv(output / "qualitative_case_predictions.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clm-eval", type=Path, default=ROOT / "outputs/study/clm/evaluation")
    parser.add_argument("--baseline-eval", type=Path, default=ROOT / "analysis/evaluation")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/study/clm/evidence")
    args = parser.parse_args()
    build(args.clm_eval, args.baseline_eval, args.output)
