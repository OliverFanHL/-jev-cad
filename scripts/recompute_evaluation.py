"""Optionally rerun registered evaluators on saved features and responses.

The model identities are checked from saved metadata and request payloads.
This script does not authenticate/download weights, make API calls, retrain an
encoder, or claim to verify unavailable large checkpoint bytes.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main(task: str, experiment: str | None):
    import pandas as pd
    import yaml
    from jev_tmcad.evaluation.classification import prepare_features, evaluate_classification
    from jev_tmcad.evaluation.retrieval import evaluate_retrieval
    from verify_results import verify

    os.chdir(ROOT)
    verify()
    provenance = json.loads(Path("ARTIFACT_PROVENANCE.json").read_text())
    geometry_digest = provenance["undistributed_surface_checkpoint"]["sha256"]
    cohort = pd.read_csv("outputs/study/full/cohort.csv", dtype={"sample_id": str})
    questions = yaml.safe_load(Path("configs/questions_compact.yaml").read_text(encoding="utf-8"))
    paths = {"jev": Path("outputs/study/full/jev_results.jsonl"),
             "laya": Path("outputs/study/full/laya_results.jsonl"),
             "laya_ft": Path("kaggle/laya/jev_tmcad_laya_ft_result/laya_results.jsonl"),
             "clm": Path("outputs/study/clm/kaggle_import/clm_results.jsonl")}
    def first(name):
        with paths[name].open(encoding="utf-8") as h:
            return json.loads(next(h))
    extra = {}
    if task == "classification":
        ft_meta = json.loads(Path("kaggle/laya/jev_tmcad_laya_ft_result/laya_checkpoint/study_metadata.json").read_text())
        ft_model = "laya-ft:" + ft_meta["checkpoint_sha256"]
        if first("laya_ft")["model"] != ft_model:
            raise ValueError("Adapted response model differs from recorded selected checkpoint identity")
        clm_meta = json.loads(Path("outputs/study/clm/kaggle_import/clm_results.meta.json").read_text())
        extra = {"laya_ft_path": paths["laya_ft"], "laya_ft_model": ft_model,
                 "clm_path": paths["clm"], "clm_model": clm_meta["model"],
                 "clm_checkpoint_sha256": clm_meta["checkpoint_sha256"]}
    features = prepare_features(
        cohort, Path("outputs/descriptors/tmcad_descriptors.csv"),
        Path("outputs/study/full/geometry_embeddings.csv"), paths["jev"], paths["laya"],
        Path("outputs/descriptors/quantile_schema.json"), questions,
        jev_model="jev-latest", laya_model=first("laya")["model"],
        include_numeric=True, include_profiles=False, require_laya=True,
        state_variant="compact", recorded_geometry_checkpoint_sha256=geometry_digest, **extra)
    if task == "classification":
        evaluate_classification(features, ROOT / "recomputed/classification",
                                c_grid=[0.01, 0.1, 1.0, 10.0], only=experiment)
    else:
        output = ROOT / "recomputed/visual_proxy_retrieval"
        evaluate_retrieval(features, Path("outputs/study/visual_proxy_evaluation/visual_proxy_judgments.csv"),
                           output, ks=[1, 5, 10], weights=[0, .25, .5, .75, 1],
                           query_scope="preselected", only=experiment)
        summary_path = output / "summary.json"
        summary = json.loads(summary_path.read_text())
        summary.update({"visual_proxy": True,
                        "relevance": "AUTOMATED visual-form proxy; not expert engineering relevance"})
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["classification", "visual-proxy-retrieval"], required=True)
    parser.add_argument("--experiment", help="Optional exact registered method ID, e.g. descriptor_only")
    args = parser.parse_args()
    main(args.task, args.experiment)
