"""Finalize image-derived visual-form grades as an automated retrieval endpoint.

This deliberately avoids the synthetic rater streams used in the separate
finalizer smoke test. The grades are visual proxies, not expert judgments.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from jev_tmcad.config import Config  # noqa: E402
from jev_tmcad.study_cli import _features  # noqa: E402
from jev_tmcad.study_ground_truth import file_sha256  # noqa: E402
from jev_tmcad.study_retrieval import evaluate_retrieval  # noqa: E402

SOURCE = ROOT / "outputs/study/full/annotations"
PROXY_SCORES = ROOT / "outputs/study/mock_expert_review/visual_proxy_scores.csv"
OUT = ROOT / "outputs/study/visual_proxy_evaluation"


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Visual-proxy output already exists: {OUT}")
    protocol = json.loads((SOURCE / "protocol.json").read_text(encoding="utf-8"))
    selected_path = SOURCE / "selected_pairs.csv"
    if file_sha256(selected_path) != protocol["selected_pairs_sha256"]:
        raise ValueError("Frozen selected pairs changed")
    selected = pd.read_csv(selected_path, dtype={"pair_id": str, "split": str,
                                                 "query_id": str, "candidate_id": str})
    proxy = pd.read_csv(PROXY_SCORES, dtype={"pair_id": str, "split": str,
                                           "query_id": str, "candidate_id": str})
    keys = ["pair_id", "split", "query_id", "candidate_id"]
    if not selected[keys].equals(proxy[keys]):
        raise ValueError("Visual scores are not aligned with the frozen pair list")
    grades = proxy["base_grade"].astype(int)
    if not grades.isin([0, 1, 2, 3]).all():
        raise ValueError("Invalid visual-proxy grade")
    OUT.mkdir(parents=True)
    judgments = OUT / "visual_proxy_judgments.csv"
    final = selected.drop(columns=["grade"]).copy()
    final["grade"] = grades
    final.to_csv(judgments, index=False)
    meta = {
        "label_source": "AUTOMATED two-view silhouette proxy; no human judgments",
        "visual_proxy": True,
        "selected_pairs_sha256": protocol["selected_pairs_sha256"],
        "selected_query_ids": protocol["selected_query_ids"],
        "judgments_sha256": file_sha256(judgments),
        "n_pairs": len(final),
        "grade_rule": "64x64 silhouette IoU averaged over two rendered views; within-query 15/15/25/45% ordinal bins",
    }
    judgments.with_suffix(".meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    cfg = Config.load(ROOT / "configs/study_full_ft.yaml")
    def config_path(key: str) -> Path:
        raw = Path(cfg.get(key))
        return raw if raw.is_absolute() else cfg.base_dir / raw
    _, features = _features(cfg, config_path)
    methods = evaluate_retrieval(features, judgments, OUT / "retrieval",
                                 ks=[1, 5, 10], weights=[0, .25, .5, .75, 1],
                                 query_scope="preselected")
    summary_path = OUT / "retrieval/summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update({"relevance": "AUTOMATED visual-form proxy; not expert engineering relevance",
                    "visual_proxy": True, "judgments_path": str(judgments)})
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "pairs": len(final),
        "grade_counts": grades.value_counts().sort_index().to_dict(),
        "methods": {name: {key: value[key] for key in
                           ("ndcg@10", "precision@10", "map", "mrr")}
                    for name, value in methods.items()},
    }, indent=2))


if __name__ == "__main__":
    main()
