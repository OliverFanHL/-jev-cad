"""Reproduce the saved automatic silhouette endpoint, never expert ratings.

Extracted from the original proxy computation without the separate synthetic
rater smoke-test machinery. Requires the frozen pairs and rendered CAD views.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]


def visual_signature(asset_id: str, renders: Path) -> tuple[np.ndarray, np.ndarray]:
    views = []
    for suffix in ("", "_B"):
        with Image.open(renders / f"{asset_id}{suffix}.webp") as image:
            gray = np.asarray(image.convert("L").resize((64, 64)), dtype=np.uint8)
        views.append(gray < 240)
    return views[0], views[1]


def visual_score(a, b) -> float:
    ious = []
    for x, y in zip(a, b):
        union = np.logical_or(x, y).sum()
        ious.append(float(np.logical_and(x, y).sum() / union) if union else 0.0)
    return float(np.mean(ious))


def main(pairs: Path, renders: Path, output: Path):
    if output.exists():
        raise FileExistsError(f"Preserve existing proxy scores: {output}")
    selected = pd.read_csv(pairs, dtype={"pair_id": str, "query_id": str, "candidate_id": str, "split": str})
    ids = set(selected["query_id"]) | set(selected["candidate_id"])
    signatures = {sid: visual_signature(sid, renders) for sid in sorted(ids)}
    selected["visual_score"] = [visual_score(signatures[q], signatures[c])
                                for q, c in zip(selected["query_id"], selected["candidate_id"])]
    selected["base_grade"] = 0
    for _, group in selected.groupby(["split", "query_id"], sort=False):
        order = group.sort_values(["visual_score", "pair_id"], ascending=[False, True]).index.to_numpy()
        n = len(order)
        n3, n2, n1 = [max(1, int(round(fraction * n))) for fraction in (.15, .15, .25)]
        selected.loc[order[:n3], "base_grade"] = 3
        selected.loc[order[n3:n3+n2], "base_grade"] = 2
        selected.loc[order[n3+n2:n3+n2+n1], "base_grade"] = 1
    output.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(output, index=False)
    print("Saved automatic visual-proxy scores; no independent expert grades were generated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, default=ROOT / "outputs/study/full/annotations/selected_pairs.csv")
    parser.add_argument("--renders", type=Path, default=ROOT / "apps/expert_review/public/renders")
    parser.add_argument("--output", type=Path, default=ROOT / "recomputed/visual_proxy_scores.csv")
    args = parser.parse_args()
    main(args.pairs, args.renders, args.output)
