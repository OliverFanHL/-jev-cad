"""Exact-file identities and source-label consistency."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from jev_tmcad.utils import write_json


def exclude_label_conflicts(frame: pd.DataFrame, audit: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select aligned readout parts with consistent labels for identical files."""
    audited = {str(s["sample_id"]) for g in audit["groups"] for s in g["samples"]}
    if not audited.issubset(set(frame["sample_id"].astype(str))):
        raise ValueError("Duplicate audit contains samples outside the feature cohort")
    conflicts = {str(s["sample_id"]): g["sha256"] for g in audit["groups"]
                 if g["conflicting_labels"] for s in g["samples"]}
    mask = frame["sample_id"].astype(str).isin(conflicts)
    excluded = frame.loc[mask, ["sample_id", "split", "label"]].copy()
    excluded["file_sha256"] = excluded["sample_id"].astype(str).map(conflicts)
    return frame.loc[~mask].copy(), excluded


def audit_exact_duplicates(cohort: pd.DataFrame, output_path: Path) -> dict:
    rows = []
    for item in cohort.to_dict("records"):
        digest = hashlib.sha256()
        with Path(item["absolute_path"]).open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        rows.append({"sample_id": item["sample_id"], "split": item["split"],
                     "label": item["label"], "sha256": digest.hexdigest()})
    table = pd.DataFrame(rows)
    repeats = []
    for digest, group in table.groupby("sha256"):
        if len(group) > 1:
            repeats.append({"sha256": digest, "samples": group.drop(columns="sha256").to_dict("records"),
                            "cross_split": group["split"].nunique() > 1,
                            "conflicting_labels": group["label"].nunique() > 1})
    result = {"n_samples": len(table), "n_duplicate_groups": len(repeats),
              "n_cross_split_groups": sum(x["cross_split"] for x in repeats),
              "n_conflicting_label_groups": sum(x["conflicting_labels"] for x in repeats),
              "groups": repeats}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, result)
    return result
