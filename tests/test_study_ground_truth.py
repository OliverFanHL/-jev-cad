from pathlib import Path

import pandas as pd
import pytest

from jev_tmcad.data.annotations import finalize_annotation_files, prepare_annotation_files


def test_blinded_two_rater_adjudication(tmp_path: Path):
    rows = []
    for split in ("val", "test"):
        for label in ("gear", "nut"):
            for i in range(2):
                sid = f"{split}-{label}-{i}"
                rows.append({"sample_id": sid, "split": split, "label": label,
                             "source_label": label, "relative_path": f"{label}/{sid}.step",
                             "absolute_path": str(tmp_path / f"{sid}.step")})
    cohort = pd.DataFrame(rows)
    pairs = []
    for split in ("val", "test"):
        ids = cohort.loc[cohort.split == split, "sample_id"].tolist()
        for qid in ids:
            for cid in ids:
                if cid != qid:
                    pairs.append({"split": split, "query_id": qid, "candidate_id": cid,
                                  "query_relative_path": f"secret/{qid}", "grade": ""})
    pool = tmp_path / "pool.csv"
    pd.DataFrame(pairs).to_csv(pool, index=False)
    package = tmp_path / "package"
    prepare_annotation_files(cohort, pool, package, per_class={"val": 1, "test": 1}, seed=42)
    rater = pd.read_csv(package / "rater_a.csv")
    assert "query_relative_path" not in rater
    assert "source_label" not in rater
    selected = pd.read_csv(package / "selected_pairs.csv")
    for name in ("rater_a", "rater_b", "adjudication"):
        frame = pd.read_csv(package / f"{name}.csv")
        frame["grade"] = [2 if q.split("-")[1] == c.split("-")[1] else 0 for q, c in
                          zip(selected.query_id, selected.candidate_id)]
        frame.to_csv(package / f"{name}.csv", index=False)
    output = tmp_path / "judged.csv"
    meta = finalize_annotation_files(package, output)
    assert len(pd.read_csv(output)) == 12
    assert meta["n_disagreements"] == 0
    with pytest.raises(FileExistsError):
        finalize_annotation_files(package, output)
