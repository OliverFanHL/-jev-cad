import pandas as pd

from jev_tmcad.semantic_state import QuantileSchema, build_compact_semantic_state
from jev_tmcad.study_data import cohort_hash, create_cohort


def test_compact_state_excludes_identity_and_label():
    row = pd.Series({"sample_id": "secret", "label": "gear", "relative_path": "gear/x.stp",
                     "dim_small_large_ratio": 0.25, "face_count": 20})
    state = build_compact_semantic_state(row, QuantileSchema(cuts={}))
    text = str(state)
    assert "secret" not in text
    assert "gear" not in text
    assert "x.stp" not in text


def test_cohort_hash_ignores_row_order():
    frame = pd.DataFrame([
        {"sample_id": "a", "split": "train", "label": "gear", "relative_path": "gear/a.stp"},
        {"sample_id": "b", "split": "test", "label": "spring", "relative_path": "spring/b.stp"},
    ])
    assert cohort_hash(frame) == cohort_hash(frame.iloc[::-1])


def test_all_cohort_keeps_every_valid_row_and_original_split(tmp_path):
    rows = []
    for split in ("train", "val", "test"):
        for label in ("gear", "spring"):
            for index in range(2):
                sample_id = f"{split}-{label}-{index}"
                rows.append({"sample_id": sample_id, "split": split, "label": label,
                             "relative_path": f"{label}/{sample_id}.stp",
                             "absolute_path": str(tmp_path / f"{sample_id}.stp"),
                             "extract_error": ""})
    rows.append({"sample_id": "failed", "split": "train", "label": "gear",
                 "relative_path": "gear/failed.stp", "absolute_path": str(tmp_path / "failed.stp"),
                 "extract_error": "STEP import failed"})
    descriptor_path = tmp_path / "descriptors.csv"
    pd.DataFrame(rows).to_csv(descriptor_path, index=False)
    cohort = create_cohort(descriptor_path, tmp_path / "unused_jev.jsonl",
                           tmp_path / "cohort.csv", source="all",
                           per_class={"train": 1, "val": 1, "test": 1}, seed=42)
    assert len(cohort) == 12
    assert "failed" not in set(cohort["sample_id"])
    assert cohort.groupby(["split", "label"]).size().eq(2).all()
