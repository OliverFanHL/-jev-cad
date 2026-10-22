import pandas as pd

from jev_tmcad.data.dataset import canonical_label, select_subset


def test_class_aliases():
    assert canonical_label("bearing") == "bearing"
    assert canonical_label("Bolt_Screw") == "bolt-screw"
    assert canonical_label("SPRINGS") == "spring"
    assert canonical_label("screw") == "bolt-screw"


def test_subset_cap_applies_within_each_split_and_class():
    data = pd.DataFrame([
        {"sample_id": f"{split}-{label}-{i}", "split": split, "label": label}
        for split in ("train", "val", "test")
        for label in ("gear", "spring")
        for i in range(5)
    ])
    selected = select_subset(data, splits=["train", "val", "test"], max_per_class=2)
    assert len(selected) == 12
    assert selected.groupby(["split", "label"]).size().eq(2).all()
