import json

import numpy as np
import pandas as pd

from jev_tmcad.provenance import class_label_mapping, with_source_labels
from jev_tmcad.reporting import prediction_frame, save_confusion


def test_source_folder_name_survives_canonical_encoding_and_plot(tmp_path):
    samples = pd.DataFrame([
        {"sample_id": "a", "split": "test", "label": "bolt-screw",
         "relative_path": "bolt_screw/a.stp", "absolute_path": str(tmp_path / "a.stp")},
        {"sample_id": "b", "split": "test", "label": "gear",
         "relative_path": "gear/b.stp", "absolute_path": str(tmp_path / "b.stp")},
    ])
    samples = with_source_labels(samples)
    classes = ["bolt-screw", "gear"]
    mapping = class_label_mapping(samples, classes)
    assert mapping == {"bolt-screw": ["bolt_screw"], "gear": ["gear"]}
    predictions = ["gear", "gear"]
    table = prediction_frame(samples, predictions, np.array([[0.2, 0.8], [0.1, 0.9]]),
                             classes, mapping)
    assert table.loc[0, "true_source_label"] == "bolt_screw"
    assert table.loc[0, "label"] == "bolt-screw"
    assert table.loc[0, "relative_path"] == "bolt_screw/a.stp"
    assert table.loc[0, "prob__gear"] == 0.8

    figure = tmp_path / "confusion.png"
    save_confusion(samples["label"], predictions, classes, mapping, "Test", figure)
    counts = pd.read_csv(figure.with_suffix(".csv"))
    assert counts.loc[0, "true_source_label"] == "bolt_screw"
    assert counts.loc[0, "predicted:gear"] == 1
    assert json.loads(figure.with_suffix(".labels.json").read_text())["axis_order_source"][0] == "bolt_screw"
    assert figure.exists()


def test_source_label_mismatch_is_rejected():
    frame = pd.DataFrame([{"sample_id": "a", "label": "gear",
                           "relative_path": "bolt_screw/a.stp"}])
    try:
        with_source_labels(frame)
    except ValueError as exc:
        assert "disagree" in str(exc)
    else:
        raise AssertionError("Invalid original-to-canonical mapping was accepted")
