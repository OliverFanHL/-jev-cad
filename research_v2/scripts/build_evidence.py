"""Build matched-study tables, SciencePlots figures, and auditable examples.

Run from the repository root with the laya-tmcad Python environment:
    python research_v2/scripts/build_evidence.py

Inputs are the registered evaluator's saved per-sample predictions and the
frozen cohort. This script never refits a model or changes a split.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parent
PROJECT = RESEARCH.parent
sys.path.append(str(HERE / "_vendor"))
sys.path.insert(0, str(PROJECT / "src"))
import scienceplots  # noqa: E402,F401 - registers the SciencePlots styles
from jev_tmcad.cad_features import _load_shape  # noqa: E402
from jev_tmcad.semantic_state import (  # noqa: E402
    build_compact_semantic_state, build_semantic_state, QuantileSchema,
)
from cad_render import render_step  # noqa: E402

EVAL = RESEARCH / "evaluation" / "classification"
OUT = RESEARCH / "figures"
DATA = RESEARCH / "evidence"
OUT.mkdir(exist_ok=True)
DATA.mkdir(exist_ok=True)

with plt.style.context(["science", "no-latex"]):
    plt.rcParams.update({
        "font.family": "Arial", "font.sans-serif": ["Arial"],
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8,
        "ytick.labelsize": 8, "axes.titlesize": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.figsize": (7.2, 4.2), "figure.dpi": 150,
        "savefig.bbox": "standard",
    })
    COLORS = {"geometry": "#2C6E8F", "semantic": "#B46B31", "fusion": "#4F7862"}

    summary = json.loads((EVAL / "summary.json").read_text(encoding="utf-8"))["methods"]
    names = list(summary)
    all_rows = pd.DataFrame([
        {"method": name, "n": item["n"], "accuracy": item["accuracy"],
         "macro_f1": item["macro_f1"], "balanced_accuracy": item["balanced_accuracy"],
         "nll": item["nll"], "brier": item["brier"]}
        for name, item in summary.items()
    ])
    assert len(all_rows) == 26 and all(all_rows.n == 1960)
    all_rows.to_csv(DATA / "matched_classification.csv", index=False)

    cohort = pd.read_csv(PROJECT / "outputs/study/full/cohort.csv", dtype={"sample_id": str})
    assert len(cohort) == 9798 and cohort.sample_id.is_unique
    pred = {}
    for name in names:
        frame = pd.read_csv(EVAL / f"{name}.csv", dtype={"sample_id": str})
        assert len(frame) == 1960 and frame.sample_id.is_unique
        assert set(frame.sample_id) == set(cohort.loc[cohort.split == "test", "sample_id"])
        pred[name] = frame.set_index("sample_id").sort_index()
    reference = pred["descriptor_plus_geometry"]
    for name, frame in pred.items():
        assert frame["label"].equals(reference["label"]), name
        assert frame["split"].eq("test").all(), name

    ytrue = reference["label"]
    main = ["descriptor_only", "geometry_embedding", "descriptor_plus_geometry",
            "jev_all_questions", "geometry_plus_jev", "laya_direct_choice",
            "laya_ft_direct_choice", "geometry_plus_laya_ft_choice_probabilities"]
    rows = []
    for cls in sorted(ytrue.unique()):
        mask = ytrue.eq(cls)
        row = {"class": cls, "n": int(mask.sum())}
        for name in main:
            row[name] = float(pred[name].loc[mask, "prediction"].eq(cls).mean())
        rows.append(row)
    recall = pd.DataFrame(rows)
    recall.to_csv(DATA / "class_recall.csv", index=False)

    def pairing(a: str, b: str) -> dict:
        ac = pred[a]["prediction"].eq(ytrue)
        bc = pred[b]["prediction"].eq(ytrue)
        return {"a": a, "b": b, "both_correct": int((ac & bc).sum()),
                "a_only": int((ac & ~bc).sum()), "b_only": int((~ac & bc).sum()),
                "both_wrong": int((~ac & ~bc).sum())}

    pairs = [pairing("laya_ft_direct_choice", "laya_direct_choice"),
             pairing("descriptor_plus_geometry", "laya_ft_direct_choice"),
             pairing("geometry_embedding", "geometry_plus_jev"),
             pairing("descriptor_plus_geometry", "geometry_plus_laya_ft_choice_probabilities"),
             pairing("geometry_embedding", "geometry_plus_laya_ft_choice_probabilities"),
             pairing("geometry_embedding", "descriptor_plus_geometry")]
    (DATA / "paired_outcomes.json").write_text(json.dumps(pairs, indent=2), encoding="utf-8")

    from build_class_error_figure import build as build_class_error_figure
    build_class_error_figure(recall, pairs, OUT)

    audit = json.loads((PROJECT / "outputs/study/full/evaluation/duplicate_audit.json").read_text(encoding="utf-8"))
    compromised = set()
    for group in audit["groups"]:
        splits = {s["split"] for s in group["samples"]}
        if "test" in splits and ("train" in splits or "val" in splits):
            compromised.update(s["sample_id"] for s in group["samples"] if s["split"] == "test")
    clean_ids = reference.index.difference(list(compromised))
    sensitivity = {"test_n": len(reference), "cross_split_duplicate_test_n": len(compromised),
                   "remaining_n": len(clean_ids), "methods": {}}
    for name in main:
        frame = pred[name].loc[clean_ids]
        sensitivity["methods"][name] = float(frame.prediction.eq(frame.label).mean())
    (DATA / "duplicate_sensitivity.json").write_text(json.dumps(sensitivity, indent=2), encoding="utf-8")

    # Fixed, auditable cases span agreement, asymmetric success, and shared
    # failure. They are illustrative and excluded from any frequency estimate.
    case_ids = [
        ("agreement", "00daa03f69a3c6948084"),
        ("adaptation_gain", "1dcf73c1bc7182e9f9af"),
        ("geometry_advantage", "0bb26142b84c91d03539"),
        ("adapted_choice_advantage", "2408daa14ba0de294b93"),
        ("shared_failure", "6476cc716ec71eef9aeb"),
        ("complex_adapted_choice_advantage", "fc8bca5774dab9bc8919"),
        ("complex_geometry_advantage", "6d76dca2861e29e8113b"),
    ]
    examples = []
    render_dir = OUT / "renders"
    render_dir.mkdir(exist_ok=True)
    for kind, sid in case_ids:
        assert sid in clean_ids, f"Cross-split duplicate qualitative case: {sid}"
        r = reference.loc[sid]
        render_path = render_dir / f"{sid}.png"
        render_step(r.absolute_path, render_path)
        with Image.open(render_path) as im:
            rgb = np.asarray(im.convert("RGB"))
            mask = np.any(rgb < 245, axis=2)
            ys, xs = np.where(mask)
            assert len(xs) > 0, f"Empty render: {sid}"
            pad = int(.08 * max(xs.max()-xs.min(), ys.max()-ys.min()))
            bounds = (max(0, int(xs.min())-pad), max(0, int(ys.min())-pad),
                      min(im.width, int(xs.max())+pad+1),
                      min(im.height, int(ys.max())+pad+1))
            crop = im.crop(bounds).convert("RGB")
            side = max(crop.size)
            canvas = Image.new("RGB", (side, side), "white")
            canvas.paste(crop, ((side-crop.width)//2, (side-crop.height)//2))
            canvas.resize((512, 512), Image.Resampling.LANCZOS).save(render_path)
        examples.append({
            "pattern": kind, "sample_id": sid, "source_path": r.absolute_path,
            "true": r.label, "render": render_path.relative_to(RESEARCH).as_posix(),
            "view_type": "STEP solid rendering",
            "descriptor_plus_geometry": r.prediction,
            "jev_direct_choice": pred["jev_direct_choice"].at[sid, "prediction"],
            "jev_noul_probe": pred["jev_all_questions"].at[sid, "prediction"],
            "base_laya_choice": pred["laya_direct_choice"].at[sid, "prediction"],
            "fine_tuned_laya_choice": pred["laya_ft_direct_choice"].at[sid, "prediction"],
        })
    (DATA / "qualitative_examples.json").write_text(json.dumps(examples, indent=2), encoding="utf-8")

    desc = pd.read_csv(PROJECT / "outputs/descriptors/tmcad_descriptors.csv", low_memory=False,
                       dtype={"sample_id": str})
    schema = QuantileSchema.from_dict(json.loads((PROJECT / "outputs/descriptors/quantile_schema.json").read_text(encoding="utf-8")))
    query_examples = []
    for item in examples[:2] + examples[4:5]:
        row = desc.loc[desc.sample_id.eq(item["sample_id"])].iloc[0]
        query_examples.append({"sample_id": item["sample_id"], "source_class_for_audit": item["true"],
                               "state_sent_to_JEV_and_Laya": build_compact_semantic_state(row, schema)})
    (DATA / "query_examples.json").write_text(json.dumps(query_examples, indent=2), encoding="utf-8")
    pilot_row = desc.loc[desc.sample_id.eq(examples[0]["sample_id"])].iloc[0]
    (DATA / "query_example_full_state_pilot.json").write_text(
        json.dumps({"status": "illustrative full-state transform of the same part; not a scored pilot request",
                    "sample_id": examples[0]["sample_id"],
                    "state": build_semantic_state(pilot_row, schema,
                                                  include_numeric_context=True,
                                                  include_profile_arrays=True)}, indent=2),
        encoding="utf-8")

    print(f"Wrote {len(all_rows)} matched method rows, {len(examples)} rendered table cases, and evidence files.")
