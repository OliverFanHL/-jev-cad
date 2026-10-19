"""Trace one held-out STEP part to the exact compact typed-decision request.

The example is explanatory, not selected as an additional performance trial.
All numbers and responses are loaded from saved study artifacts.
"""
from pathlib import Path
import json
import sys
import numpy as np
from PIL import Image

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT / "src"))
from jev_tmcad.semantic_state import QuantileSchema, build_compact_semantic_state  # noqa: E402
from cad_render import render_step  # noqa: E402

SID = "08b6e4a49adf79f320db"  # nut/165.stp, held-out test partition
FILES = {
    "JEV": PROJECT / "outputs/study/full/jev_results.jsonl",
    "Base Laya": PROJECT / "outputs/study/full/laya_results.jsonl",
    "FT Laya": PROJECT / "kaggle_laya/jev_tmcad_laya_ft_result/laya_results.jsonl",
}


def find_result(path):
    with path.open(encoding="utf-8") as stream:
        return next(json.loads(line) for line in stream if SID in line)


desc = pd.read_csv(PROJECT / "outputs/descriptors/tmcad_descriptors.csv",
                   low_memory=False, dtype={"sample_id": str})
row = desc.loc[desc.sample_id.eq(SID)].iloc[0]
assert row["split"] == "test" and row["label"] == "nut"
schema = QuantileSchema.from_dict(json.loads(
    (PROJECT / "outputs/descriptors/quantile_schema.json").read_text(encoding="utf-8")))
state = build_compact_semantic_state(row, schema)
results = {name: find_result(path) for name, path in FILES.items()}
for result in results.values():
    assert result["sample_id"] == SID and result["split"] == "test"

record = {
    "sample_id": SID, "source_path_for_audit": row["absolute_path"],
    "source_class_for_audit_only": "nut", "split": "test",
    "feature_derivation": {
        "bbox_lengths_xyz": [float(row["bbox_x"]), float(row["bbox_y"]), float(row["bbox_z"])],
        "sorted_extents": [float(row["dim_small"]), float(row["dim_mid"]), float(row["dim_large"])],
        "extent_mid_large_unrounded": float(row["dim_mid_large_ratio"]),
        "extent_mid_large_sent": state["geometry"]["extent_mid_large"],
        "planar_area_fraction_unrounded": float(row["face_area_frac_plane"]),
        "planar_area_fraction_sent": state["geometry"]["plane_area"],
    },
    "state_sent_to_each_model": state,
    "question_source": "configs/questions_compact.yaml",
    "choice_instruction": "Choose the mechanical part class supported by the geometry facts.",
    "choice_options": ["bearing", "bolt-screw", "bracket", "coupling", "flange",
                       "gear", "nut", "pulley", "shaft", "spring"],
    "example_noul_instructions": ["Is the part disk or ring shaped?",
                                  "Is a threaded fastener role plausible?"],
    "saved_responses": {
        name: {"top_choice": r["predicted_class"],
               "top_choice_probability": r["class_probabilities"][r["predicted_class"]],
               "disk_or_ring_score": r["semantic_signature"]["geom__disk_or_ring_form"],
               "threaded_fastener_score": r["semantic_signature"]["func__threaded_fastener_character"]}
        for name, r in results.items()},
}
(ROOT / "evidence/input_trace_nut_165.json").write_text(
    json.dumps(record, indent=2), encoding="utf-8")

render_dir = ROOT / "figures/renders"
render_dir.mkdir(exist_ok=True)
render_path = render_dir / "input_trace_render.png"
render_step(Path(row["absolute_path"]), render_path)
with Image.open(render_path) as im:
    rgb = np.asarray(im.convert("RGB"))
    ys, xs = np.where(np.any(rgb < 245, axis=2))
    assert len(xs) > 0
    pad = int(.08 * max(xs.max()-xs.min(), ys.max()-ys.min()))
    crop = im.crop((max(0, int(xs.min())-pad), max(0, int(ys.min())-pad),
                    min(im.width, int(xs.max())+pad+1),
                    min(im.height, int(ys.max())+pad+1))).convert("RGB")
    side = max(crop.size)
    canvas = Image.new("RGB", (side, side), "white")
    canvas.paste(crop, ((side-crop.width)//2, (side-crop.height)//2))
    canvas.resize((512, 512), Image.Resampling.LANCZOS).save(render_path)
print("Wrote held-out STEP trace record and CAD render for the LaTeX input presentation.")
