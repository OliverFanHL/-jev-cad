"""Compact-state demonstrations and geometric-tier questions."""
from __future__ import annotations

import copy
import numpy as np
import pandas as pd

from jev_tmcad.semantic_state import build_compact_semantic_state, QuantileSchema

PROPERTIES = {"fill_tier": "fill", "elongation_tier": "axis_elongation", "face_count_tier": "face_count"}
LEVELS = ["low", "middle", "high"]


def state_of(row):
    state = build_compact_semantic_state(row, QuantileSchema({}))
    if any(not np.isfinite(value) for value in state["geometry"].values()):
        raise ValueError("Nonfinite compact geometry")
    return state


def derive_rubrics(training: pd.DataFrame) -> dict:
    if set(training["split"]) != {"train"}:
        raise ValueError("Rubric thresholds must use training parts only")
    states = [state_of(row)["geometry"] for _, row in training.iterrows()]
    rubrics = {}
    for name, field in PROPERTIES.items():
        cuts = np.quantile([s[field] for s in states], [1/3, 2/3])
        if not np.isfinite(cuts).all() or cuts[0] >= cuts[1]:
            raise ValueError(f"Degenerate ordinal rubric: {name}")
        rubrics[name] = {"field": field, "low_max": float(cuts[0]), "middle_max": float(cuts[1])}
    return rubrics


def reference_for(row, rubrics):
    state = state_of(row)["geometry"]
    properties = {}
    for name, rubric in rubrics.items():
        x = state[rubric["field"]]
        level = 0 if x <= rubric["low_max"] else 1 if x <= rubric["middle_max"] else 2
        properties[name] = {"level": level, "at_least_middle": level >= 1, "high": level == 2}
    return {"part_class": str(row["label"]), "properties": properties}


def demonstration(row, rubrics):
    reference = reference_for(row, rubrics)
    return {"geometry_state": state_of(row),
            "reference_answers": {"part_class": reference["part_class"],
                                  "properties": {name: {"level_name": LEVELS[r["level"]],
                                                        "at_least_middle": r["at_least_middle"], "high": r["high"]}
                                                 for name, r in reference["properties"].items()}}}


def make_spec(original, rubrics):
    classification = copy.deepcopy(original["classification"])
    classification["instructions"] = (
        "Predict the recorded mechanical part class of query only from its geometry_state. "
        "The labeled_examples, when present, are training demonstrations. Do not answer for an example. "
        + classification["instructions"])
    spec = {"classification": classification, "semantic_signature": {}, "auxiliary_choices": {}, "ordinal_scores": {}}
    for name, r in rubrics.items():
        field, a, b = r["field"], format(r["low_max"], ".12g"), format(r["middle_max"], ".12g")
        descriptions = {
            "low": f"Low {name}: query.geometry_state.geometry.{field} is at most {a}, including equality.",
            "middle": f"Middle {name}: query.geometry_state.geometry.{field} is greater than {a} and at most {b}, including equality at the upper boundary.",
            "high": f"High {name}: query.geometry_state.geometry.{field} is greater than {b}."}
        instruction = (f"Evaluate {name} for query only using the stated boundaries. "
                       f"Examples provide the correct level_name for this property. "
                       "Low, middle and high are ordered geometric tiers, not manufacturing judgments.")
        spec["auxiliary_choices"][f"choice__{name}"] = {"output_key":name,"instructions": instruction, "criteria": descriptions}
        spec["ordinal_scores"][f"score__{name}"] = {"output_key":name,"instructions": instruction, "criteria": list(descriptions.values())}
        spec["semantic_signature"][f"{name}__at_least_middle"] = {
            "instructions": f"For query only, is geometry_state.geometry.{field} greater than {a}? This means the {name} level is middle or high. Equality means no. Examples provide the correct at_least_middle answer."}
        spec["semantic_signature"][f"{name}__high"] = {
            "instructions": f"For query only, is geometry_state.geometry.{field} greater than {b}? This means the {name} level is high. Equality means no. Examples provide the correct high answer."}
    return spec
