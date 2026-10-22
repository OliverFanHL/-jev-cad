"""Synthetic contract checks; no model, server, or CAD experiments are run."""
from __future__ import annotations

import json
import sys
import types

import pandas as pd
import pytest

from jev_tmcad.models.clm import run_clm_for_dataframe
from jev_tmcad.semantic_state import QuantileSchema


def test_clm_records_are_per_sample_and_resume_with_a_pinned_head(tmp_path, monkeypatch):
    checkpoint = tmp_path / "head.pt"
    checkpoint.write_bytes(b"synthetic head identity")
    output = tmp_path / "clm.jsonl"
    spec = {"classification": {"id": "part_class", "instructions": "Choose class",
                               "criteria": {"bearing": "bearing part", "gear": "gear part"}},
            "semantic_signature": {"geom__rotational": {"instructions": "Is it rotational?"}}}
    rows = pd.DataFrame([{"sample_id": "a", "split": "train", "label": "bearing"},
                         {"sample_id": "b", "split": "test", "label": "gear"}])
    calls = []

    class FakeEmbedder:
        def __init__(self, **kwargs):
            self.session = types.SimpleNamespace(close=lambda: None)

        def healthy(self):
            return True

    class FakeEngine:
        def __init__(self, **kwargs):
            pass

        def models(self):
            return [{"name": "clm-latest"}]

        def answer(self, state, questions, model):
            calls.append((state, questions, model))
            return {"model": model, "answers": {
                "part_class": {"choice": "bearing", "confidence": .6,
                               "probabilities": {"bearing": .8, "gear": .2}},
                "geom__rotational": {"noul": .7}},
                "usage": {"billing_units": 2, "input_tokens": 20, "output_tokens": 0}}

    package = types.ModuleType("clm")
    package.Engine = FakeEngine
    embedder = types.ModuleType("clm.embedder")
    embedder.Embedder = FakeEmbedder
    monkeypatch.setitem(sys.modules, "clm", package)
    monkeypatch.setitem(sys.modules, "clm.embedder", embedder)
    monkeypatch.setattr("jev_tmcad.models.clm.version", lambda name: "0.1.0")
    kwargs = {"checkpoint": checkpoint, "state_variant": "compact"}
    preview = run_clm_for_dataframe(rows, QuantileSchema({}), spec, output,
                                    dry_run=True, **kwargs)
    assert preview["pending"] == 2 and not output.exists()
    result = run_clm_for_dataframe(rows, QuantileSchema({}), spec, output, **kwargs)
    assert result["completed"] == 2 and len(calls) == 1
    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert {r["sample_id"] for r in records} == {"a", "b"}
    assert records[1]["cache_source"] == "payload_reuse"
    assert records[1]["reused_from_sample_id"] == "a"
    assert records[0]["semantic_signature"] == {"geom__rotational": .7}
    run_clm_for_dataframe(rows, QuantileSchema({}), spec, output, **kwargs)
    assert len(calls) == 1 and len(output.read_text().splitlines()) == 2
    checkpoint.write_bytes(b"changed head")
    with pytest.raises(ValueError, match="different model, encoder, or head checkpoint"):
        run_clm_for_dataframe(rows, QuantileSchema({}), spec, output, **kwargs)
