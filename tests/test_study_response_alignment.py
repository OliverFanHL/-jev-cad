"""Regression checks for compact payloads shared by distinct CAD samples."""
import json
import tempfile
import unittest
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from jev_tmcad.dataset import TMCAD_CLASSES
from jev_tmcad.semantic_state import QuantileSchema
from jev_tmcad.study_evaluation import _backend_frame
from jev_tmcad.laya_client import run_laya_for_dataframe
from jev_tmcad.utils import sha256_json


class ResponseAlignmentTests(unittest.TestCase):
    def setUp(self):
        self.rows = pd.DataFrame([
            {"sample_id": "a", "split": "train", "label": TMCAD_CLASSES[0]},
            {"sample_id": "b", "split": "test", "label": TMCAD_CLASSES[1]},
        ])
        self.spec = {"classification": {"id": "part_class", "instructions": "Class?",
                                         "criteria": {name: name for name in TMCAD_CLASSES}},
                     "semantic_signature": {"geom__shape": {"instructions": "Shape?"}}}
        self.state = {"same_compact_state": True}
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "responses.jsonl"

    def record(self, sample_id, model, choice):
        row = self.rows.set_index("sample_id").loc[sample_id]
        request_model = "jev-latest" if model.startswith("jev-") else model
        return {"sample_id": sample_id, "split": row["split"],
                "true_label": row["label"], "model": model,
                "request_hash": sha256_json({"model": request_model,
                                             "state": self.state, "questions": self.spec}),
                "predicted_class": choice,
                "class_probabilities": {name: 0.1 for name in TMCAD_CLASSES},
                "semantic_signature": {"geom__shape": 0.5},
                "latency_seconds": 1.0}

    def evaluate(self, backend, model):
        with patch("jev_tmcad.study_evaluation.build_compact_semantic_state",
                   return_value=self.state):
            return _backend_frame(self.path, self.rows, self.spec, QuantileSchema({}),
                                  backend=backend, model=model, required=True,
                                  include_numeric=True, include_profiles=False,
                                  state_variant="compact")

    def save(self, records):
        self.path.write_text("".join(json.dumps(record) + "\n" for record in records),
                             encoding="utf-8")

    def test_jev_uses_sample_and_hash_together(self):
        self.save([self.record("a", "jev-1.13.0", TMCAD_CLASSES[0]),
                   self.record("b", "jev-1.13.0", TMCAD_CLASSES[1])])
        result = self.evaluate("jev", "jev-latest")
        self.assertEqual(result["jev__choice"].tolist(), TMCAD_CLASSES[:2])
        self.assertTrue(result["jev__reused_from_sample_id"].isna().all())

    def test_laya_reuses_only_matching_payload_with_provenance(self):
        model = "repo/subfolder@revision"
        self.save([self.record("a", model, TMCAD_CLASSES[0])])
        result = self.evaluate("laya", model)
        self.assertEqual(result.loc[1, "laya__reused_from_sample_id"], "a")
        self.assertEqual(result.loc[1, "laya__cache_source"], "payload_reuse")
        self.assertTrue(pd.isna(result.loc[1, "laya__latency_seconds"]))
        self.assertEqual(result.loc[1, "label"], TMCAD_CLASSES[1])

    def test_mismatched_hash_is_rejected(self):
        record = self.record("a", "jev-1.13.0", TMCAD_CLASSES[0])
        record["request_hash"] = "invalid"
        self.save([record])
        with self.assertRaisesRegex(ValueError, "incompatible request hash"):
            self.evaluate("jev", "jev-latest")

    def test_future_laya_run_saves_each_sample_without_second_inference(self):
        model = "repo/subfolder@revision"
        self.save([self.record("a", model, TMCAD_CLASSES[0])])
        with patch.dict(sys.modules, {"laya": types.ModuleType("laya")}), patch(
            "jev_tmcad.laya_client.build_compact_semantic_state", return_value=self.state
        ):
            records = run_laya_for_dataframe(
                self.rows, QuantileSchema({}), self.spec, self.path,
                repository="repo", subfolder="subfolder", revision="revision",
                state_variant="compact")
        self.assertEqual(len(records), 2)
        self.assertEqual(records[1]["sample_id"], "b")
        self.assertEqual(records[1]["reused_from_sample_id"], "a")
        self.assertIsNone(records[1]["latency_seconds"])


if __name__ == "__main__":
    unittest.main()
