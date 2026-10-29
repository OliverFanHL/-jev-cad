"""Representation, selection, prompting, and checkpoint integration checks."""
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest

from jev_tmcad.data.cohort import cohort_hash
from jev_tmcad.data.dataset import TMCAD_CLASSES
from jev_tmcad.evaluation.audit import exclude_label_conflicts
from jev_tmcad.evaluation.classification import evaluate_classification
from jev_tmcad.evaluation.experiments import CLASSIFICATION_PROBES, feature_blocks
from jev_tmcad.geometry.encoder import _model, extract_embeddings, mesh_cache_hash, train_geometry
from jev_tmcad.models.prompting import state_of
from jev_tmcad.semantic_state import COMPACT_FIELDS, compact_numeric_frame
from jev_tmcad.workflows.prompting import build_prompt_requests, prompt_feature_frame


def parts(per_class=3):
    rows = []
    for split in ("train", "val", "test"):
        for label_index, label in enumerate(TMCAD_CLASSES):
            for j in range(per_class):
                sid = f"{split}_{label_index}_{j}"
                rows.append({"sample_id": sid, "label": label, "split": split,
                             "relative_path": f"{label}/{sid}.stp", "absolute_path": f"/{sid}.stp",
                             **{column: label_index + j / 10 + i / 100
                                for i, column in enumerate(COMPACT_FIELDS.values())}})
    return pd.DataFrame(rows)


def test_compact_probe_matches_serialized_input():
    frame = parts()
    numeric = compact_numeric_frame(frame)
    assert numeric.shape == (len(frame), 14)
    for i, row in frame.iterrows():
        assert numeric.iloc[i].to_list() == list(state_of(row)["geometry"].values())


def test_fusion_recipes_run_on_a_shared_cohort(tmp_path, monkeypatch):
    import jev_tmcad.evaluation.classification as classification
    monkeypatch.setattr(classification, "_save_method", lambda *args: None)
    frame = parts()
    frame = pd.concat([frame, compact_numeric_frame(frame)], axis=1)
    frame["desc__one"] = np.arange(len(frame)) % 30
    frame["geom__000"] = np.arange(len(frame)) % 30
    frame["jev__sig__geom__one"] = (np.arange(len(frame)) % 10) / 10
    for name in ("compact_state", "compact_state_plus_jev", "descriptor_plus_geometry_plus_jev"):
        blocks = feature_blocks(frame)
        result = evaluate_classification(frame, tmp_path / name, c_grid=[0.1, 1], only=name)[name]
        assert result["n"] == 30 and result["n_train"] == result["n_val"] == 30
        assert result["n_features"] == sum(len(blocks[b]) for b in CLASSIFICATION_PROBES[name])
        assert result["selected_C"] in (0.1, 1)
        assert np.isfinite(result["nll"]) and np.isfinite(result["brier"])


def test_direct_choice_uses_probability_argmax(tmp_path, monkeypatch):
    import jev_tmcad.evaluation.classification as classification
    monkeypatch.setattr(classification, "_save_method", lambda *args: None)
    frame = parts()
    frame["jev__choice"] = TMCAD_CLASSES[-1]
    for c in TMCAD_CLASSES:
        frame[f"jev__prob__{c}"] = frame.label.eq(c).astype(float)
    result = evaluate_classification(frame, tmp_path, c_grid=[1], only="jev_direct_choice")
    assert result["jev_direct_choice"]["accuracy"] == 1.0


def test_file_conflicts_filter_all_splits_by_identity():
    frame = parts()
    ids = [frame.iloc[i].sample_id for i in (0, 30, 60)]
    audit = {"groups": [{"sha256": "abc", "conflicting_labels": True,
                         "samples": [{"sample_id": sid} for sid in ids]}]}
    kept, excluded = exclude_label_conflicts(frame, audit)
    assert set(excluded.sample_id) == set(ids)
    assert len(kept) == len(frame) - 3
    assert excluded.file_sha256.eq("abc").all()


def test_prompt_supports_exclude_query_state_and_ignore_query_labels():
    frame = parts(5)
    questions = {"classification": {"id": "part_class", "instructions": "Classify geometry.",
                                     "criteria": {c: c for c in TMCAD_CLASSES}}}
    requests = list(build_prompt_requests(frame, questions, shots=[0, 4], seed=42))
    changed = frame.copy()
    changed.loc[changed.split.ne("train"), "label"] = TMCAD_CLASSES[-1]
    repeated = list(build_prompt_requests(changed, questions, shots=[0, 4], seed=42))
    assert [r["request_hash"] for r in repeated] == [r["request_hash"] for r in requests]
    for request in requests:
        examples = request["payload"]["state"]["labeled_examples"]
        assert len(examples) == 10 * request["shots_per_class"]
        assert all(sid.startswith("train_") for sid in request["support_ids"])
        query_state = request["payload"]["state"]["query"]["geometry_state"]
        assert all(example["geometry_state"] != query_state for example in examples)
    records = [{k: v for k, v in request.items() if k != "payload"} |
               {"model": "jev-1.13.0", "class_probabilities": {c: 1 / 10 for c in TMCAD_CLASSES}}
               for request in requests]
    matched = prompt_feature_frame(iter(requests), records)
    assert len(matched) == len(frame)
    records[0]["request_hash"] = "changed"
    with pytest.raises(ValueError, match="incompatible"):
        prompt_feature_frame(iter(requests), records)


def test_prompt_command_caches_shared_payload_and_resumes(tmp_path, monkeypatch):
    import jev_tmcad.workflows.prompting as prompting
    from jev_tmcad.utils import read_jsonl
    config = tmp_path / "run.yaml"
    config.write_text("prompting:\n  output_dir: responses\n", encoding="utf-8")
    requests = [{"sample_id": sid, "split": "test", "shots_per_class": 0,
                 "request_hash": "shared", "support_ids": [],
                 "payload": {"model": "jev-latest", "state": {}, "questions": {}}}
                for sid in ("a", "b")]
    monkeypatch.setattr(prompting, "requests_from_config", lambda cfg: iter(requests))
    calls = []
    def acquire(*args):
        calls.append(args)
        return {"model": "jev-1.13.0", "class_probabilities": {c: .1 for c in TMCAD_CLASSES}}
    monkeypatch.setattr(prompting, "_call_typesafe", acquire)
    args = SimpleNamespace(config=str(config), dry_run=True)
    prompting.cmd_prompt(args)
    assert not calls and not (tmp_path / "responses/responses.jsonl").exists()
    args.dry_run = False
    prompting.cmd_prompt(args)
    prompting.cmd_prompt(args)
    assert len(calls) == 1
    assert {r["sample_id"] for r in read_jsonl(tmp_path / "responses/responses.jsonl")} == {"a", "b"}
    requests[0]["request_hash"] = "changed"
    with pytest.raises(ValueError, match="inputs changed"):
        prompting.cmd_prompt(args)


def test_manuscript_encoder_and_checkpoint_roundtrip(tmp_path):
    torch = pytest.importorskip("torch")
    model = _model(32)
    assert sum(p.numel() for p in model.parameters()) == 24938
    frame = parts().iloc[[0, 3, 30, 33, 60, 63]].copy()
    meshes = []
    for i, row in frame.iterrows():
        path = tmp_path / f"{row.sample_id}.npz"
        np.savez(path, features=np.random.default_rng(i).normal(size=(8, 6)).astype("float32"),
                 knn=np.array([[(j + k + 1) % 8 for k in range(3)] for j in range(8)]))
        meshes.append({"sample_id": row.sample_id, "cache_key": row.sample_id, "mesh_path": str(path)})
    meshes = pd.DataFrame(meshes)
    checkpoint = tmp_path / "encoder.pt"
    train_geometry(frame, meshes, checkpoint, seed=42, device="cpu", epochs=1,
                   batch_size=2, learning_rate=0.001, weight_decay=0.0001,
                   embedding_dim=32, patience=1)
    saved = torch.load(checkpoint, weights_only=False)
    assert saved["edge_channels"] == (32, 64) and saved["projection_dim"] == 64
    embeddings = extract_embeddings(frame, meshes, checkpoint, tmp_path / "features.csv",
                                     device="cpu", batch_size=2)
    assert embeddings.filter(regex="^geom__").shape == (6, 32)
    # Earlier checkpoints store the original widths implicitly.
    wide = _model(128, edge_channels=(64, 128), projection_dim=256)
    torch.save({"state_dict": wide.state_dict(), "embedding_dim": 128, "epoch": 1,
                "cohort_hash": cohort_hash(frame), "mesh_hash": mesh_cache_hash(meshes)}, checkpoint)
    embeddings = extract_embeddings(frame, meshes, checkpoint, tmp_path / "wide.csv",
                                     device="cpu", batch_size=2)
    assert embeddings.filter(regex="^geom__").shape == (6, 128)
