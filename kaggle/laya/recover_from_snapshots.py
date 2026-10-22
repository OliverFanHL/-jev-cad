"""Kaggle notebook cell: finalize a completed Laya run without retraining.

Paste this entire file into a new cell in the *same* Kaggle session after the
notebook setup cell. It also runs automatically in the revised notebook.
"""
import gc
import json
import math
import shutil
from importlib.metadata import version

import laya
import numpy as np
import torch
from safetensors.torch import load_file

from jev_tmcad.data.annotations import file_sha256
from jev_tmcad.training.laya import _base_checkpoint, _logits, _tokenize
from jev_tmcad.utils import read_jsonl, sha256_json, write_json


def recover_completed_laya_training(bundle, output, config, questions):
    """Select the best saved epoch and perform train-only temperature fitting."""
    checkpoint = output / "laya_checkpoint"
    history_path = output / "laya_checkpoint.history.json"
    if (checkpoint / "study_metadata.json").exists():
        return json.loads((checkpoint / "study_metadata.json").read_text(encoding="utf-8"))
    if checkpoint.exists():
        raise FileExistsError(f"Partial checkpoint exists; inspect it before recovery: {checkpoint}")
    history_record = json.loads(history_path.read_text(encoding="utf-8"))
    history = history_record["epochs"]
    expected_epochs = int(config["epochs"])
    if len(history) != expected_epochs or [r["epoch"] for r in history] != list(range(1, expected_epochs + 1)):
        raise ValueError("Training history is incomplete; recovery cannot assume four finished epochs")
    for key in ("batch_size", "accumulation_steps", "precision"):
        if history_record[key] != config[key]:
            raise ValueError(f"Saved training history uses another {key}")
    scores = [float(r["validation_macro_f1"]) for r in history]
    if not all(math.isfinite(s) for s in scores):
        raise ValueError("Validation history contains a nonfinite score")
    best_index = max(range(len(scores)), key=scores.__getitem__)
    best_epoch = best_index + 1
    snapshot = output / "laya_epoch_checkpoints" / f"epoch_{best_epoch:02d}.safetensors"
    summary = json.loads(snapshot.with_suffix(".json").read_text(encoding="utf-8"))
    if (summary["epoch"] != best_epoch or
            float(summary["validation_macro_f1"]) != scores[best_index] or
            summary["model_sha256"] != file_sha256(snapshot)):
        raise ValueError("Selected epoch snapshot differs from the validation history")
    print(f"Using saved epoch {best_epoch}/{expected_epochs} (validation macro F1 {scores[best_index]:.4f}); no training batches will run.", flush=True)

    data_dir = bundle / "data"
    data_meta = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))
    if data_meta["question_sha256"] != sha256_json(questions):
        raise ValueError("Choice question differs from the frozen fine-tuning data")
    options = list(questions["classification"]["criteria"])
    if data_meta["options"] != options:
        raise ValueError("Choice option order differs from the frozen fine-tuning data")
    data_hashes = {name: file_sha256(data_dir / f"{name}.jsonl") for name in
                   ("train", "calibration", "validation")}
    if data_hashes != data_meta["partition_hashes"]:
        raise ValueError("Frozen training, calibration, or validation data changed")
    cal_rows = read_jsonl(data_dir / "calibration.jsonl")
    if not cal_rows or any(r["split"] != "train" or r["partition"] != "calibration" for r in cal_rows):
        raise ValueError("Calibration must contain train-only samples")

    # The failed trainer may still be retained by IPython's exception traceback.
    import sys
    for name in ("last_traceback", "last_value", "last_type"):
        if hasattr(sys, name):
            setattr(sys, name, None)
    try:
        shell = get_ipython()
    except NameError:
        shell = None
    if shell is not None and getattr(shell, "last_execution_result", None) is not None:
        shell.last_execution_result.error_in_exec = None
    gc.collect()
    torch.cuda.empty_cache()
    print("Loading pinned base Laya and fitting temperature on train-only calibration records...", flush=True)
    base, resolved = _base_checkpoint(config["base_repository"],
                                      config["base_subfolder"], config["base_revision"])
    agent = laya.load(str(base), device=config["device"])
    cal = _tokenize(cal_rows, agent.tok, agent.cfg, questions["classification"])
    agent.model.load_state_dict(load_file(str(snapshot)), strict=True)
    z = np.asarray(_logits(agent.model, cal, agent.tok.pad_token_id,
                           int(config["batch_size"]), agent.device, config["precision"]), dtype=float)
    y = np.asarray([int(item["label"]) for item in cal], dtype=int)
    if z.shape != (len(y), len(options)) or not np.isfinite(z).all() or np.any((y < 0) | (y >= len(options))):
        raise ValueError("Calibration logits or Choice indices are invalid")

    def nll(temp):
        scaled = z / temp
        scaled -= scaled.max(axis=1, keepdims=True)
        logp = scaled - np.log(np.exp(scaled).sum(axis=1, keepdims=True))
        return -float(np.mean(logp[np.arange(len(y)), y]))

    temperature = float(min(np.geomspace(.5, 5., 81), key=nll))
    checkpoint.mkdir(parents=True)
    agent.model.encoder.save_pretrained(checkpoint / "encoder", safe_serialization=True)
    agent.tok.save_pretrained(checkpoint / "tokenizer")
    shutil.copy2(snapshot, checkpoint / "model.safetensors")
    model_cfg = dict(agent.cfg)
    model_cfg.pop("temperature_by_options", None)
    model_cfg["temperature"] = [temperature, 1.0, 1.0]
    write_json(checkpoint / "rl_agent_config.json", model_cfg)
    progress_path = output / "laya_checkpoint.progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.exists() else {}
    meta = {"base_repository": config["base_repository"], "base_subfolder": config["base_subfolder"],
            "base_revision": resolved, "laya_version": version("laya"),
            "data_metadata": str(data_dir / "metadata.json"), "data_hashes": data_hashes,
            "best_validation_macro_f1": scores[best_index], "best_epoch": best_epoch,
            "calibration_temperature": temperature, "calibration_nll": nll(temperature),
            "training_seconds": None, "elapsed_seconds_at_failure": progress.get("elapsed_seconds"),
            "accumulation_steps": int(config["accumulation_steps"]),
            "batch_size": int(config["batch_size"]), "precision": config["precision"],
            "save_epoch_snapshots": True, "epochs": history,
            "checkpoint_sha256": file_sha256(checkpoint / "model.safetensors"),
            "recovered_from_snapshot": str(snapshot),
            "recovery_note": "All training epochs completed before calibration failed; no optimizer steps were rerun."}
    write_json(checkpoint / "study_metadata.json", meta)
    del agent
    gc.collect()
    torch.cuda.empty_cache()
    return meta


if (output / "laya_checkpoint.history.json").exists() and not (output / "laya_checkpoint" / "study_metadata.json").exists():
    training = recover_completed_laya_training(bundle, output, config, questions)
    print("Recovered without retraining. Best validation epoch:", training["best_epoch"],
          "| calibration temperature:", training["calibration_temperature"])
