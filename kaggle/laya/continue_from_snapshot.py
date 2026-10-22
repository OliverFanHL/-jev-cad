"""Optional Kaggle notebook cell: four more epochs from saved model weights.

Paste into the *existing* Kaggle notebook after its setup cells. This is a
weight-only continuation: the original run did not save AdamW/scaler/RNG state.
No code from a replacement upload Dataset is needed.
"""
import hashlib
import json
import math
import random
import shutil
import time
from importlib.metadata import version
from pathlib import Path

import laya
import numpy as np
import torch
import torch.nn.functional as F
from laya.common import collate_items, proper_reward
from safetensors.torch import load_file, save_file
from tqdm.auto import tqdm

from jev_tmcad.data.annotations import file_sha256
from jev_tmcad.training.laya import _base_checkpoint, _logits, _tokenize, _validation_macro_f1
from jev_tmcad.utils import read_jsonl, sha256_json, write_json


def _choice_temperature(items, logits, n_classes):
    y = np.asarray([int(item["label"]) for item in items], dtype=int)
    z = np.asarray(logits, dtype=float)
    if not len(y) or z.shape != (len(y), n_classes) or not np.isfinite(z).all() or np.any((y < 0) | (y >= n_classes)):
        raise ValueError("Invalid train-only calibration logits or Choice IDs")

    def nll(temperature):
        scores = z / temperature
        scores -= scores.max(axis=1, keepdims=True)
        log_probs = scores - np.log(np.exp(scores).sum(axis=1, keepdims=True))
        return -float(np.mean(log_probs[np.arange(len(y)), y]))

    temperature = float(min(np.geomspace(.5, 5., 81), key=nll))
    return temperature, nll(temperature)


def continue_laya_from_saved_weights(bundle, output, config, questions, continuation_config):
    if not continuation_config["retain_original_training_settings"]:
        raise ValueError("This diagnostic requires the original training settings")
    if int(continuation_config["additional_epochs"]) != 4 or int(continuation_config["start_epoch"]) != 4:
        raise ValueError("This trial is defined as four epochs following the original four")
    if float(continuation_config["continuation_sigma"]) != float(config["sigma_end"]):
        raise ValueError("Keep the original RL noise schedule endpoint during continuation")
    if int(config["batch_size"]) != 8 or int(config["accumulation_steps"]) != 1 or config["precision"] != "fp16":
        raise ValueError("The continuation requires the recorded Kaggle batch and precision settings")
    if not torch.cuda.is_available():
        raise RuntimeError("Select a Kaggle GPU")

    original_history_path = output / "laya_checkpoint.history.json"
    original = json.loads(original_history_path.read_text(encoding="utf-8"))
    prior = original["epochs"]
    if len(prior) != 4 or [row["epoch"] for row in prior] != [1, 2, 3, 4]:
        raise ValueError("Expected a complete, validated original four-epoch run")
    for key in ("batch_size", "accumulation_steps", "precision"):
        if original[key] != config[key]:
            raise ValueError(f"Original training {key} differs from continuation settings")
    prior_scores = [float(row["validation_macro_f1"]) for row in prior]
    if not all(math.isfinite(score) for score in prior_scores):
        raise ValueError("Original validation history contains a nonfinite score")
    start_epoch = 1 + max(range(4), key=prior_scores.__getitem__)
    if start_epoch != 4:
        raise ValueError("The best original epoch is not the current epoch; inspect the trajectory first")
    start_snapshot = output / "laya_epoch_checkpoints/epoch_04.safetensors"
    start_summary = json.loads(start_snapshot.with_suffix(".json").read_text(encoding="utf-8"))
    start_digest = file_sha256(start_snapshot)
    if (start_summary["model_sha256"] != start_digest or
            float(start_summary["validation_macro_f1"]) != prior_scores[3]):
        raise ValueError("The saved starting snapshot does not match the epoch-four validation record")
    original_checkpoint_meta = output / "laya_checkpoint/study_metadata.json"
    if original_checkpoint_meta.exists():
        base_meta = json.loads(original_checkpoint_meta.read_text(encoding="utf-8"))
        if base_meta["best_epoch"] != 4 or base_meta["checkpoint_sha256"] != start_digest:
            raise ValueError("The selected original checkpoint differs from the starting snapshot")

    data_dir = bundle / "data"
    data_meta = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))
    if data_meta["question_sha256"] != sha256_json(questions):
        raise ValueError("Frozen Choice question changed")
    options = list(questions["classification"]["criteria"])
    if data_meta["options"] != options:
        raise ValueError("Frozen Choice option order changed")
    data_hashes = {name: file_sha256(data_dir / f"{name}.jsonl") for name in
                   ("train", "calibration", "validation")}
    if data_hashes != data_meta["partition_hashes"]:
        raise ValueError("Frozen fine-tuning partitions changed")
    train_rows, cal_rows, val_rows = (read_jsonl(data_dir / f"{name}.jsonl") for name in
                                     ("train", "calibration", "validation"))
    if (not train_rows or not cal_rows or not val_rows or
            any(row["split"] != "train" for row in train_rows + cal_rows) or
            any(row["split"] != "val" for row in val_rows) or
            set(row["state_hash"] for row in train_rows) & set(row["state_hash"] for row in cal_rows)):
        raise ValueError("Frozen training, calibration, and validation partitions are invalid")

    trial = output / "continuation"
    selection_path = trial / "selection.json"
    if selection_path.exists():
        return json.loads(selection_path.read_text(encoding="utf-8"))
    if trial.exists():
        raise FileExistsError(f"Partial continuation exists; inspect the saved epochs before retrying: {trial}")
    snapshot_dir = trial / "epoch_checkpoints"
    snapshot_dir.mkdir(parents=True)
    started = time.perf_counter()

    def progress(stage, **fields):
        record = {"stage": stage, "elapsed_seconds": round(time.perf_counter() - started, 1), **fields}
        temporary = trial / "progress.json.tmp"
        temporary.write_text(json.dumps(record, indent=2), encoding="utf-8")
        temporary.replace(trial / "progress.json")

    write_json(trial / "continuation_config.json", continuation_config)
    write_json(trial / "original_history.json", original)
    progress("loading", source_snapshot=str(start_snapshot))
    seed = int(config["project_seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    base, resolved = _base_checkpoint(config["base_repository"], config["base_subfolder"],
                                      config["base_revision"])
    agent = laya.load(str(base), device=config["device"])
    model, tokenizer, device = agent.model, agent.tok, agent.device
    model.load_state_dict(load_file(str(start_snapshot)), strict=True)
    if hasattr(model.encoder, "gradient_checkpointing_enable"):
        model.encoder.gradient_checkpointing_enable()
    train = _tokenize(train_rows, tokenizer, agent.cfg, questions["classification"])
    cal = _tokenize(cal_rows, tokenizer, agent.cfg, questions["classification"])
    val = _tokenize(val_rows, tokenizer, agent.cfg, questions["classification"])
    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": float(config["encoder_lr"])},
        {"params": [p for name, p in model.named_parameters() if not name.startswith("encoder.")],
         "lr": float(config["head_lr"])}], weight_decay=float(config["weight_decay"]))
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    batch_size = int(config["batch_size"])
    group_size = int(config["group_size"])
    accumulation = int(config["accumulation_steps"])
    sigma = float(continuation_config["continuation_sigma"])
    rl_weight = float(config["rl_weight"])
    history = list(prior)
    best_epoch, best_score, best_snapshot = 4, prior_scores[3], start_snapshot
    print("Starting weight-only continuation from epoch 4; AdamW and GradScaler start fresh.", flush=True)
    for global_epoch in range(5, 9):
        epoch_start = time.perf_counter()
        model.train()
        order = np.random.default_rng(seed + global_epoch - 1).permutation(len(train))
        batches = range(0, len(order), batch_size)
        losses = []
        optimizer.zero_grad(set_to_none=True)
        progress("training", epoch=global_epoch, batches_done=0, batches_total=len(batches))
        for batch_number, offset in enumerate(tqdm(batches, desc=f"Continuation epoch {global_epoch}/8", leave=False), 1):
            b = collate_items([[train[int(i)]] for i in order[offset:offset + batch_size]], tokenizer.pad_token_id)
            ids, att, pos, mask, qtype, target = (b[key].to(device) for key in
                                                   ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype", "target"))
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                logits, _ = model(ids, att, pos, mask, qtype)
            logits = logits.float()
            ce = -(target * F.log_softmax(logits, -1)).sum(-1).mean()
            eps = torch.randn((group_size, *logits.shape), device=device) * sigma
            eps -= eps.mean(dim=-1, keepdim=True)
            sampled_probs = torch.softmax(logits.detach()[None, :, :] + eps, dim=-1)
            rewards = proper_reward(sampled_probs, target, qtype, mask, w_sph=.75, w_rps=1.)
            advantages = (rewards - rewards.mean(0, keepdim=True)) / rewards.std(0, keepdim=True).clamp_min(1e-6)
            sampled = torch.multinomial(sampled_probs.reshape(-1, sampled_probs.shape[-1]), 1).reshape(group_size, -1)
            logp = F.log_softmax(logits, -1).unsqueeze(0).expand(group_size, -1, -1)
            rl = -(advantages.detach() * logp.gather(-1, sampled[..., None]).squeeze(-1)).mean()
            loss = ce + rl_weight * rl
            scaler.scale(loss / accumulation).backward()
            if batch_number % accumulation == 0 or offset + batch_size >= len(order):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.detach().cpu()))
            if batch_number == 1 or batch_number % 25 == 0 or batch_number == len(batches):
                progress("training", epoch=global_epoch, batches_done=batch_number,
                         batches_total=len(batches), recent_loss=round(float(np.mean(losses[-25:])), 5))
        progress("validating", epoch=global_epoch)
        snapshot = snapshot_dir / f"epoch_{global_epoch:02d}.safetensors"
        save_file({key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()}, snapshot)
        validation_logits = _logits(model, val, tokenizer.pad_token_id, batch_size, device, config["precision"])
        score = _validation_macro_f1(val, validation_logits, len(options))
        record = {"epoch": global_epoch, "train_loss": float(np.mean(losses)),
                  "validation_macro_f1": score, "epoch_seconds": time.perf_counter() - epoch_start}
        history.append(record)
        write_json(snapshot.with_suffix(".json"), {**record, "model_sha256": file_sha256(snapshot)})
        write_json(trial / "history.json", {"epochs": history, "best_validation_macro_f1": max(best_score, score),
                                           "batch_size": batch_size, "accumulation_steps": accumulation,
                                           "precision": config["precision"], "optimizer_state_restored": False})
        if score > best_score:
            best_epoch, best_score, best_snapshot = global_epoch, score, snapshot
        progress("epoch_complete", epoch=global_epoch, validation_macro_f1=score,
                 best_epoch=best_epoch, best_validation_macro_f1=best_score)
        print(f"Epoch {global_epoch}: validation macro F1 {score:.4f}; best epoch {best_epoch}", flush=True)

    selection = {"original_best_epoch": 4, "original_validation_macro_f1": prior_scores[3],
                 "selected_epoch": best_epoch, "selected_validation_macro_f1": best_score,
                 "validation_improvement": best_score - prior_scores[3],
                 "optimizer_state_restored": False, "loss_scaler_state_restored": False,
                 "rng_state_restored": False, "effective_batch_size": batch_size * accumulation,
                 "source_snapshot_sha256": start_digest, "data_hashes": data_hashes,
                 "question_sha256": sha256_json(questions),
                 "status": "new_checkpoint_selected" if best_epoch > 4 else "original_checkpoint_retained"}
    if best_epoch > 4:
        progress("calibrating", selected_epoch=best_epoch)
        model.load_state_dict(load_file(str(best_snapshot)), strict=True)
        calibration_logits = _logits(model, cal, tokenizer.pad_token_id, batch_size, device, config["precision"])
        temperature, calibration_nll = _choice_temperature(cal, calibration_logits, len(options))
        checkpoint = trial / "laya_checkpoint"
        checkpoint.mkdir()
        model.encoder.save_pretrained(checkpoint / "encoder", safe_serialization=True)
        tokenizer.save_pretrained(checkpoint / "tokenizer")
        shutil.copy2(best_snapshot, checkpoint / "model.safetensors")
        model_cfg = dict(agent.cfg)
        model_cfg.pop("temperature_by_options", None)
        model_cfg["temperature"] = [temperature, 1.0, 1.0]
        write_json(checkpoint / "rl_agent_config.json", model_cfg)
        meta = {"base_repository": config["base_repository"], "base_subfolder": config["base_subfolder"],
                "base_revision": resolved, "laya_version": version("laya"),
                "data_metadata": str(data_dir / "metadata.json"), "data_hashes": data_hashes,
                "best_validation_macro_f1": best_score, "best_epoch": best_epoch,
                "calibration_temperature": temperature, "calibration_nll": calibration_nll,
                "training_seconds": time.perf_counter() - started,
                "batch_size": batch_size, "accumulation_steps": accumulation,
                "precision": config["precision"], "save_epoch_snapshots": True,
                "epochs": history, "checkpoint_sha256": file_sha256(checkpoint / "model.safetensors"),
                "continuation_from_epoch_04_sha256": start_digest,
                "optimizer_state_restored": False}
        write_json(checkpoint / "study_metadata.json", meta)
        selection["checkpoint_sha256"] = meta["checkpoint_sha256"]
    write_json(selection_path, selection)
    progress("complete", selected_epoch=best_epoch, status=selection["status"])
    return selection
