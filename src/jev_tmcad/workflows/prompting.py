"""Matched zero-shot and few-shot Jev acquisition and feature alignment."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from jev_tmcad.config import Config, load_questions
from jev_tmcad.data.cohort import load_cohort, valid_descriptors
from jev_tmcad.data.dataset import TMCAD_CLASSES
from jev_tmcad.models.jev import SQLiteResponseCache, _call_typesafe
from jev_tmcad.models.prompting import demonstration, derive_rubrics, make_spec, state_of
from jev_tmcad.utils import append_jsonl, read_jsonl, sha256_json


def build_prompt_requests(frame, questions, *, shots=(0, 4), seed=42,
                          model="jev-latest", exact_groups=None):
    """Draw balanced training supports and replace query-equivalent examples."""
    if not shots or len(set(shots)) != len(shots) or any(k < 0 or int(k) != k for k in shots):
        raise ValueError("Shot counts must be distinct nonnegative integers")
    if frame["sample_id"].duplicated().any():
        raise ValueError("Prompt inputs require unique sample IDs")
    work = frame.copy().reset_index(drop=True)
    states = {str(row.sample_id): state_of(row) for _, row in work.iterrows()}
    work["state_hash"] = [sha256_json(states[str(sid)]) for sid in work.sample_id]
    work["exact_group"] = [str((exact_groups or {}).get(str(sid), sid)) for sid in work.sample_id]
    train = work[work.split.eq("train")]
    rubrics = derive_rubrics(train)
    spec = make_spec(questions, rubrics)
    bad_states = train.groupby("state_hash").label.nunique().loc[lambda x: x > 1].index
    bad_files = train.groupby("exact_group").label.nunique().loc[lambda x: x > 1].index
    support = train[~train.state_hash.isin(bad_states) & ~train.exact_group.isin(bad_files)]
    support = support.sort_values("sample_id").drop_duplicates("state_hash").drop_duplicates("exact_group")
    rng = np.random.default_rng(seed)
    pools = {label: group.iloc[rng.permutation(len(group))]
             for label, group in support.groupby("label", sort=True)}
    for _, query in work.iterrows():
        for k in shots:
            selected = []
            for label in TMCAD_CLASSES:
                if k == 0:
                    continue
                pool = pools.get(label, support.iloc[:0])
                eligible = pool[pool.state_hash.ne(query.state_hash) & pool.exact_group.ne(query.exact_group)]
                if len(eligible) < k:
                    raise ValueError(f"Insufficient training demonstrations for {label}: need {k}")
                selected.extend(eligible.iloc[:k].to_dict("records"))
            state = {
                "task": "Answer the typed questions for query only. labeled_examples are correctly labeled training demonstrations, not additional queries.",
                "labeled_examples": [demonstration(pd.Series(row), rubrics) for row in selected],
                "query": {"geometry_state": states[str(query.sample_id)]},
            }
            payload = {"model": model, "state": state, "questions": spec}
            yield {"sample_id": str(query.sample_id), "split": str(query.split),
                   "shots_per_class": int(k), "support_ids": [str(r["sample_id"]) for r in selected],
                   "request_hash": sha256_json(payload), "payload": payload}


def prompt_feature_frame(requests, records):
    """Validate saved prompt identities and expose Choice features by condition."""
    indexed = {}
    models = set()
    for record in records:
        key = (str(record["sample_id"]), record["shots_per_class"])
        if key in indexed:
            raise ValueError(f"Duplicate prompted response: {key}")
        indexed[key] = record
        models.add(record.get("model"))
    if len(models) != 1 or not next(iter(models)):
        raise ValueError("Prompt responses require one resolved model identity")
    rows = {}
    expected = set()
    for request in requests:
        sid, k = request["sample_id"], request["shots_per_class"]
        expected.add((sid, k))
        record = indexed.get((sid, k))
        if record is None or record["request_hash"] != request["request_hash"] or record["split"] != request["split"]:
            raise ValueError(f"Missing or incompatible prompted response: {sid}/{k}")
        probs = record["class_probabilities"]
        values = np.array([probs.get(c, np.nan) for c in TMCAD_CLASSES], dtype=float)
        if set(probs) != set(TMCAD_CLASSES) or not np.isfinite(values).all() or (values < 0).any() or not np.isclose(values.sum(), 1, atol=1e-6):
            raise ValueError(f"Invalid prompted Choice probabilities: {sid}/{k}")
        prefix = f"jev_fs{k}"
        row = rows.setdefault(sid, {"sample_id": sid})
        row[f"{prefix}__choice"] = TMCAD_CLASSES[int(values.argmax())]
        row[f"{prefix}__request_hash"] = request["request_hash"]
        row[f"{prefix}__model"] = record["model"]
        row.update({f"{prefix}__prob__{c}": float(v) for c, v in zip(TMCAD_CLASSES, values)})
    if set(indexed) != expected:
        raise ValueError("Prompt response identities differ from the configured requests")
    return pd.DataFrame(rows.values())


def requests_from_config(cfg):
    def path(key):
        value = Path(cfg.get(key))
        return value if value.is_absolute() else cfg.base_dir / value
    cohort = load_cohort(path("study.cohort_path"))
    frame = cohort[["sample_id"]].merge(valid_descriptors(path("study.descriptor_path")),
                                         on="sample_id", validate="one_to_one")
    if len(frame) != len(cohort):
        raise ValueError("Prompt descriptors must cover the cohort")
    groups = {}
    if cfg.get("study.duplicate_audit_path"):
        audit = json.loads(path("study.duplicate_audit_path").read_text(encoding="utf-8"))
        groups = {str(s["sample_id"]): g["sha256"] for g in audit["groups"] for s in g["samples"]}
    return build_prompt_requests(frame, load_questions(path("study.questions_path")),
                                  shots=cfg.get("prompting.shots_per_class", [0, 4]),
                                  seed=int(cfg.get("project.seed", 42)),
                                  model=cfg.get("jev.model", "jev-latest"), exact_groups=groups)


def cmd_prompt(args):
    cfg = Config.load(args.config)
    load_dotenv(cfg.base_dir / ".env")
    output = cfg.base_dir / cfg.get("prompting.output_dir", "outputs/study/prompting")
    output.mkdir(parents=True, exist_ok=True)
    results = output / "responses.jsonl"
    existing = read_jsonl(results)
    saved = {(str(r["sample_id"]), r["request_hash"]) for r in existing}
    current = {(str(r["sample_id"]), r["shots_per_class"]): r["request_hash"] for r in existing}
    cache = SQLiteResponseCache(output / "cache.sqlite")
    planned = acquired = 0
    try:
        preview = output / "requests.jsonl"
        if args.dry_run:
            preview.write_text("", encoding="utf-8")
        for request in requests_from_config(cfg):
            planned += 1
            key = (request["sample_id"], request["shots_per_class"])
            if key in current and current[key] != request["request_hash"]:
                raise ValueError("Prompt inputs changed; choose a new output directory")
            if args.dry_run:
                append_jsonl(preview, request)
                continue
            if (request["sample_id"], request["request_hash"]) in saved:
                continue
            digest = request["request_hash"]
            answer = cache.get(digest)
            if answer is None:
                payload = request["payload"]
                answer = _call_typesafe(payload["state"], payload["questions"], payload["model"],
                                         int(cfg.get("jev.retry_attempts", 5)),
                                         float(cfg.get("jev.retry_base_seconds", 1.5)),
                                         float(cfg.get("jev.timeout_seconds", 30)))
                cache.put(digest, model=payload["model"], response=answer)
                acquired += 1
            append_jsonl(results, {k: v for k, v in request.items() if k != "payload"} | answer)
    finally:
        cache.close()
    print(json.dumps({"planned": planned, "new_calls": acquired, "output_dir": str(output)}))


def register_prompt_commands(sub):
    parser = sub.add_parser("study-prompt", help="Collect matched zero-shot and few-shot Jev responses")
    parser.add_argument("--dry-run", action="store_true", help="Save request previews")
    parser.set_defaults(func=cmd_prompt)
