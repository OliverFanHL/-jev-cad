"""Create a small, checked Kaggle Dataset ZIP from the existing frozen study.

Run locally. No STEP/mesh files, API credentials, or existing result records are
included. The archive contains exact train/calibration/validation examples,
all-sample descriptor-derived inference states, code, questions and config.
"""
from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from jev_tmcad.config import load_questions  # noqa: E402
from jev_tmcad.semantic_state import QuantileSchema  # noqa: E402
from jev_tmcad.data.cohort import cohort_hash, load_cohort  # noqa: E402
from jev_tmcad.training.laya import _rows  # noqa: E402
from jev_tmcad.utils import sha256_json  # noqa: E402


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    local_cfg = yaml.safe_load((ROOT / "configs/study_full_ft.yaml").read_text(encoding="utf-8"))
    study = local_cfg["study"]
    directory = ROOT / local_cfg["laya_finetune"]["data_dir"]
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    cohort = load_cohort(ROOT / study["cohort_path"])
    descriptor_path = ROOT / study["descriptor_path"]
    if metadata["cohort_hash"] != cohort_hash(cohort):
        raise ValueError("Training data and cohort hashes differ")
    if metadata["descriptor_sha256"] != sha256(descriptor_path.read_bytes()):
        raise ValueError("Descriptor cache changed since fine-tuning preparation")
    questions_path = ROOT / study["questions_path"]
    spec = load_questions(questions_path)
    if metadata["question_sha256"] != sha256_json(spec):
        raise ValueError("Question protocol changed since fine-tuning preparation")
    schema = QuantileSchema.from_dict(json.loads((ROOT / study["schema_path"]).read_text(encoding="utf-8")))
    prepared: dict[str, dict] = {}
    for split_name, frozen_split in (("train", "train"), ("calibration", "train"), ("validation", "val")):
        path = directory / f"{split_name}.jsonl"
        if metadata["partition_hashes"][split_name] != sha256(path.read_bytes()):
            raise ValueError(f"Prepared {split_name} records changed")
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            sid = str(row["sample_id"])
            if sid in prepared or row["split"] != frozen_split or row["partition"] != split_name:
                raise ValueError(f"Invalid or duplicate prepared {split_name} sample {sid}")
            prepared[sid] = row
    if len(prepared) != metadata["counts"]["train"] + metadata["counts"]["calibration"] + metadata["counts"]["validation"]:
        raise ValueError("Prepared data count differs from frozen metadata")
    state_rows = list(_rows(cohort, descriptor_path, schema, metadata["state_variant"],
                            metadata["numeric"], metadata["profiles"]))
    if len(state_rows) != len(cohort):
        raise ValueError("Inference state coverage differs from the full cohort")
    for row in state_rows:
        prior = prepared.get(row["sample_id"])
        if prior is not None and any(prior[key] != row[key] for key in
                                     ("split", "label", "state_hash", "state")):
            raise ValueError(f"Inference state differs from prepared training/validation state: {row['sample_id']}")
    old_models = pd.read_json(ROOT / study["laya_results_path"], lines=True, nrows=1)
    base_model = str(old_models.iloc[0]["model"])
    kaggle_cfg = yaml.safe_load((ROOT / "kaggle/laya/kaggle_config.yaml").read_text(encoding="utf-8"))
    expected_model = f"{kaggle_cfg['base_repository']}/{kaggle_cfg['base_subfolder']}@{kaggle_cfg['base_revision']}"
    if base_model != expected_model:
        raise ValueError(f"Pinned Kaggle base checkpoint {expected_model} differs from historical Laya {base_model}")

    files: dict[str, bytes] = {}
    prefix = "jev_tmcad_laya_ft/"
    def add(name: str, data: bytes) -> None:
        files[prefix + name] = data

    for name in ("train.jsonl", "calibration.jsonl", "validation.jsonl", "metadata.json"):
        add(f"data/{name}", (directory / name).read_bytes())
    encoded_states = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                             for row in state_rows).encode("utf-8")
    add("data/inference_states.jsonl", encoded_states)
    add("config/questions_compact.yaml", questions_path.read_bytes())
    add("config/kaggle_config.yaml", (ROOT / "kaggle/laya/kaggle_config.yaml").read_bytes())
    add("config/local_study_full_ft.yaml", (ROOT / "configs/study_full_ft.yaml").read_bytes())
    add("config/requirements_kaggle.txt", (ROOT / "kaggle/laya/requirements_kaggle.txt").read_bytes())
    for source in sorted((ROOT / "src/jev_tmcad").rglob("*.py")):
        add(f"code/src/jev_tmcad/{source.relative_to(ROOT / 'src/jev_tmcad').as_posix()}", source.read_bytes())
    manifest = {
        "bundle_type": "jev_tmcad_laya_finetune_v2",
        "cohort_hash": metadata["cohort_hash"],
        "question_sha256": metadata["question_sha256"],
        "base_model": base_model,
        "n_samples": len(state_rows),
        "split_counts": {s: int(n) for s, n in cohort.groupby("split").size().items()},
        "training_partition_counts": metadata["counts"],
        "source_metadata": "Opaque IDs preserve local source paths and labels; keep the Kaggle Dataset private",
        "files": {name.removeprefix(prefix): {"sha256": sha256(data), "bytes": len(data)}
                  for name, data in sorted(files.items())},
    }
    add("manifest.json", json.dumps(manifest, indent=2).encode("utf-8"))
    target = ROOT / "kaggle/laya/upload/jev_tmcad_laya_finetune_v2.zip"
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, data in sorted(files.items()):
            archive.writestr(name, data)
    with zipfile.ZipFile(target) as archive:
        if archive.testzip() is not None:
            raise ValueError("Kaggle upload ZIP failed its CRC check")
        saved = json.loads(archive.read(prefix + "manifest.json"))
        for name, info in saved["files"].items():
            if sha256(archive.read(prefix + name)) != info["sha256"]:
                raise ValueError(f"Kaggle archive hash mismatch: {name}")
    print(json.dumps({"upload_zip": str(target), "archive_bytes": target.stat().st_size,
                      "samples": len(state_rows), "training_partitions": metadata["counts"]}, indent=2))


if __name__ == "__main__":
    main()
