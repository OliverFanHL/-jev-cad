"""Verify and import a completed Kaggle CLM result ZIP for later integration."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PREFIX = "jev_tmcad_clm_results/"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def import_zip(path: Path, target: Path) -> None:
    if target.exists():
        raise FileExistsError(f"Import destination already exists: {target}")
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if any(not name.startswith(PREFIX) or ".." in Path(name).parts or
               Path(name).is_absolute() for name in names):
            raise ValueError("Unsafe Kaggle CLM result ZIP paths")
        manifest = json.loads(archive.read(PREFIX + "output_manifest.json"))
        if manifest["pending"] != 0 or manifest["completed"] != 9798:
            raise ValueError("CLM output ZIP is incomplete; attach it to another Kaggle session")
        input_manifest = archive.read(PREFIX + "input_manifest.json")
        if sha256(input_manifest) != manifest["input_manifest_sha256"]:
            raise ValueError("CLM output input manifest changed")
        expected_input = json.loads(input_manifest)
        if expected_input["bundle_type"] != "jev_tmcad_clm_v1":
            raise ValueError("Unexpected CLM input bundle")
        if set(names) != {PREFIX + name for name in manifest["files"]} | {PREFIX + "output_manifest.json"}:
            raise ValueError("CLM output ZIP member set differs from manifest")
        for name, info in manifest["files"].items():
            data = archive.read(PREFIX + name)
            if len(data) != info["bytes"] or sha256(data) != info["sha256"]:
                raise ValueError(f"CLM output changed: {name}")
        raw = archive.read(PREFIX + "clm_results.jsonl").decode("utf-8")
        records = [json.loads(line) for line in raw.splitlines() if line]
        if len(records) != 9798 or len({str(r["sample_id"]) for r in records}) != len(records):
            raise ValueError("CLM response IDs are incomplete or duplicated")
        with (ROOT / "outputs/study/full/cohort.csv").open(newline="", encoding="utf-8") as stream:
            cohort = {r["sample_id"]: r for r in csv.DictReader(stream)}
        if set(cohort) != {str(r["sample_id"]) for r in records}:
            raise ValueError("Kaggle CLM sample IDs differ from local frozen cohort")
        for record in records:
            row = cohort[str(record["sample_id"])]
            if (record["split"] != row["split"] or record["true_label"] != row["label"] or
                    record["checkpoint_sha256"] != manifest["checkpoint_sha256"] or
                    record.get("encoder_backend") != "vllm"):
                raise ValueError(f"Kaggle CLM split, label, head, or encoder differs for {record['sample_id']}")
        target.mkdir(parents=True)
        for name in manifest["files"]:
            destination = target / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(archive.read(PREFIX + name))
        (target / "output_manifest.json").write_bytes(archive.read(PREFIX + "output_manifest.json"))
    print(f"Verified and imported {len(records)} CLM records to {target}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("zip", type=Path)
    parser.add_argument("--target", type=Path, default=ROOT / "outputs/study/clm/kaggle_import")
    args = parser.parse_args()
    import_zip(args.zip, args.target)
