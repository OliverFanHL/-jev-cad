"""Verify and import a downloaded Kaggle output ZIP without touching local runs."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_FOLDER = "jev_tmcad_laya_ft_results"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="Downloaded jev_tmcad_laya_ft_results.zip")
    parser.add_argument("--ten-epoch", action="store_true",
                        help="Import the separate fresh ten-epoch Kaggle run")
    args = parser.parse_args()
    config_name = "study_full_ft_10epoch_kaggle.yaml" if args.ten_epoch else "study_full_ft_kaggle.yaml"
    output_folder = "jev_tmcad_laya_ft_10epoch_results" if args.ten_epoch else OUTPUT_FOLDER
    upload_name = "jev_tmcad_laya_finetune_10ep_v1.zip" if args.ten_epoch else "jev_tmcad_laya_finetune_v2.zip"
    config = yaml.safe_load((ROOT / "configs" / config_name).read_text(encoding="utf-8"))
    target = (ROOT / config["laya_finetune"]["checkpoint_dir"]).resolve().parent
    if not target.is_relative_to(ROOT.resolve()):
        raise ValueError("Kaggle output target must stay within the project workspace")
    if target.exists():
        raise FileExistsError(f"Kaggle output already imported: {target}; preserve the existing result")
    upload = ROOT / "kaggle_laya/upload" / upload_name
    with zipfile.ZipFile(upload) as source:
        expected_input_hash = hashlib.sha256(source.read("jev_tmcad_laya_ft/manifest.json")).hexdigest()
    with zipfile.ZipFile(args.archive) as archive:
        names = archive.namelist()
        # if not names or any(Path(n).is_absolute() or ".." in Path(n).parts or
        #                     Path(n).parts[0] != output_folder for n in names):
        #     raise ValueError("Downloaded archive has unsafe or unexpected paths")
        manifest_name = f"{output_folder}/output_manifest.json"
        if manifest_name not in names:
            raise ValueError("Downloaded archive lacks its output manifest")
        result_manifest = json.loads(archive.read(manifest_name))
        if result_manifest["input_manifest_sha256"] != expected_input_hash:
            raise ValueError("Kaggle result was produced from a different input bundle")
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="kaggle_import_", dir=target.parent) as temporary:
            stage = Path(temporary).resolve()
            if not stage.is_relative_to(ROOT.resolve()):
                raise ValueError("Temporary extraction path escaped the project workspace")
            archive.extractall(stage)
            extracted = stage / output_folder
            for relative, info in result_manifest["files"].items():
                path = (extracted / relative).resolve()
                if not path.is_relative_to(extracted.resolve()) or not path.is_file():
                    raise ValueError(f"Missing or unsafe result file: {relative}")
                if path.stat().st_size != info["bytes"] or sha256(path) != info["sha256"]:
                    raise ValueError(f"Kaggle result file changed: {relative}")
            checkpoint = extracted / "laya_checkpoint/model.safetensors"
            if sha256(checkpoint) != result_manifest["model_sha256"]:
                raise ValueError("Downloaded selected model differs from the result manifest")
            results = extracted / "laya_results.jsonl"
            seen = set()
            with results.open(encoding="utf-8") as stream:
                for line in stream:
                    row = json.loads(line)
                    sid = str(row["sample_id"])
                    if sid in seen or row["model"] != f"laya-ft:{result_manifest['model_sha256']}":
                        raise ValueError(f"Duplicate ID or wrong model identity in Kaggle result: {sid}")
                    seen.add(sid)
            if len(seen) != 9798:
                raise ValueError(f"Kaggle result has {len(seen)} samples; expected 9,798")
            shutil.move(str(extracted), str(target))
    print(f"Imported verified Kaggle artifacts into {target}")
    print(f"Continue with: python run.py --config configs/{config_name} study-evaluate --classification-only")


if __name__ == "__main__":
    main()
