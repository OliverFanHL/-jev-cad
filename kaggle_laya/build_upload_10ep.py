"""Package the frozen study for a fresh ten-epoch Kaggle Laya run.

The train/calibration/validation and inference states are copied byte for byte
from the verified v2 bundle. Only the epoch setting, current trainer source,
and manifest are updated. The existing four-epoch bundle is not modified.
"""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "jev_tmcad_laya_ft/"
SOURCE = ROOT / "kaggle_laya/upload/jev_tmcad_laya_finetune_v2.zip"
TARGET = ROOT / "kaggle_laya/upload/jev_tmcad_laya_finetune_10ep_v1.zip"


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    with zipfile.ZipFile(SOURCE) as archive:
        files = {name: archive.read(name) for name in archive.namelist()}
    manifest = json.loads(files[PREFIX + "manifest.json"])
    if manifest["bundle_type"] != "jev_tmcad_laya_finetune_v2":
        raise ValueError("The source must be the verified four-epoch v2 bundle")
    original_data_hashes = {name: info["sha256"] for name, info in manifest["files"].items()
                            if name.startswith("data/")}
    config_name = "config/kaggle_config.yaml"
    config = files[PREFIX + config_name].decode("utf-8")
    if config.count("epochs: 4\n") != 1 or "save_epoch_snapshots: true" not in config:
        raise ValueError("Unexpected source training config")
    expected_config = config.replace("epochs: 4\n", "epochs: 10\n")
    new_config = (ROOT / "kaggle_laya/kaggle_config_10ep.yaml").read_text(encoding="utf-8")
    # Ignore only the explanatory first comment; every effective setting must match.
    if new_config.splitlines()[1:] != expected_config.splitlines()[1:]:
        raise ValueError("Ten-epoch config changes another training or evaluation setting")
    files[PREFIX + config_name] = new_config.encode("utf-8")
    local_config_name = "config/local_study_full_ft.yaml"
    files[PREFIX + local_config_name] = (
        ROOT / "configs/study_full_ft_10epoch_kaggle.yaml").read_bytes()
    source_name = "code/src/jev_tmcad/study_laya_finetune.py"
    files[PREFIX + source_name] = (ROOT / "src/jev_tmcad/study_laya_finetune.py").read_bytes()
    if (b"_validation_macro_f1" not in files[PREFIX + source_name] or
            b"_calibrate_choice_temperature" not in files[PREFIX + source_name]):
        raise ValueError("Current trainer lacks the fixed validation or calibration path")
    manifest["bundle_type"] = "jev_tmcad_laya_finetune_10ep_v1"
    manifest["training_epochs"] = 10
    manifest["parent_bundle_type"] = "jev_tmcad_laya_finetune_v2"
    for name in (config_name, local_config_name, source_name):
        data = files[PREFIX + name]
        manifest["files"][name] = {"sha256": sha256(data), "bytes": len(data)}
    if {name: info["sha256"] for name, info in manifest["files"].items()
            if name.startswith("data/")} != original_data_hashes:
        raise ValueError("Frozen data changed while packaging")
    files[PREFIX + "manifest.json"] = json.dumps(manifest, indent=2).encode("utf-8")
    for name, info in manifest["files"].items():
        if sha256(files[PREFIX + name]) != info["sha256"]:
            raise ValueError(f"Bundle member differs from manifest: {name}")
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(TARGET, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, data in sorted(files.items()):
            archive.writestr(name, data)
    with zipfile.ZipFile(TARGET) as archive:
        if archive.testzip() is not None:
            raise ValueError("Ten-epoch upload ZIP failed CRC validation")
    print(f"Created {TARGET} ({TARGET.stat().st_size:,} bytes); frozen data hashes unchanged")


if __name__ == "__main__":
    main()
