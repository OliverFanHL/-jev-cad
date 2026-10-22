"""Build the frozen, private Kaggle CLM input ZIP without model inference."""
from __future__ import annotations

import csv
import hashlib
import json
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PREFIX = "jev_tmcad_clm/"
BASELINE_METHODS = ("descriptor_plus_geometry", "jev_direct_choice",
                    "jev_all_questions", "geometry_plus_jev", "laya_ft_direct_choice")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> Path:
    cohort_path = ROOT / "outputs/study/full/cohort.csv"
    rows = list(csv.DictReader(cohort_path.open(newline="", encoding="utf-8")))
    if len(rows) != 9798 or len({r["sample_id"] for r in rows}) != len(rows):
        raise ValueError("Frozen cohort size or sample IDs changed")
    identity = [dict((key, r[key]) for key in ("sample_id", "split", "label", "relative_path"))
                for r in sorted(rows, key=lambda item: item["sample_id"])]
    cohort_hash = digest(json.dumps(identity, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False).encode("utf-8"))
    recorded = json.loads((ROOT / "outputs/study/full/cohort.meta.json").read_text(encoding="utf-8"))
    if cohort_hash != recorded["cohort_hash"]:
        raise ValueError("Frozen cohort hash differs from its metadata")
    geometry_meta = json.loads((ROOT / "outputs/study/full/geometry_embeddings.meta.json").read_text(encoding="utf-8"))
    if geometry_meta["cohort_hash"] != cohort_hash:
        raise ValueError("Geometry embeddings belong to another cohort")
    checkpoint = ROOT / "outputs/study/full/geometry_encoder.pt"
    if digest(checkpoint.read_bytes()) != geometry_meta["checkpoint_sha256"]:
        raise ValueError("Geometry checkpoint changed after embedding extraction")
    files: dict[str, bytes] = {}

    def add(name: str, path: Path) -> None:
        files[name] = path.read_bytes()

    for name, relative in {
        "data/cohort.csv": "outputs/study/full/cohort.csv",
        "data/cohort.meta.json": "outputs/study/full/cohort.meta.json",
        "data/descriptors.csv": "outputs/descriptors/tmcad_descriptors.csv",
        "data/schema.json": "outputs/descriptors/quantile_schema.json",
        "data/geometry_embeddings.csv": "outputs/study/full/geometry_embeddings.csv",
        "data/geometry_embeddings.meta.json": "outputs/study/full/geometry_embeddings.meta.json",
        "data/geometry_encoder.pt": "outputs/study/full/geometry_encoder.pt",
        "config/questions_compact.yaml": "configs/questions_compact.yaml",
        "config/study_clm.yaml": "kaggle/clm/study_clm_vllm.yaml",
        "baseline/evaluation/classification/summary.json": "analysis/evaluation/classification/summary.json",
        "baseline/duplicate_audit.json": "outputs/study/full/evaluation/duplicate_audit.json",
        "baseline/qualitative_examples.json": "analysis/evidence/qualitative_examples.json",
    }.items():
        add(name, ROOT / relative)
    for method in BASELINE_METHODS:
        add(f"baseline/evaluation/classification/{method}.csv",
            ROOT / f"analysis/evaluation/classification/{method}.csv")
    for source in sorted((ROOT / "src/jev_tmcad").rglob("*.py")):
        add(f"code/src/jev_tmcad/{source.relative_to(ROOT / 'src/jev_tmcad').as_posix()}", source)
    add("code/build_clm_evidence.py", ROOT / "analysis/scripts/build_clm_evidence.py")
    manifest = {
        "bundle_type": "jev_tmcad_clm_v1", "cohort_hash": cohort_hash,
        "n_samples": len(rows),
        "split_counts": {split: sum(r["split"] == split for r in rows)
                         for split in ("train", "val", "test")},
        "questions_sha256": digest(files["config/questions_compact.yaml"]),
        "source_metadata": "Contains original paths and labels; upload as a private Kaggle Dataset",
        "files": {name: {"sha256": digest(data), "bytes": len(data)}
                  for name, data in sorted(files.items())},
    }
    files["manifest.json"] = json.dumps(manifest, indent=2).encode("utf-8")
    target = HERE / "upload/jev_tmcad_clm_v1.zip"
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED, compresslevel=6,
                         allowZip64=True) as archive:
        for name, data in sorted(files.items()):
            archive.writestr(PREFIX + name, data)
    with zipfile.ZipFile(target) as archive:
        if archive.testzip() is not None:
            raise ValueError("Kaggle CLM upload ZIP failed CRC validation")
        for name, info in manifest["files"].items():
            data = archive.read(PREFIX + name)
            if len(data) != info["bytes"] or digest(data) != info["sha256"]:
                raise ValueError(f"Kaggle upload member changed: {name}")
    print(f"Prepared {target} ({target.stat().st_size:,} bytes, {len(rows)} frozen samples)")
    return target


if __name__ == "__main__":
    main()
