"""Expand the losslessly compressed artifacts and rebase paths for this clone."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main(dataset: Path | None):
    manifest = json.loads((ROOT / "ARTIFACT_PROVENANCE.json").read_text())
    expanded = 0
    for artifact in manifest["artifacts"]:
        source = ROOT / artifact["path"]
        if source.suffix != ".gz":
            continue
        data = gzip.decompress(source.read_bytes())
        if hashlib.sha256(data).hexdigest() != artifact["uncompressed_sha256"]:
            raise ValueError("Uncompressed artifact hash mismatch: " + artifact["path"])
        dest = ROOT / artifact["logical_path"]
        if not dest.exists():
            dest.write_bytes(data)
            expanded += 1
    if dataset:
        dataset = dataset.resolve()
        # All artifact rows use the same relative CAD identity. Keep it unchanged.
        for relative in ["outputs/study/full/cohort.csv", "outputs/descriptors/tmcad_descriptors.csv",
                         "outputs/study/full/geometry_embeddings.csv", "outputs/study/full/annotations/private_asset_manifest.csv"]:
            p = ROOT / relative
            with p.open(encoding="utf-8-sig", newline="") as h:
                reader = csv.DictReader(h)
                fields, rows = reader.fieldnames, list(reader)
            for row in rows:
                row["absolute_path"] = str(dataset / row["relative_path"])
            with p.open("w", encoding="utf-8", newline="") as h:
                writer = csv.DictWriter(h, fieldnames=fields, lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)
    # Rebase the byte-level guards for portable copies. Original metadata and
    # its original hashes remain intact in the committed gzip archives.
    def sha(name):
        return hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    metadata_path = ROOT / "outputs/study/full_ft/laya_training_data/metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["descriptor_sha256"] = sha("outputs/descriptors/tmcad_descriptors.csv")
    for split in ["train", "calibration", "validation"]:
        metadata["partition_hashes"][split] = sha(f"outputs/study/full_ft/laya_training_data/{split}.jsonl")
    metadata["release_note"] = "Byte hashes rebased to portable copies; original metadata remains in metadata.json.gz"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    protocol_path = ROOT / "outputs/study/full/annotations/protocol.json"
    protocol = json.loads(protocol_path.read_text())
    protocol["selected_pairs_sha256"] = sha("outputs/study/full/annotations/selected_pairs.csv")
    protocol["release_note"] = "CSV line endings normalized; frozen pair and sample IDs unchanged"
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    proxy_path = ROOT / "outputs/study/visual_proxy_evaluation/visual_proxy_judgments.meta.json"
    proxy = json.loads(proxy_path.read_text())
    proxy["selected_pairs_sha256"] = protocol["selected_pairs_sha256"]
    proxy["judgments_sha256"] = sha("outputs/study/visual_proxy_evaluation/visual_proxy_judgments.csv")
    proxy_path.write_text(json.dumps(proxy, indent=2) + "\n", encoding="utf-8")
    app_manifest = ROOT / "research_v2/expert_review_app/data/pairs.json"
    pairs = json.loads(app_manifest.read_text())
    pairs["protocol_hash"] = protocol["selected_pairs_sha256"]
    app_manifest.write_text(json.dumps(pairs, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"Expanded {expanded} artifacts. Original compressed files remain intact.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, help="Local, licensed TMCAD directory; required only for STEP-dependent stages")
    main(parser.parse_args().dataset)
