import numpy as np
import pandas as pd

from jev_tmcad.mesh_data import MESH_VERSION, _file_digest, build_mesh_cache
from jev_tmcad.utils import sha256_json


def test_parallel_mesh_run_reuses_existing_cache(tmp_path):
    output_dir = tmp_path / "mesh"
    output_dir.mkdir()
    rows = []
    cfg = {"version": MESH_VERSION, "points": 32, "neighbors": 2,
           "tolerance": 0.003, "seed": 42}
    for number in range(3):
        sample_id = f"{number + 1:020x}"
        step_path = tmp_path / f"{sample_id}.stp"
        step_path.write_bytes(b"cached fixture")
        key = sha256_json({"source_sha256": _file_digest(step_path), "config": cfg})
        np.savez_compressed(output_dir / f"{sample_id}.npz",
                            features=np.zeros((32, 6), dtype=np.float32),
                            knn=np.zeros((32, 2), dtype=np.int64), cache_key=key)
        rows.append({"sample_id": sample_id, "absolute_path": str(step_path),
                     "split": "train", "label": "gear"})
    cohort = pd.DataFrame(rows)
    result = build_mesh_cache(cohort, output_dir, points=32, neighbors=2,
                              tolerance=0.003, seed=42, workers=2, checkpoint_every=1)
    assert len(result) == len(cohort)
    assert result["status"].eq("cached").all()
    assert pd.read_csv(output_dir / "manifest.csv")["sample_id"].nunique() == len(cohort)
