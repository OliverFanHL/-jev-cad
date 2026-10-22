"""Input bundles must carry all modules after the package was reorganized."""
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_ten_epoch_bundle_refreshes_nested_package_and_preserves_data(tmp_path, monkeypatch):
    source = ROOT / "kaggle/laya/build_upload_10ep.py"
    spec = importlib.util.spec_from_file_location("build_upload_10ep", source)
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    shutil.copytree(ROOT / "src", tmp_path / "src")
    for relative in ("kaggle/laya/kaggle_config_10ep.yaml", "configs/study_full_ft_10epoch_kaggle.yaml"):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    files = {
        "data/train.jsonl": b'{"fixture": true}\n',
        "config/kaggle_config.yaml": (ROOT / "kaggle/laya/kaggle_config.yaml").read_bytes(),
        "code/src/jev_tmcad/study_laya_finetune.py": b"# earlier package layout\n",
    }
    manifest = {"bundle_type": "jev_tmcad_laya_finetune_v2", "files": {
        name: {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
        for name, data in files.items()
    }}
    input_zip = tmp_path / "base.zip"
    output_zip = tmp_path / "ten-epoch.zip"
    with zipfile.ZipFile(input_zip, "w") as archive:
        for name, data in files.items():
            archive.writestr(builder.PREFIX + name, data)
        archive.writestr(builder.PREFIX + "manifest.json", json.dumps(manifest))
    monkeypatch.setattr(builder, "ROOT", tmp_path)
    monkeypatch.setattr(builder, "SOURCE", input_zip)
    monkeypatch.setattr(builder, "TARGET", output_zip)
    builder.main()
    with zipfile.ZipFile(output_zip) as archive:
        saved = json.loads(archive.read(builder.PREFIX + "manifest.json"))
        assert archive.read(builder.PREFIX + "data/train.jsonl") == files["data/train.jsonl"]
        assert saved["files"]["data/train.jsonl"] == manifest["files"]["data/train.jsonl"]
        assert "code/src/jev_tmcad/study_laya_finetune.py" not in saved["files"]
        expected = {"code/src/jev_tmcad/" + p.relative_to(ROOT / "src/jev_tmcad").as_posix()
                    for p in (ROOT / "src/jev_tmcad").rglob("*.py")}
        assert expected.issubset(saved["files"])
        for name, info in saved["files"].items():
            assert hashlib.sha256(archive.read(builder.PREFIX + name)).hexdigest() == info["sha256"]
        archive.extractall(tmp_path / "unpacked")
    packaged_src = tmp_path / "unpacked" / builder.PREFIX / "code/src"
    # The extracted bundle must import without help from the local source tree.
    subprocess.run([sys.executable, "-I", "-c",
                    "import sys; sys.path.insert(0, sys.argv[1]); "
                    "import jev_tmcad.cli, jev_tmcad.training.laya, jev_tmcad.models.clm",
                    str(packaged_src)], cwd=tmp_path, check=True)
