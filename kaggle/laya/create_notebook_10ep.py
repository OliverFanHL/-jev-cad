"""Generate the independent, fresh ten-epoch Kaggle study notebook."""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
original = json.loads((HERE / "laya_finetune_kaggle.ipynb").read_text(encoding="utf-8"))
source = original["cells"]


def cell_text(cell: dict) -> str:
    return "".join(cell["source"])


def replace_cell(cell: dict, text: str) -> dict:
    return {**cell, "execution_count": None, "outputs": [],
            "source": text.splitlines(keepends=True)} if cell["cell_type"] == "code" else {
                **cell, "source": text.splitlines(keepends=True)}


cells = [source[i] for i in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)]
cells[0] = replace_cell(cells[0], """# Fresh ten-epoch Laya fine-tuning on TMCAD

Start from the pinned base Laya checkpoint and train for up to ten epochs using the prepared CAD states. The RL noise decreases from 0.4 to 0.1 across the run. Validation macro F1 selects the checkpoint, and the calibration partition fits the Choice temperature.

Upload `jev_tmcad_laya_finetune_10ep_v1.zip` as a private Kaggle Dataset. Attach it under Add Input, select a GPU, enable Internet for dependencies and weights, and run the cells.

Each epoch saves a weight snapshot and validation record. The output ZIP contains the selected Laya checkpoint, inference records, and epoch snapshots. Allow storage for both the snapshots and the ZIP.
""")

input_code = cell_text(cells[1]).replace(
    "jev_tmcad_laya_finetune_v2.zip", "jev_tmcad_laya_finetune_10ep_v1.zip")
cells[1] = replace_cell(cells[1], input_code)

setup = cell_text(cells[2]).replace(
    "manifest['bundle_type'] == 'jev_tmcad_laya_finetune_v2'", 
    "manifest['bundle_type'] == 'jev_tmcad_laya_finetune_10ep_v1'")
setup = setup.replace("'Attach the corrected v2 upload ZIP'", "'Attach the ten-epoch v1 upload ZIP'")
setup = setup.replace("if '_validation_macro_f1' not in source_file.read_text(encoding='utf-8'):",
                      "if not {'_validation_macro_f1', '_calibrate_choice_temperature'}.issubset(set(source_file.read_text(encoding='utf-8').split())):")
# Check substrings, not whitespace tokens; function names appear next to punctuation.
setup = setup.replace(
    "if not {'_validation_macro_f1', '_calibrate_choice_temperature'}.issubset(set(source_file.read_text(encoding='utf-8').split())):",
    "if any(name not in source_file.read_text(encoding='utf-8') for name in ('_validation_macro_f1', '_calibrate_choice_temperature')):")
setup = setup.replace("Attach the v2 ZIP.", "Attach the ten-epoch ZIP.")
setup = setup.replace("output = Path('/kaggle/working/jev_tmcad_laya_ft_results')",
                      "output = Path('/kaggle/working/jev_tmcad_laya_ft_10epoch_results')")
setup = setup.replace("assert manifest['bundle_type'] == 'jev_tmcad_laya_finetune_10ep_v1', 'Attach the ten-epoch v1 upload ZIP'",
                      "assert manifest['bundle_type'] == 'jev_tmcad_laya_finetune_10ep_v1' and manifest['training_epochs'] == 10, 'Attach the ten-epoch v1 upload ZIP'")
setup += "\nassert int(config['epochs']) == 10 and config['save_epoch_snapshots'] is True\n"
cells[2] = replace_cell(cells[2], setup)

cells[3] = replace_cell(cells[3], """## Train, validate, and preserve every epoch

Training starts at the pinned base model with a fresh optimizer. The ten epoch weights and scores are saved before the final validation-selected model is calibrated and exported. The progress JSON records the current batch and stage. A recovery cell completes checkpoint export when the ten saved epoch records are available.
""")
train = cell_text(cells[5])
train = train.replace("if not (checkpoint / 'study_metadata.json').exists():", 
                      "if not (checkpoint / 'study_metadata.json').exists():")
cells[5] = replace_cell(cells[5], train)

verification = {
    "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
    "source": """from jev_tmcad.data.annotations import file_sha256
if len(training['epochs']) != 10 or [row['epoch'] for row in training['epochs']] != list(range(1, 11)):
    raise ValueError('Ten completed epoch records are required before inference')
snapshot_dir = output / 'laya_epoch_checkpoints'
for row in training['epochs']:
    snapshot = snapshot_dir / f"epoch_{row['epoch']:02d}.safetensors"
    summary = json.loads(snapshot.with_suffix('.json').read_text(encoding='utf-8'))
    if (summary['epoch'] != row['epoch'] or
            summary['validation_macro_f1'] != row['validation_macro_f1'] or
            summary['model_sha256'] != file_sha256(snapshot)):
        raise ValueError(f"Saved epoch {row['epoch']} is missing or differs from validation history")
expected_best = 1 + max(range(10), key=lambda i: training['epochs'][i]['validation_macro_f1'])
if training['best_epoch'] != expected_best or training['checkpoint_sha256'] != file_sha256(checkpoint / 'model.safetensors'):
    raise ValueError('Selected best checkpoint differs from validated training history')
print('Verified all ten epoch models and best selected epoch:', expected_best)
""".splitlines(keepends=True), "id": "tmcad10-06"}
cells.insert(6, verification)

export = cell_text(cells[-1])
export = export.replace("jev_tmcad_laya_ft_results.zip", "jev_tmcad_laya_ft_10epoch_results.zip")
export = export.replace("with zipfile.ZipFile(archive_path, 'w', compression=zipfile.ZIP_STORED, allowZip64=True) as archive:",
                        "with zipfile.ZipFile(archive_path, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as archive:")
export = export.replace("readme = '''Kaggle Laya/TMCAD output", "readme = '''Fresh ten-epoch Kaggle Laya/TMCAD output")
export = export.replace("training logs.\\n'", "training logs; all ten epochs are preserved.\\n'")
old_archive = """with zipfile.ZipFile(archive_path, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as archive:
    for path in sorted(output.rglob('*')):
        if path.is_file():
            archive.write(path, arcname=str(Path(output.name) / path.relative_to(output)))
print('Download:', archive_path)
print('Output folder:', output)"""
new_archive = """# A second copy of all large epoch weights is needed while building the ZIP.
# Reserve storage for archive creation.
archive_bytes_upper_bound = sum(path.stat().st_size for path in output.rglob('*') if path.is_file())
free_bytes = shutil.disk_usage('/kaggle/working').free
if free_bytes < archive_bytes_upper_bound + 1024**3:
    print('Insufficient free space for a safe all-files ZIP; models and reports remain in:', output)
    print('Free bytes:', free_bytes, '| estimated uncompressed archive bytes:', archive_bytes_upper_bound)
else:
    with zipfile.ZipFile(archive_path, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as archive:
        for path in sorted(output.rglob('*')):
            if path.is_file():
                archive.write(path, arcname=str(Path(output.name) / path.relative_to(output)))
    print('Download:', archive_path)
print('Output folder:', output)"""
if old_archive not in export:
    raise ValueError("Expected original archive cell was not found")
export = export.replace(old_archive, new_archive)
export += "\nprint('Selected best epoch:', training['best_epoch'], 'of 10')\n"
cells[-1] = replace_cell(cells[-1], export)

for number, cell in enumerate(cells, 1):
    cell["id"] = f"tmcad10-{number:02d}"
    if cell["cell_type"] == "code":
        cell["execution_count"] = None
        cell["outputs"] = []

notebook = {"cells": cells, "metadata": original["metadata"],
            "nbformat": 4, "nbformat_minor": 5}
target = HERE / "laya_finetune_10epoch_kaggle.ipynb"
target.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(target)
