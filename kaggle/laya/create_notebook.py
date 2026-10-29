"""Build the self-contained Kaggle notebook without a notebook authoring dependency."""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source.splitlines(keepends=True)}


cells = [
    markdown("""# TMCAD fine-tuning of Laya on Kaggle

This notebook adapts Laya on the prepared CAD states using FP16 and an effective batch of eight. Training fits Choice targets, validation macro F1 selects the checkpoint, and a separate training partition calibrates the Choice temperature.
"""),
    code("""from pathlib import Path
import json, sys, zipfile, subprocess

inputs = Path('/kaggle/input')
matches = [p for p in inputs.rglob('manifest.json') if p.parent.name == 'jev_tmcad_laya_ft']
if not matches:
    zips = list(inputs.rglob('jev_tmcad_laya_finetune_v2.zip'))
    if len(zips) != 1:
        raise FileNotFoundError('Attach exactly one private Dataset containing jev_tmcad_laya_finetune_v2.zip')
    unpacked = Path('/kaggle/working/unpacked_input')
    unpacked.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zips[0]) as archive:
        if any(Path(n).is_absolute() or '..' in Path(n).parts for n in archive.namelist()):
            raise ValueError('Unsafe archive path')
        archive.extractall(unpacked)
    matches = list(unpacked.rglob('manifest.json'))
if len(matches) != 1:
    raise ValueError(f'Expected one study bundle manifest, found {len(matches)}')
bundle = matches[0].parent
print('Input bundle:', bundle)
print('Installing pinned Laya dependencies; Kaggle CUDA PyTorch is retained')
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '-r',
                       str(bundle / 'config/requirements_kaggle.txt')])
"""),
    code("""import hashlib, importlib, platform, shutil, time
import pandas as pd
import torch
import yaml

manifest = json.loads((bundle / 'manifest.json').read_text(encoding='utf-8'))
assert manifest['bundle_type'] == 'jev_tmcad_laya_finetune_v2', 'Attach the corrected v2 upload ZIP'
source_file = bundle / 'code/src/jev_tmcad/training/laya.py'
if '_validation_macro_f1' not in source_file.read_text(encoding='utf-8'):
    raise RuntimeError(f'The attached Dataset still has the old trainer: {source_file}. Attach the v2 ZIP.')
for relative, info in manifest['files'].items():
    path = bundle / relative
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != info['sha256']:
        raise ValueError(f'Bundle file missing or changed: {relative}')
if not torch.cuda.is_available():
    raise RuntimeError('Select a Kaggle GPU accelerator before training')
print('Verified', len(manifest['files']), 'files and', manifest['n_samples'], 'sample states')
print('GPU:', torch.cuda.get_device_name(0), '| Torch:', torch.__version__)
config = yaml.safe_load((bundle / 'config/kaggle_config.yaml').read_text(encoding='utf-8'))
questions = yaml.safe_load((bundle / 'config/questions_compact.yaml').read_text(encoding='utf-8'))
sys.path.insert(0, str(bundle / 'code/src'))
# Rerunning in one Kaggle kernel can retain the old package in Python's cache.
for module_name in list(sys.modules):
    if module_name == 'jev_tmcad' or module_name.startswith('jev_tmcad.'):
        del sys.modules[module_name]
importlib.invalidate_caches()
from jev_tmcad.training.laya import (train_laya_finetune,
    run_finetuned_laya_states, evaluate_finetuned_laya_choice, _validation_macro_f1)
import jev_tmcad.training.laya as trainer_module
if Path(trainer_module.__file__).resolve() != source_file.resolve():
    raise RuntimeError(f'Python imported {trainer_module.__file__}, expected {source_file}')
from jev_tmcad.utils import sha256_json
assert sha256_json(questions) == manifest['question_sha256']
assert f"{config['base_repository']}/{config['base_subfolder']}@{config['base_revision']}" == manifest['base_model']
assert _validation_macro_f1([{'label': 0}, {'label': 1}], [[5, 0], [0, 5]], 2) == 1.0
output = Path('/kaggle/working/jev_tmcad_laya_ft_results')
output.mkdir(parents=True, exist_ok=True)
for name in ('kaggle_config.yaml', 'questions_compact.yaml', 'local_study_full_ft.yaml',
             'requirements_kaggle.txt'):
    shutil.copy2(bundle / 'config' / name, output / name)
shutil.copy2(bundle / 'manifest.json', output / 'input_manifest.json')
shutil.copy2(bundle / 'data/metadata.json', output / 'prepared_data_metadata.json')
environment = {'python': platform.python_version(), 'platform': platform.platform(),
               'torch': torch.__version__, 'cuda': torch.version.cuda,
               'gpu': torch.cuda.get_device_name(0), 'config': config}
(output / 'environment.json').write_text(json.dumps(environment, indent=2), encoding='utf-8')
"""),
    markdown("""## Train and calibrate

Training uses only `train.jsonl`; the train-only calibration groups fit Choice temperature; validation macro F1 selects the epoch. Each epoch produces a weight snapshot and a history update. `laya_checkpoint.progress.json` records the current stage and batch number.

If every epoch is already complete but calibration previously failed, the next cell verifies the saved history and selected snapshot, then finishes calibration and checkpoint export. It performs no training steps. Keep the same Kaggle session and its `/kaggle/working` files. For an existing notebook session, paste `kaggle/laya/recover_from_snapshots.py` into a new cell after setup, run it, and then continue at inference.
"""),
    code((HERE / "recover_from_snapshots.py").read_text(encoding="utf-8")),
    code("""from importlib.metadata import version
environment['laya'] = version('laya')
(output / 'environment.json').write_text(json.dumps(environment, indent=2), encoding='utf-8')
checkpoint = output / 'laya_checkpoint'
if not (checkpoint / 'study_metadata.json').exists():
    training = train_laya_finetune(
        bundle / 'data', checkpoint, questions,
        repository=config['base_repository'], subfolder=config['base_subfolder'],
        revision=config['base_revision'], seed=int(config['project_seed']),
        device=config['device'], epochs=int(config['epochs']),
        batch_size=int(config['batch_size']),
        accumulation_steps=int(config['accumulation_steps']),
        precision=config['precision'],
        save_epoch_snapshots=bool(config['save_epoch_snapshots']),
        encoder_lr=float(config['encoder_lr']), head_lr=float(config['head_lr']),
        weight_decay=float(config['weight_decay']), group_size=int(config['group_size']),
        sigma_start=float(config['sigma_start']), sigma_end=float(config['sigma_end']),
        rl_weight=float(config['rl_weight']))
else:
    training = json.loads((checkpoint / 'study_metadata.json').read_text(encoding='utf-8'))
pd.DataFrame(training['epochs']).to_csv(output / 'training_history.csv', index=False)
print('Best validation macro F1:', training['best_validation_macro_f1'],
      'at epoch', training['best_epoch'])
"""),
    markdown("""## Run the selected model on the matched cohort

The following stage uses the frozen per-sample compact states and the unchanged Choice/Noul question specification. It writes the JSONL record format expected by local `study-evaluate`. Existing complete records are reused by request hash, and the file is appended incrementally.
"""),
    code("""states = bundle / 'data/inference_states.jsonl'
results = output / 'laya_results.jsonl'
if config['run_full_inference']:
    n_records = run_finetuned_laya_states(states, questions, checkpoint, results,
                                           device=config['device'])
    if n_records != manifest['n_samples']:
        raise ValueError(f'Inference coverage: {n_records} of {manifest["n_samples"]}')
    print('Fine-tuned Laya responses:', n_records)
"""),
    markdown("""## Direct Choice report and downloadable output

These are direct Choice metrics only. The local evaluator remains authoritative for the full method matrix and expert-graded retrieval. Test metrics are descriptive because this test split had already been inspected before fine-tuning was added.
"""),
    code(r"""if config['run_full_inference'] and config['run_direct_choice_evaluation']:
    direct = evaluate_finetuned_laya_choice(states, results, output / 'direct_choice_evaluation')
    print(pd.DataFrame(direct['methods']).T[['n', 'accuracy', 'macro_f1', 'balanced_accuracy', 'nll', 'brier']])

readme = '''Kaggle Laya/TMCAD output\n\n'''
readme += 'laya_checkpoint/: Laya-loadable selected model (model.safetensors, encoder/, tokenizer/, rl_agent_config.json, study_metadata.json).\n'
readme += 'laya_epoch_checkpoints/: epoch weight snapshots and validation summaries.\n'
readme += 'laya_checkpoint.history.json, laya_checkpoint.progress.json, training_history.csv: training logs.\n'
readme += 'laya_results.jsonl: matched all-sample Choice/Noul predictions, request hashes, source metadata and timing.\n'
readme += 'direct_choice_evaluation/: per-sample predictions, split metrics and confusion tables.\n'
readme += 'config/*.yaml and input_manifest.json: frozen settings and source file hashes.\n\n'
readme += 'Locally, verify/import this ZIP with kaggle/laya/import_results.py, then use configs/study_full_ft_kaggle.yaml for the matched evaluator. Retrieval also requires finalized expert grades.\n'
(output / 'README_outputs.txt').write_text(readme, encoding='utf-8')
files = [p for p in output.rglob('*') if p.is_file() and p.name != 'output_manifest.json']
result_manifest = {'input_manifest_sha256': hashlib.sha256((bundle / 'manifest.json').read_bytes()).hexdigest(),
                   'model_sha256': hashlib.sha256((checkpoint / 'model.safetensors').read_bytes()).hexdigest(),
                   'files': {str(p.relative_to(output)): {'sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
                                                           'bytes': p.stat().st_size} for p in sorted(files)}}
(output / 'output_manifest.json').write_text(json.dumps(result_manifest, indent=2), encoding='utf-8')
archive_path = Path('/kaggle/working/jev_tmcad_laya_ft_results.zip')
with zipfile.ZipFile(archive_path, 'w', compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
    for path in sorted(output.rglob('*')):
        if path.is_file():
            archive.write(path, arcname=str(Path(output.name) / path.relative_to(output)))
print('Download:', archive_path)
print('Output folder:', output)
"""),
    markdown("""## Optional convergence trial: epochs 5–8

Start from the saved epoch-four model weights and run four additional epochs with the prepared fitting, calibration, and validation partitions. The continuation uses a fresh optimizer, loss scaler, and random state, and holds the RL noise at 0.1.

Set `continuation_config['run_continuation'] = True` after the initial run completes. Outputs go under `continuation/`. Validation macro F1 selects the best checkpoint across the initial and continued epochs, followed by test inference for an improved checkpoint.
"""),
    code("""import yaml
continuation_config = yaml.safe_load(""" + repr((HERE / "continuation_config.yaml").read_text(encoding="utf-8")) + """)
# Change only this flag to True when ready to run the optional continuation.
continuation_config['run_continuation'] = False
print(continuation_config)
"""),
    code((HERE / "continue_from_snapshot.py").read_text(encoding="utf-8")),
    code("""if continuation_config['run_continuation']:
    baseline_path = output / 'direct_choice_evaluation/direct_choice_metrics.json'
    if not baseline_path.exists():
        # A fresh Kaggle session can attach the previous notebook's downloaded ZIP
        # as a second private Dataset. Restore it before touching continuation state.
        previous = list(Path('/kaggle/input').rglob('jev_tmcad_laya_ft_results.zip'))
        if len(previous) != 1:
            raise FileNotFoundError('Attach exactly one prior jev_tmcad_laya_ft_results.zip or use the original live session')
        with zipfile.ZipFile(previous[0]) as prior_archive:
            members = prior_archive.namelist()
            if not members or any(Path(name).is_absolute() or '..' in Path(name).parts or
                                  Path(name).parts[0] != output.name for name in members):
                raise ValueError('Unsafe prior result ZIP paths')
            if f'{output.name}/output_manifest.json' not in members:
                raise ValueError('Prior result ZIP lacks its output manifest')
            prior_manifest = json.loads(prior_archive.read(f'{output.name}/output_manifest.json'))
            for name, info in prior_manifest['files'].items():
                payload = prior_archive.read(f'{output.name}/{name}')
                if len(payload) != info['bytes'] or hashlib.sha256(payload).hexdigest() != info['sha256']:
                    raise ValueError(f'Prior result ZIP member changed: {name}')
            prior_archive.extractall('/kaggle/working')
        previous_data = json.loads((output / 'prepared_data_metadata.json').read_text(encoding='utf-8'))
        current_data = json.loads((bundle / 'data/metadata.json').read_text(encoding='utf-8'))
        if (previous_data['cohort_hash'] != current_data['cohort_hash'] or
                previous_data['question_sha256'] != current_data['question_sha256'] or
                previous_data['partition_hashes'] != current_data['partition_hashes']):
            raise ValueError('Prior result ZIP used another frozen cohort, question, or training partition')
        print('Restored and verified prior Kaggle result ZIP')
    if not baseline_path.exists():
        raise FileNotFoundError('Complete the original inference/direct Choice report first')
    baseline_metrics = json.loads(baseline_path.read_text(encoding='utf-8'))['methods']
    original_history = json.loads((output / 'laya_checkpoint.history.json').read_text(encoding='utf-8'))['epochs']
    if abs(baseline_metrics['val']['macro_f1'] - original_history[3]['validation_macro_f1']) > 1e-10:
        raise ValueError('Original direct Choice validation metrics differ from saved epoch 4')
    selection = continue_laya_from_saved_weights(bundle, output, config, questions, continuation_config)
    trial = output / 'continuation'
    if selection['status'] == 'new_checkpoint_selected':
        extended_results = trial / 'laya_results.jsonl'
        n_records = run_finetuned_laya_states(bundle / 'data/inference_states.jsonl', questions,
                                               trial / 'laya_checkpoint', extended_results,
                                               device=config['device'])
        if n_records != manifest['n_samples']:
            raise ValueError(f'Continuation inference coverage: {n_records} of {manifest["n_samples"]}')
        extended_metrics = evaluate_finetuned_laya_choice(bundle / 'data/inference_states.jsonl',
                        extended_results, trial / 'direct_choice_evaluation')['methods']
    else:
        extended_metrics = None  # Retain the validation-selected original checkpoint.
    comparison = {'selection': selection, 'original_direct_choice': baseline_metrics,
                  'selected_continuation_direct_choice': extended_metrics,
                  'interpretation': 'Test inference follows validation checkpoint selection.'}
    (trial / 'comparison.json').write_text(json.dumps(comparison, indent=2), encoding='utf-8')
    with zipfile.ZipFile('/kaggle/working/jev_tmcad_laya_continuation_results.zip', 'w',
                         compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for path in sorted(trial.rglob('*')):
            if path.is_file():
                archive.write(path, arcname=str(Path('continuation') / path.relative_to(trial)))
    print('Validation-selected epoch:', selection['selected_epoch'])
    print('Download: /kaggle/working/jev_tmcad_laya_continuation_results.zip')
    if extended_metrics is not None:
        print(pd.DataFrame({'original': baseline_metrics['test'],
                            'selected_continuation': extended_metrics['test']}))
"""),
]

for number, cell in enumerate(cells, 1):
    cell["id"] = f"tmcad-{number:02d}"

notebook = {"cells": cells,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 5}
target = HERE / "laya_finetune_kaggle.ipynb"
target.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(target)
