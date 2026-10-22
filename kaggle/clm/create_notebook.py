"""Write a self-contained Kaggle CLM notebook; this does not run inference."""
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
    markdown("""# CLM on the frozen TMCAD cohort

Attach the new vLLM `jev_tmcad_clm_v1.zip` as a **private Kaggle Dataset**, select a GPU, and enable Internet. Prefer a two-GPU accelerator or one GPU with enough memory for Qwen3-8B. This notebook runs the official CLM projection head against the official Qwen3-8B vLLM pooling setup. The input includes the same frozen cohort, descriptor state, questions, geometry embeddings, and baseline prediction records used by the JEV benchmark. CLM receives only the label-free state and question texts. No CLM fine-tuning is performed.

The notebook writes one record per sample and can resume from a **vLLM-generated** `jev_tmcad_clm_results.zip` attached as a second private Dataset. Results from the earlier Transformers notebook are incompatible and must not be attached. Each session stops after eight hours or complete coverage, then exports a fresh ZIP. When coverage reaches 9,798, it also runs the registered classification evaluator and writes the figure/table data. Expert-graded retrieval runs only if a finalized `judged_pairs.csv` and its `.meta.json` are attached.
"""),
    code("""from pathlib import Path
import sys, subprocess, zipfile

inputs = Path('/kaggle/input')
matches = [p for p in inputs.rglob('manifest.json') if p.parent.name == 'jev_tmcad_clm']
if not matches:
    zips = list(inputs.rglob('jev_tmcad_clm_v1.zip'))
    if len(zips) != 1:
        raise FileNotFoundError('Attach exactly one private Dataset containing jev_tmcad_clm_v1.zip')
    unpacked = Path('/kaggle/working/unpacked_clm_input')
    with zipfile.ZipFile(zips[0]) as archive:
        if any(Path(n).is_absolute() or '..' in Path(n).parts for n in archive.namelist()):
            raise ValueError('Unsafe CLM input ZIP path')
        archive.extractall(unpacked)
    matches = [p for p in unpacked.rglob('manifest.json') if p.parent.name == 'jev_tmcad_clm']
if len(matches) != 1:
    raise ValueError(f'Expected one CLM input manifest, found {len(matches)}')
bundle = matches[0].parent
print('Input bundle:', bundle)

# Install the Linux CUDA vLLM wheel before importing torch in this kernel.
# Pin the version so repeated Kaggle sessions use the same serving stack.
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',
                       'vllm==0.14.1'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps',
                       'contrastive-lm==0.1.0'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',
                       'pandas>=2.1', 'scikit-learn>=1.4',
                       'matplotlib>=3.8', 'PyYAML>=6', 'tqdm>=4.66', 'requests>=2.28'])
"""),
    code("""import hashlib, importlib, json, platform, shutil, time
import pandas as pd
import torch
import yaml

manifest = json.loads((bundle / 'manifest.json').read_text(encoding='utf-8'))
if manifest['bundle_type'] != 'jev_tmcad_clm_v1' or manifest['n_samples'] != 9798:
    raise ValueError('Wrong CLM input bundle')
for name, info in manifest['files'].items():
    payload = (bundle / name).read_bytes()
    if len(payload) != info['bytes'] or hashlib.sha256(payload).hexdigest() != info['sha256']:
        raise ValueError(f'Input bundle file changed: {name}')
if not torch.cuda.is_available():
    raise RuntimeError('Select a Kaggle GPU accelerator before running CLM')
if any(torch.cuda.get_device_capability(i)[0] < 7
       for i in range(torch.cuda.device_count())):
    raise RuntimeError('vLLM requires compute capability 7.0 or higher; select T4 or newer GPUs')
print('Verified', len(manifest['files']), 'files; GPUs:',
      [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
for gpu in range(torch.cuda.device_count()):
    free, total = torch.cuda.mem_get_info(gpu)
    print(f'GPU {gpu}: {free / 2**30:.1f} GiB free / {total / 2**30:.1f} GiB total')
if torch.cuda.device_count() == 1 and torch.cuda.mem_get_info(0)[1] < 20 * 2**30:
    raise RuntimeError('This notebook requires two 16 GB GPUs or one GPU with at least 20 GB; '
                       'vLLM must load Qwen3-8B without CPU offload')
config = yaml.safe_load((bundle / 'config/study_clm.yaml').read_text(encoding='utf-8'))
if config['clm']['encoder_backend'] != 'vllm':
    raise ValueError('Attach the new vLLM input ZIP; the older Transformers bundle is incompatible')
questions = yaml.safe_load((bundle / 'config/questions_compact.yaml').read_text(encoding='utf-8'))
sys.path.insert(0, str(bundle / 'code/src'))
for name in list(sys.modules):
    if name == 'jev_tmcad' or name.startswith('jev_tmcad.'):
        del sys.modules[name]
importlib.invalidate_caches()
from jev_tmcad.data.cohort import load_cohort, valid_descriptors
from jev_tmcad.semantic_state import QuantileSchema
from jev_tmcad.models.clm import run_clm_for_dataframe, file_sha256
from jev_tmcad.evaluation.clm import prepare_clm_features
from jev_tmcad.evaluation.classification import evaluate_classification
from jev_tmcad.evaluation.retrieval import evaluate_retrieval
assert hashlib.sha256((bundle / 'config/questions_compact.yaml').read_bytes()).hexdigest() == manifest['questions_sha256']
cohort = load_cohort(bundle / 'data/cohort.csv')
local_descriptor = Path('/kaggle/working/clm_descriptors.csv')
print('Staging verified descriptor CSV locally:',
      (bundle / 'data/descriptors.csv').stat().st_size, 'bytes', flush=True)
shutil.copyfile(bundle / 'data/descriptors.csv', local_descriptor)
desc = valid_descriptors(local_descriptor)
print('Loaded', len(desc), 'valid descriptor rows', flush=True)
rows = cohort[['sample_id']].merge(desc, on='sample_id', validate='one_to_one')
if len(rows) != manifest['n_samples']:
    raise ValueError('CLM descriptor coverage differs from frozen cohort')
for key in ('split', 'label', 'source_label', 'relative_path', 'absolute_path'):
    if rows[key].astype(str).ne(cohort[key].astype(str)).any():
        raise ValueError(f'CLM descriptor {key} differs from frozen cohort')
schema = QuantileSchema.from_dict(json.loads((bundle / 'data/schema.json').read_text()))
output = Path('/kaggle/working/jev_tmcad_clm_results')
prior_zips = list(inputs.rglob('jev_tmcad_clm_results.zip'))
if len(prior_zips) > 1:
    raise ValueError('Attach at most one prior CLM results ZIP')
if prior_zips:
    with zipfile.ZipFile(prior_zips[0]) as archive:
        names = archive.namelist()
        if any(Path(n).is_absolute() or '..' in Path(n).parts or
               Path(n).parts[0] != output.name for n in names):
            raise ValueError('Unsafe prior result ZIP path')
        previous = json.loads(archive.read(f'{output.name}/output_manifest.json'))
        if previous['input_manifest_sha256'] != hashlib.sha256((bundle / 'manifest.json').read_bytes()).hexdigest():
            raise ValueError('Prior CLM results used a different input bundle; do not attach Transformers results')
        previous_rows = archive.read(f'{output.name}/clm_results.jsonl').splitlines()
        if previous_rows and json.loads(previous_rows[0]).get('encoder_backend') != 'vllm':
            raise ValueError('Prior CLM results were not produced by vLLM')
        for name, info in previous['files'].items():
            data = archive.read(f'{output.name}/{name}')
            if len(data) != info['bytes'] or hashlib.sha256(data).hexdigest() != info['sha256']:
                raise ValueError(f'Prior result changed: {name}')
        archive.extractall('/kaggle/working')
    print('Restored', prior_zips[0])
output.mkdir(parents=True, exist_ok=True)
shutil.copy2(bundle / 'manifest.json', output / 'input_manifest.json')
shutil.copy2(bundle / 'config/study_clm.yaml', output / 'study_clm.yaml')
shutil.copy2(bundle / 'config/questions_compact.yaml', output / 'questions_compact.yaml')
environment = {'python': platform.python_version(), 'torch': torch.__version__,
               'cuda': torch.version.cuda, 'gpu': [torch.cuda.get_device_name(i)
               for i in range(torch.cuda.device_count())],
               'encoder_revision': config['clm']['encoder_revision']}
(output / 'environment.json').write_text(json.dumps(environment, indent=2), encoding='utf-8')
"""),
    markdown("""## Run or resume CLM inference

The official CLM head is downloaded to temporary Kaggle working storage. The notebook starts vLLM's pooling server for the pinned Qwen3-8B revision, verifies a normalized 4096-dimensional embedding, then calls the official CLM Engine through its embedding client. The per-session sample cap covers the full cohort; the eight-hour time limit stops inference in time to export a resumable ZIP if coverage is still incomplete.
"""),
    code("""import math, os, signal
import requests
from clm.heads import download
from importlib.metadata import version

head_dir = Path('/kaggle/working/clm_head')
checkpoint = Path(download(dest_dir=str(head_dir)))
environment['contrastive_lm'] = version('contrastive-lm')
environment['vllm'] = version('vllm')
environment['transformers'] = version('transformers')
environment['head_sha256'] = file_sha256(checkpoint)
(output / 'environment.json').write_text(json.dumps(environment, indent=2), encoding='utf-8')

# Edit only these execution limits if needed; they do not alter the protocol.
max_new_samples = 9798
max_runtime_seconds = 8 * 60 * 60
runner = dict(checkpoint=checkpoint, model=config['clm']['model'],
              encoder_backend='vllm', encoder_model=config['clm']['encoder_model'],
              encoder_revision=config['clm']['encoder_revision'],
              emb_url=config['clm']['emb_url'], emb_model=config['clm']['emb_model'],
              timeout_seconds=float(config['clm']['timeout_seconds']),
              retry_attempts=int(config['clm']['retry_attempts']),
              include_numeric=bool(config['cad']['include_numeric_context']),
              include_profiles=bool(config['cad']['include_profile_arrays']),
              state_variant=config['cad']['state_variant'])
result_path = output / 'clm_results.jsonl'
preview = run_clm_for_dataframe(rows, schema, questions, result_path,
                                dry_run=True, **runner)
if preview['pending'] == 0:
    result = preview
else:
    executable = shutil.which('vllm')
    if not executable:
        raise RuntimeError('vLLM installation did not provide the vllm executable')
    dtype = ('half' if any(torch.cuda.get_device_capability(i)[0] < 8
                           for i in range(torch.cuda.device_count())) else 'bfloat16')
    command = [executable, 'serve', config['clm']['encoder_model'],
               '--served-model-name', config['clm']['emb_model'],
               '--runner', 'pooling', '--max-model-len', '2048',
               '--revision', config['clm']['encoder_revision'],
               '--tokenizer-revision', config['clm']['encoder_revision'],
               '--tensor-parallel-size', str(torch.cuda.device_count()),
               '--dtype', dtype, '--attention-backend', 'TRITON_ATTN',
               '--gpu-memory-utilization', '0.90',
               '--max-num-seqs', '32', '--host', '127.0.0.1', '--port', '8090']
    environment['vllm_server'] = {'model': config['clm']['encoder_model'],
                                  'revision': config['clm']['encoder_revision'],
                                  'dtype': dtype, 'tensor_parallel_size': torch.cuda.device_count(),
                                  'runner': 'pooling', 'max_model_len': 2048,
                                  'attention_backend': 'TRITON_ATTN'}
    (output / 'environment.json').write_text(json.dumps(environment, indent=2), encoding='utf-8')
    server_log = output / 'vllm_server.log'
    print('Starting vLLM:', ' '.join(command))
    # Kaggle T4 containers may lack the libcuda.so linker name needed by
    # FlashInfer JIT. Triton attention avoids that JIT link step.
    with server_log.open('w', encoding='utf-8') as log:
        server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                  start_new_session=True)
    try:
        started = time.monotonic()
        deadline = started + 20 * 60
        next_report = started + 20
        log_offset = [0]
        def report_startup():
            with server_log.open('r', encoding='utf-8', errors='replace') as stream:
                stream.seek(log_offset[0])
                new_log = stream.read()
                log_offset[0] = stream.tell()
            if new_log:
                print('vLLM startup log (latest):\\n' + new_log[-3000:], flush=True)
            else:
                print(f'Waiting for vLLM health: {time.monotonic() - started:.0f}s; '
                      f'process running, log: {server_log}', flush=True)
        while True:
            if server.poll() is not None:
                report_startup()
                raise RuntimeError(f'vLLM exited during startup with code {server.returncode}. '
                                   f'Inspect {server_log}')
            try:
                ready = requests.get('http://127.0.0.1:8090/health', timeout=5)
                if ready.ok:
                    break
            except requests.RequestException:
                pass
            now = time.monotonic()
            if now >= next_report:
                report_startup()
                next_report = now + 20
            if now >= deadline:
                report_startup()
                raise TimeoutError(f'vLLM did not become healthy within 20 minutes. '
                                   f'Inspect {server_log}')
            time.sleep(5)
        probe = requests.post(config['clm']['emb_url'],
                              json={'model': config['clm']['emb_model'],
                                    'input': ['CLM pooling readiness check']}, timeout=120)
        if not probe.ok:
            print(f'Embedding probe HTTP {probe.status_code}: {probe.text[:3000]}', flush=True)
            with server_log.open('rb') as stream:
                stream.seek(0, 2)
                stream.seek(max(0, stream.tell() - 8000))
                print('vLLM server log tail:\\n' +
                      stream.read().decode('utf-8', errors='replace'), flush=True)
            probe.raise_for_status()
        vector = probe.json()['data'][0]['embedding']
        norm = math.sqrt(sum(float(x) ** 2 for x in vector))
        if len(vector) != 4096 or not 0.99 <= norm <= 1.01:
            raise ValueError(f'Unexpected vLLM embedding: dimension={len(vector)}, norm={norm}')
        print('vLLM ready: 4096-dimensional normalized embedding; tensor parallel GPUs:',
              torch.cuda.device_count())
        try:
            result = run_clm_for_dataframe(
                rows, schema, questions, result_path, **runner,
                max_new_samples=max_new_samples,
                max_runtime_seconds=max_runtime_seconds)
        except KeyboardInterrupt:
            # The JSONL is appended after every complete part. Preserve it on a manual stop.
            if not result_path.is_file():
                raise
            completed = len(result_path.read_text(encoding='utf-8').splitlines())
            result = {'completed': completed, 'cohort_size': len(rows),
                      'pending': len(rows) - completed}
            result_path.with_suffix('.meta.json').write_text(
                json.dumps(result, indent=2), encoding='utf-8')
            print('Interrupted after saving', completed, 'complete records')
    finally:
        if server.poll() is None:
            os.killpg(server.pid, signal.SIGTERM)
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                server.wait(timeout=30)
print('CLM coverage:', result['completed'], '/', result['cohort_size'],
      '| remaining:', result['pending'])
recent = [json.loads(line)['latency_seconds']
          for line in (output / 'clm_results.jsonl').read_text(encoding='utf-8').splitlines()[-20:]]
recent = [seconds for seconds in recent if seconds is not None]
if recent:
    seconds_per_sample = sum(recent) / len(recent)
    print(f'Recent model latency: {seconds_per_sample:.1f} s/sample; '
          f'rough remaining model time: {result["pending"] * seconds_per_sample / 3600:.1f} hours')
"""),
    markdown("""## Matched classification and optional expert retrieval

Classification is evaluated only after all 9,798 CLM records are present. It calls the repository's registered evaluator with the same validation C grid and metric names. Existing baseline predictions are included only for matched evidence tables. Retrieval runs only when a finalized expert judgment CSV and matching metadata are attached as a private input Dataset.
"""),
    code("""if result['pending'] == 0:
    features = prepare_clm_features(
        cohort, local_descriptor,
        bundle / 'data/geometry_embeddings.csv', bundle / 'data/geometry_encoder.pt',
        output / 'clm_results.jsonl', checkpoint, bundle / 'data/schema.json', questions,
        model=config['clm']['model'], state_variant=config['cad']['state_variant'],
        include_numeric=bool(config['cad']['include_numeric_context']),
        include_profiles=bool(config['cad']['include_profile_arrays']))
    methods = evaluate_classification(features, output / 'evaluation/classification',
                                      c_grid=[float(x) for x in config['classification']['c_grid']])
    print('Classification methods:', list(methods))
    evidence_module = bundle / 'code/build_clm_evidence.py'
    import importlib.util
    spec = importlib.util.spec_from_file_location('build_clm_evidence', evidence_module)
    evidence = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evidence)
    evidence.build(output / 'evaluation', bundle / 'baseline/evaluation', output / 'evidence',
                   cohort_path=bundle / 'data/cohort.csv',
                   audit_path=bundle / 'baseline/duplicate_audit.json',
                   cases_path=bundle / 'baseline/qualitative_examples.json')
    judgments = list(inputs.rglob('judged_pairs.csv'))
    if judgments:
        if len(judgments) != 1 or not judgments[0].with_suffix('.meta.json').exists():
            raise ValueError('Attach exactly one finalized judged_pairs.csv and its .meta.json')
        for method in ('clm', 'geometry_plus_clm'):
            directory = output / 'evaluation/experiments/retrieval' / method / 'retrieval'
            evaluate_retrieval(features, judgments[0], directory,
                               ks=[int(x) for x in config['retrieval']['ks']],
                               weights=[float(x) for x in config['retrieval']['fusion_weights']],
                               only=method, query_scope=config['retrieval']['query_scope'])
        print('Completed expert-judged CLM retrieval')
else:
    print('Classification and retrieval wait for complete CLM coverage')
"""),
    code("""(output / 'README_outputs.txt').write_text(
    'CLM results for the frozen TMCAD cohort.\\n'
    'clm_results.jsonl and .meta.json: resumable per-sample Choice/Noul outputs.\\n'
    'evaluation/classification: registered metrics, test predictions and confusion figures after full coverage.\\n'
    'evidence: matched table, class recall, paired outcomes and duplicate sensitivity.\\n'
    'evaluation/experiments/retrieval: only when finalized expert grades are attached.\\n'
    'Attach this ZIP as a private Dataset to resume if clm_results.meta.json reports pending > 0.\\n',
    encoding='utf-8')
files = [p for p in output.rglob('*') if p.is_file() and p.name != 'output_manifest.json']
out_manifest = {
    'input_manifest_sha256': hashlib.sha256((bundle / 'manifest.json').read_bytes()).hexdigest(),
    'checkpoint_sha256': file_sha256(checkpoint),
    'completed': result['completed'], 'pending': result['pending'],
    'files': {str(p.relative_to(output)): {'sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
                                         'bytes': p.stat().st_size} for p in sorted(files)}
}
(output / 'output_manifest.json').write_text(json.dumps(out_manifest, indent=2), encoding='utf-8')
archive_path = Path('/kaggle/working/jev_tmcad_clm_results.zip')
with zipfile.ZipFile(archive_path, 'w', compression=zipfile.ZIP_DEFLATED,
                     compresslevel=6, allowZip64=True) as archive:
    for path in sorted(output.rglob('*')):
        if path.is_file():
            archive.write(path, arcname=str(Path(output.name) / path.relative_to(output)))
print('Download:', archive_path, '| complete:', result['pending'] == 0)
"""),
]

notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3",
                                                    "language": "python", "name": "python3"},
                                "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 5}
target = HERE / "clm_tmcad_kaggle.ipynb"
target.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Prepared {target} ({len(cells)} cells)")
