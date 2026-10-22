# Running the workflow

Run these commands from the repository root with Python 3.10–3.12.
Start with the [README](../README.md) for installation.

```text
STEP files → descriptors → frozen cohort → surface embeddings
                       └→ model responses ─┬→ classification
                                          └→ retrieval + relevance grades
```

Each stage saves its output for the next one. Paths and model settings are in
the YAML configurations; see [configs/README.md](../configs/README.md).

## 1. Scan CAD files and extract descriptors

Install the geometry extra, then point the scanner at a local TMCAD directory:

```bash
python -m pip install -e ".[geometry]"
python run.py scan --dataset /path/to/TMCAD
python run.py extract --workers 1
python run.py inspect-state -n 3
```

The scanner records file identities, class labels, and train/validation/test
splits in `outputs/manifest.csv`. Extraction writes
`outputs/descriptors/tmcad_descriptors.csv`. `inspect-state` fits the quantile
schema on training descriptors and shows the representation sent to models.

## 2. Select the cohort and build surface features

`study_full.yaml` selects all parts with valid descriptors and keeps their
existing splits:

```bash
python run.py --config configs/study_full.yaml study-plan
python run.py --config configs/study_full.yaml study-mesh
python run.py --config configs/study_full.yaml study-train-geometry
python run.py --config configs/study_full.yaml study-embed
```

The cohort, mesh cache, encoder checkpoint, and embeddings are written under
`outputs/study/full/`. PyTorch handles encoder training; a GPU is useful for
larger runs. `study-audit` checks exact duplicate CAD files across splits.

For a new run alongside existing results, copy the configuration and change its
output paths first. Keep the same cohort throughout acquisition and evaluation.

## 3. Collect model responses

The full configuration uses the compact descriptor state and the questions in
`configs/questions_compact.yaml`: one class Choice question and 22 Noul propositions.

For JEV, set `TYPESAFE_API_KEY` in the environment or `.env`. Preview requests
before making live calls:

```bash
python run.py --config configs/study_full.yaml study-jev --dry-run
python run.py --config configs/study_full.yaml study-jev
```

For base Laya, install its extra and run inference:

```bash
python -m pip install -e ".[laya]"
python run.py --config configs/study_full.yaml study-laya
```

CLM has its own configuration for the head checkpoint, encoder, and backend:

```bash
python run.py --config configs/study_clm.yaml study-clm --dry-run
python run.py --config configs/study_clm.yaml study-clm
```

Obtain the weights and backend dependencies for the selected model before live
inference. Responses are saved by sample ID with their request and model
identities, so evaluation can check that they match the cohort and questions.

## 4. Adapt Laya (optional)

Prepare fitting, calibration, and validation examples, train Laya, then collect
responses from the selected checkpoint:

```bash
python run.py --config configs/study_full_ft.yaml study-laya-ft-prepare
python run.py --config configs/study_full_ft.yaml study-laya-ft-train
python run.py --config configs/study_full_ft.yaml study-laya-ft-infer
```

Validation macro F1 selects the checkpoint. A separate partition of training
examples fits the Choice temperature. For GPU execution on Kaggle, follow the
[Kaggle guide](../kaggle/README.md).

## 5. Evaluate classification

Once the configured features and responses are available:

```bash
python run.py --config configs/study_full.yaml study-evaluate --classification-only
```

Use `study_full_ft.yaml` to include locally adapted Laya, or the corresponding
Kaggle configuration after importing a notebook result. Evaluation writes
metrics, predictions, confusion tables, and sample alignment records to
`study.evaluation_dir`.

List the comparisons or run one by its ID:

```bash
python run.py study-experiments
python run.py --config configs/study_full.yaml study-evaluate --classification-only --experiment classification/descriptor_only
```

## 6. Prepare relevance grades and evaluate retrieval

Create the candidate pool and rating forms with a configuration that includes
the annotation paths:

```bash
python run.py --config configs/study_full_ft.yaml study-pool
python run.py --config configs/study_full_ft.yaml study-grade-prepare
```

The [review app](../apps/expert_review/README.md) displays blinded CAD pairs and
collects each reviewer's grades. After both reviewers finish and disagreements
are adjudicated, place the exported CSVs in the configured annotation directory:

```bash
python run.py --config configs/study_full_ft.yaml study-grade-finalize
python run.py --config configs/study_full_ft.yaml study-evaluate
```

The automatic silhouette comparison is a separate tool in the
[analysis guide](../analysis/README.md). Its scores measure visual similarity;
expert relevance grades come from the review workflow.

## Working with a saved result bundle

The public repository does not include the saved study outputs. If you have a
companion bundle, place its archives and `ARTIFACT_PROVENANCE.json` in the
repository, keeping their directory structure. The helper scripts also accept
the earlier `research_v2/` and `kaggle_laya/` bundle paths.

```bash
python scripts/materialize_artifacts.py
python scripts/verify_results.py
python scripts/recompute_evaluation.py --task classification
```

Materialization expands the archives into the current layout. Verification
checks hashes, sample pairing, and metrics using the standard library.
Recomputation fits the downstream probes on saved features and writes to
`recomputed/classification/`. It needs the installed evaluation dependencies.

For STEP-dependent work, add `--dataset /path/to/TMCAD` to materialization to
update local CAD paths. To evaluate the saved automatic visual-similarity
endpoint, use `--task visual-proxy-retrieval` with the recomputation script.

The historical cohort contains 9,798 parts: 6,858 training, 980 validation, and
1,960 test parts. It includes exact-file overlap across splits, and the test set
was inspected during development. Keep those limitations in mind when using
the historical metrics. Saved bundle metadata records checkpoint identities,
environment versions, and the source of retrieval scores.
