# Public source-only release

The commands below require companion numerical artifacts that are **not included in the public repository at this time**. Their expected locations document the original workflow. No public data download, checkpoint, or archived dataset release is claimed.

# Reproduction guide

Run commands from the repository root. Python 3.10-3.12 is supported by the
project metadata; the recorded Kaggle environments used Python 3.12.13. The
complete environment records are kept with the corresponding result bundle.
They are provenance snapshots, not a claim that one environment served every
stage of the study.

## Levels of reproduction

1. `python scripts/verify_results.py` checks the distributed artifacts, sample
   pairing, classification metrics, overlap sensitivity membership, and saved
   retrieval aggregate means using the standard library only.
2. `python scripts/materialize_artifacts.py` expands archival artifacts without
   replacing existing expanded files. It rebases the local training-data and
   selected-pair byte guards to the portable copies. Original metadata, source
   hashes, and compressed archives remain intact. Then `pip install -e .` and
   `python scripts/recompute_evaluation.py --task classification` rerun the
   original evaluator on saved features/responses. `--task visual-proxy-retrieval`
   reruns the separately identified automated endpoint. Neither contacts a model.
3. Fresh acquisition/training additionally requires locally obtained STEP data,
   optional GPU/model dependencies, and for JEV an API key. Those stages can
   incur computation or service costs and are never triggered by verification.

## Data and path identities

The frozen `sample_id`, `relative_path`, label, and split fields are the join
keys. The cohort hash excludes absolute paths and is unchanged by relocation.
Never regenerate the cohort over saved artifacts. For STEP-dependent work,
obtain the source TMCAD files under their actual access/license conditions, then:

```bash
python scripts/materialize_artifacts.py --dataset /your/local/TMCAD
```

This rebases the cohort, descriptor, embedding, and asset-manifest local paths.
The original saved response paths remain historical metadata; payloads contain
the descriptor state and questions, not the CAD path or ground-truth label.
Absolute paths in a personal `configs/default.yaml` can be set separately for
fresh scanning. Do not change original relative paths or source class mappings.

The preserved training-data metadata hashes refer to the original archived
bytes. Materialization rewrites the *expanded* metadata guards for the portable
descriptor/training copies and records that action in `release_note`. This does
not imply retraining or a change of training examples.

## Surface representation

Install an environment with CadQuery/OpenCASCADE and PyTorch, for example the
project's optional `geometry` dependencies. The registered commands are:

```bash
python run.py --config configs/study_full.yaml study-mesh
python run.py --config configs/study_full.yaml study-train-geometry
python run.py --config configs/study_full.yaml study-embed
```

Use a new output directory/configuration for fresh runs. The provided embeddings
and metadata are the recorded epoch-17 outputs, not newly regenerated ones.
The serialized checkpoint is withheld from public upload. The saved-feature
adapter checks its recorded identity without claiming to verify absent weights.
The mesh cache is rebuildable and is omitted from this repository.

## Typed response models

`configs/questions_compact.yaml` contains one ten-option Choice question and
22 Noul propositions. `semantic_state.py` constructs the shared 14-field compact
descriptor state. Saved response IDs, request hashes, model identities, splits,
and labels are validated before feature assembly. Base Laya originally stored
9,182 unique-payload rows; the evaluator restores coverage for 616 sample IDs
only after independently validating the identical payload and records reuse.

For fresh JEV requests, supply `TYPESAFE_API_KEY` via the environment; do not
commit `.env`. `study-jev --dry-run` sends no requests. The historical request
used `jev-latest`; the saved resolved model was `jev-1.13.0`. Availability of an
old hosted revision is not guaranteed. New responses must use a separate run.

Base Laya's pinned repository/revision and adaptation settings are recorded in
the Kaggle configuration and completed `study_metadata.json`. The actual
ten-epoch adaptation uses the `*_10ep.py`, `*_10ep.yaml`, and
`laya_finetune_10epoch_kaggle.ipynb` files. The earlier notebook/build helper is
retained because the ten-epoch packager depends on it. The selected checkpoint
was epoch 4 with calibration temperature 2.108482517142911. The archived
training inputs contain 6,181 fitting, 677 calibration, and 980 validation
records. No unlabelled Noul target annotations were manufactured.

```bash
python kaggle_laya/build_upload.py
python kaggle_laya/build_upload_10ep.py
python kaggle_clm/build_upload.py
```

The notebook builders and import validators remain available. These commands
prepare portable inputs; they do not launch remote jobs. Follow the notebook's
input instructions for a fresh, separate execution. Do not overwrite the saved
completed output directory.

The completed CLM run used contrastive-lm 0.1.0, the released CLM v0.1-8B head,
and Qwen3-8B revision `b968826d9c46dd6066d109eabc6255188de91218`; its environment
records vLLM 0.14.1 and Transformers 4.57.6. The Transformers adapter is also
provided for local inference, but was not the recorded Kaggle backend. Obtain
upstream weights separately under their licenses. No CAD-specific CLM adaptation
was performed in the supplied run.

## Retrieval and annotation

The frozen extension has 50 validation and 100 test queries and 6,645 pairs.
The saved automatic proxy endpoint is distinct from the independent engineering
relevance protocol. Do not substitute class equality, silhouette scores, or
synthetic rater streams for independent expert assessments.

The independent `rater_a.csv`, `rater_b.csv`, and `adjudication.csv` are **blank
forms** in this collection. The annotation app source is under
`research_v2/expert_review_app/`. Its separate README explains preparation and
the required local database/secrets. Rendered assets are regenerated from STEP
files and are omitted here. The whole analysis repository contains class labels;
deploy only the blinded app package when maintaining rater blinding.

## Verification boundaries

The saved metric checks use a tolerance of 2e-6 for probability-derived
classification metrics and 1e-10 for retrieval aggregate means. They check
consistency with stored predictions, not the scientific validity of every
methodological assumption. No new training, inference, annotation, revised
split, or performance estimate was generated while assembling this release.
