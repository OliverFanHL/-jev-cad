# Code map

Start with `run.py` for command line usage. It loads `jev_tmcad.cli`, which
registers the basic scan/extract commands and the full workflow commands from
`workflows/study.py`.

## Python package

| Module | Responsibility | Useful entry points |
| --- | --- | --- |
| `data/dataset.py` | Scan STEP files, normalize class names, create splits | `scan_tmcad`, `add_stratified_splits` |
| `data/cohort.py` | Freeze and validate the shared sample population | `create_cohort`, `load_cohort` |
| `data/mesh.py` | Build cached surface neighborhoods | `build_mesh_cache` |
| `data/annotations.py` | Prepare blinded rating forms and finalize grades | `prepare_annotation_files`, `finalize_annotation_files` |
| `geometry/descriptors.py` | Measure CAD shapes | `extract_descriptor` |
| `geometry/extraction.py` | Process a dataset into a descriptor table | `extract_dataset_descriptors` |
| `geometry/encoder.py` | Train the surface encoder and extract embeddings | `train_geometry`, `extract_embeddings` |
| `models/jev.py`, `models/laya.py`, `models/clm.py` | Acquire and cache typed responses | `run_jev_for_dataframe`, `run_laya_for_dataframe`, `run_clm_for_dataframe` |
| `models/clm_transformers.py` | Local Transformers encoder for CLM | `TransformersEmbedder` |
| `training/laya.py` | Prepare adaptation data, train, calibrate, infer | `prepare_finetune_data`, `train_laya_finetune` |
| `evaluation/experiments.py` | Define feature combinations and comparison IDs | `experiment_catalog` |
| `evaluation/classification.py` | Align features and evaluate the full study | `prepare_features`, `evaluate_classification` |
| `evaluation/clm.py` | Assemble CLM features for the shared evaluator | `prepare_clm_features` |
| `evaluation/retrieval.py` | Rank candidates and score retrieval | `evaluate_retrieval` |
| `evaluation/reporting.py`, `evaluation/qualitative.py` | Write prediction tables, confusion plots, and example reports | `prediction_frame`, `generate_qualitative_report` |
| `evaluation/audit.py`, `evaluation/efficiency.py` | Check duplicate files and summarize timing | `audit_exact_duplicates`, `summarize_efficiency` |
| `evaluation/pilot_classification.py`, `evaluation/pilot_retrieval.py` | Evaluators for the initial descriptor/JEV workflow | `evaluate_classification`, `evaluate_retrieval` |
| `workflows/study.py` | Connect stages to `study-*` commands | `register_study_commands` |

Shared modules remain at the package root: `config.py` loads run settings,
`semantic_state.py` builds the model input, `provenance.py` keeps source labels
and file identities, and `utils.py` provides file and hashing helpers.

## Following a part through the code

1. `data/dataset.py` assigns a sample ID, source label, and split.
2. `geometry/extraction.py` calls the descriptor functions for each STEP file.
3. `semantic_state.py` turns those measurements into the shared model input.
4. `data/cohort.py` selects the parts used by every comparison.
5. `geometry/encoder.py` and `models/` produce surface and response features.
6. `evaluation/classification.py` aligns records by sample ID and runs the
   comparisons defined in `evaluation/experiments.py`.
7. `evaluation/retrieval.py` uses the same feature blocks to rank candidates.
8. Scripts in `analysis/` turn saved evaluation outputs into figures and tables.

## Changes from the earlier layout

The command names, experiment IDs, and question definitions are unchanged.
Python imports now follow the package groups above. For example:

```python
from jev_tmcad.models.jev import run_jev_for_dataframe
from jev_tmcad.evaluation.classification import prepare_features
```

| Earlier location | Current location |
| --- | --- |
| `research_v2/scripts/` | `analysis/scripts/` |
| `research_v2/evaluation/`, `research_v2/evidence/` | `analysis/evaluation/`, `analysis/evidence/` |
| `research_v2/study_analysis.yaml` | `configs/analysis.yaml` |
| `research_v2/expert_review_app/` | `apps/expert_review/` |
| `kaggle_laya/`, `kaggle_clm/` | `kaggle/laya/`, `kaggle/clm/` |

The artifact helpers recognize these earlier directory names when reading an
existing result bundle. For a new Kaggle run, rebuild the upload ZIP so it
contains the current package layout.
