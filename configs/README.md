# Choosing a configuration

Pass a configuration before the command:

```bash
python run.py --config configs/study_full.yaml study-experiments
```

| File | Use it for |
| --- | --- |
| `default.yaml` | Scanning, descriptor extraction, and the basic JEV workflow |
| `study_full.yaml` | The full valid-descriptor cohort with geometry, JEV, and base Laya |
| `study_full_ft.yaml` | The full workflow with locally adapted Laya and retrieval annotations |
| `study_full_ft_kaggle.yaml` | Evaluation after importing the four-epoch Kaggle Laya run |
| `study_full_ft_10epoch_kaggle.yaml` | Evaluation after importing the ten-epoch Kaggle Laya run |
| `study_prompting.yaml` | Matched zero-shot/four-shot Jev acquisition and fusion |
| `study_clm.yaml` | The CLM extension, including checkpoint and encoder settings |
| `analysis.yaml` | Analysis of the recorded full study, writing to `analysis/evaluation/` |
| `study.yaml`, `study_compact.yaml` | Earlier subset runs, useful for smaller checks |
| `questions.yaml` | The original question set |
| `questions_compact.yaml` | The compact class Choice and Noul question set used by the full workflow |

Paths in files under `configs/` are relative to the repository root. Kaggle
runtime configurations live beside their notebooks under `kaggle/` and are
read by the notebook tools.

For a fresh run, copy the closest configuration and set its input and output
paths. Set `project.dataset_root` for scanning, and use separate `study.*` and
`laya_finetune.*` output paths when keeping more than one run.

`study_full.yaml` uses compact model inputs, even though its name refers to the
full cohort size. `study_compact.yaml` is an earlier subset configuration.

Main configurations use the 32-dimensional surface encoder and conflict-filtered
classification readouts. Run `study-audit` before evaluation. Set
`geometry.edge_channels`, `geometry.projection_dim`, and `geometry.embedding_dim`
to choose another encoder size.
