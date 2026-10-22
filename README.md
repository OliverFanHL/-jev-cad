# JEV-CAD

Tools for classifying mechanical CAD parts and finding similar parts using
geometry and responses from JEV, Laya, and CLM. This is the code for
*Geometry–response fusion for mechanical CAD analysis with typed decision models*.

The workflow starts with STEP files, extracts geometric measurements and surface
features, asks the response models a shared set of questions, and compares the
resulting representations through classification and retrieval.

## Getting started

Use Python 3.10–3.12. From the repository root:

```bash
python -m pip install -e ".[dev]"
python run.py --help
python run.py study-experiments
python -m pytest tests -q
```

`--help` lists the commands. `study-experiments` lists the available comparisons
and the features each one uses.

To work with your own CAD files, install the geometry dependencies and start
with scanning and descriptor extraction:

```bash
python -m pip install -e ".[geometry]"
python run.py scan --dataset /path/to/TMCAD
python run.py extract --workers 1
python run.py inspect-state -n 3
```

Set `TYPESAFE_API_KEY` in your environment or a local `.env` file to use JEV.
See [.env.example](.env.example). Laya and CLM use locally obtained model weights;
their setup and the remaining stages are covered in the [workflow guide](docs/REPRODUCTION.md).

The repository includes the source and configurations. CAD files, trained
weights, and saved study results are held separately.

## Finding your way around

| Location | What you will find |
| --- | --- |
| [src/jev_tmcad/](src/jev_tmcad/) | The Python package: data, geometry, models, training, evaluation, and workflow commands |
| [configs/](configs/) | Dataset settings, model questions, and run configurations |
| [analysis/](analysis/) | Scripts for figures, comparison tables, and CAD renders |
| [kaggle/](kaggle/) | Laya and CLM notebooks, input packaging, and result import tools |
| [apps/expert_review/](apps/expert_review/) | Browser app for reviewing pairs of CAD parts |
| [scripts/](scripts/) | Tools for opening saved artifact bundles and checking their results |
| [tests/](tests/) | Tests for data handling, model responses, and evaluation |

For a guide to individual modules and how they connect, see the
[code map](docs/CODE_MAP.md). The [configuration guide](configs/README.md) explains
which YAML file to use for each workflow.

## Common tasks

| Task | Where to start |
| --- | --- |
| Run the full geometry and response pipeline | [Workflow guide](docs/REPRODUCTION.md) |
| Add or change a model backend | [models/](src/jev_tmcad/models/) |
| Change the questions sent to models | [questions_compact.yaml](configs/questions_compact.yaml) |
| Add a classification or retrieval comparison | [experiments.py](src/jev_tmcad/evaluation/experiments.py) |
| Fine-tune Laya or run CLM on Kaggle | [Kaggle guide](kaggle/README.md) |
| Build figures from saved outputs | [Analysis guide](analysis/README.md) |
| Set up the CAD review app | [App guide](apps/expert_review/README.md) |

## Dependencies and reuse

The default installation covers the command line tools and evaluation.
Optional extras add CAD processing (`geometry`), Laya (`laya`), or both (`study`).
Figure scripts use the packages in [requirements-figures.txt](requirements-figures.txt).

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for dependency and data
licensing details. A project license has not yet been selected.
