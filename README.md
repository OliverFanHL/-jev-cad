# JEV-CAD

Geometry and typed model responses for mechanical CAD classification and retrieval.
Implementation of *Geometryâ€“response fusion for mechanical CAD analysis with typed
decision models*.

STEP solids produce whole-part descriptors and learned surface embeddings. Jev,
CLM, and Laya answer a shared question map from a compact geometric state.
Classification combines these features with a common logistic readout; retrieval
combines cosine-similarity scores.

## Getting started

Use Python 3.10â€“3.12:

```bash
python -m pip install -e ".[dev]"
python run.py --help
python run.py study-experiments
python -m pytest -q
```

Install `.[geometry]` for STEP processing and the surface encoder, `.[laya]` for
local Laya, or `.[study]` for both. Set `TYPESAFE_API_KEY` in your environment or
[.env](.env.example) for Jev. Supply CAD files and model weights locally.

```bash
python -m pip install -e ".[geometry]"
python run.py scan --dataset /path/to/TMCAD
python run.py extract --workers 1
python run.py --config configs/study_full.yaml study-plan
```

Continue with the [workflow guide](docs/REPRODUCTION.md).
The [method guide](docs/METHOD.md) describes the representations and evaluation
protocol; [configs/README.md](configs/README.md) explains the run settings.

## Project layout

| Location | Purpose |
| --- | --- |
| [src/jev_tmcad/](src/jev_tmcad/) | Data, geometry, model clients, training, evaluation, and commands |
| [configs/](configs/) | Dataset settings, question maps, and run configurations |
| [kaggle/](kaggle/) | Notebook builders, input packaging, and result import |
| [apps/expert_review/](apps/expert_review/) | Blinded CAD pair grading and adjudication |
| [tests/](tests/) | Data alignment, model interfaces, and evaluation checks |

See the [code map](docs/CODE_MAP.md) for module entry points and
[third-party notices](THIRD_PARTY_NOTICES.md) for resource licenses.
