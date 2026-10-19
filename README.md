# Geometry–response fusion for mechanical CAD analysis

Reusable source code for **Geometry–response fusion for mechanical CAD analysis
with typed decision models**. The implementation combines deterministic CAD
measurements, a learned local surface representation, and structured responses
from typed decision models, with separate direct-decision, linear-probe, and
fusion evaluation paths.

**Release status: source code only.** The paired experimental outputs, cohort
records, feature matrices, model responses, annotation records, model weights,
and raw STEP solids are not publicly released in this repository at this time.
The authors retain a separate local collection of saved outputs. Documentation
and scripts specify the expected input locations; they do not imply that those
inputs are downloadable here. No archived data DOI is claimed.

## Source layout

| Location | Responsibility |
| --- | --- |
| `src/jev_tmcad/`, `run.py` | Data preparation, descriptors, state construction, surface encoder, model interfaces, evaluation |
| `configs/` | Question maps and study settings |
| `research_v2/study_analysis.yaml` | Original downstream analysis configuration |
| `kaggle_laya/` | Input packaging, adaptation notebooks, and output import validation |
| `kaggle_clm/` | CLM packaging, inference notebook, and import validation |
| `research_v2/scripts/` | Figure, rendering, and evidence postprocessors |
| `research_v2/expert_review_app/` | Independent-rater/adjudication application source; no study data, renders, database, or tokens |
| `scripts/` | Companion-artifact materialization, saved-result audit, optional cached-feature evaluator |
| `tests/` | Existing implementation tests; fixtures are test examples, not experimental observations |

The package contains no credentials, cached requests, partial result exports,
superseded guide versions, synthetic expert-rating generators, temporary
revision scripts, downloaded compilers, or manuscript submission drafts.

## Install and inspect

Use Python 3.10–3.12:

```bash
python -m pip install -e ".[dev]"
python run.py --help
python run.py --config configs/study_full.yaml study-experiments
python -m pytest tests -q
```

These commands install the source, list the workflow/registered methods, and run
implementation tests. They do not start training or send model requests.
New JEV acquisition requires your own `TYPESAFE_API_KEY` supplied through the
environment. `.env.example` has no credential. Optional CAD, GPU, and Laya
stages need their declared extra dependencies and separately obtained inputs.

## Reproduction stages

`docs/REPRODUCTION.md` describes the full workflow and its expected companion
artifacts. The original configurations are retained for auditability. Model
identities, question wording, class mappings, and training/readout distinctions
are explicit in code. The ten-epoch Laya packaging helper depends on the earlier
input packager; both are retained. Recorded environment metadata will accompany
any later result release rather than being inferred from a fresh installation.

The optional saved-feature adapter accepts a recorded surface-checkpoint identity
after verifying the companion feature archives. The original evaluator's strict
checkpoint-byte check remains the default for all other calls. This option does
not assert verification of an undistributed checkpoint.

The saved-result audit and figure builders require the separately held paired
outputs. They cannot reproduce the manuscript's numbers from source code alone.
Fresh raw-data execution additionally needs licensed STEP inputs, suitable
hardware, upstream model weights, and any applicable API access.

For plotting, install `requirements-figures.txt` separately. The annotation app
requires a separately generated blinded manifest and renders; its source alone
does not constitute completed independent assessments. The automatic
silhouette-proxy code is named explicitly and does not generate expert ratings.

## Rights

See `THIRD_PARTY_NOTICES.md`. No open-source software/data license has been
selected on the authors' behalf. Obtain upstream dependencies, source CAD data,
and model weights under their applicable licenses. Public source visibility
does not grant rights to redistribute those external resources.
