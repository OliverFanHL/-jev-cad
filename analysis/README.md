# Figures and analysis

These scripts turn saved evaluation outputs into comparison tables, figures,
and CAD images. Run them from the repository root after preparing the inputs
described in the [workflow guide](../docs/REPRODUCTION.md).

```bash
python -m pip install -r requirements-figures.txt
```

| Script in `scripts/` | Produces |
| --- | --- |
| `build_evidence.py` | Classification tables, paired comparisons, and example figures |
| `build_clm_evidence.py` | CLM comparison tables and duplicate-sensitivity summaries |
| `build_ablation_figure.py` | Comparisons of Choice, Noul, and fused features |
| `build_class_error_figure.py` | Per-class error figure |
| `build_framework.py` | Method overview diagram |
| `build_input_example.py` | A CAD part and its compact model input |
| `build_expert_review_guide.py` | Review-interface illustration |
| `render_step_candidates.py` | Fixed-view STEP renders and a contact sheet |
| `build_visual_proxy_scores.py` | Automatic two-view silhouette similarity scores |
| `evaluate_visual_proxy.py` | Retrieval metrics from those automatic scores |

`cad_render.py` and `clm_manuscript_data.py` are shared helpers.
The review-interface illustration was previously named
`build_expert_review_guide_v10_compact.py`.

The scripts read baseline predictions from `analysis/evaluation/` and imported
CLM outputs from `outputs/study/clm/kaggle_import/`. Tables go in
`analysis/evidence/`; figures go in `analysis/figures/`. The public repository
does not include those result inputs.

For example, with saved classification results available:

```bash
python analysis/scripts/build_clm_evidence.py --clm-eval outputs/study/clm/kaggle_import/evaluation
python analysis/scripts/build_ablation_figure.py
```

CAD render and example scripts also need local STEP files, CadQuery, and VTK.
The renderer's part list and the framework diagram's STEP source path can be
adjusted to your local dataset.

For the automatic visual comparison:

```bash
python analysis/scripts/build_visual_proxy_scores.py
python analysis/scripts/evaluate_visual_proxy.py
```

The first command reads the selected pair list and the review app's renders,
then writes `recomputed/visual_proxy_scores.csv`. The second evaluates retrieval
using those scores. This measures silhouette similarity; use the
[review app](../apps/expert_review/README.md) to collect expert relevance grades.
