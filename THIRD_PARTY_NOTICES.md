# Third-party resources and release scope

This public collection contains source code only. Companion derived analysis outputs remain unpublished.
It does not redistribute the original STEP dataset, language-model base weights,
CLM head, large adapted Laya checkpoint, or third-party libraries as source.

- JEV is accessed through the external TypeSafe SDK/service. Service terms and
  availability govern fresh API execution.
- Laya is an upstream project: https://github.com/NandhaKishorM/laya . The
  recorded base model is `convaiinnovations/laya`, `typed-decisions`, revision
  `5e7b2b1b8ca2ecdd3f2322d94069c9b6ce7e844b`. Upstream code/model licenses apply.
- The CLM evaluation uses `contrastive-lm` 0.1.0, the released
  `Contrastive-LM/CLM-v0.1-8B` head and `Qwen/Qwen3-8B`. Check those upstream
  code/model terms before acquisition or redistribution.
- CAD processing uses CadQuery/OpenCASCADE; local learning uses PyTorch;
  downstream probes use scikit-learn. Figures use Matplotlib, SciencePlots,
  Pillow, and, where required for CAD rendering, VTK. Install these separately
  under their respective licenses. No vendored dependency tree is included.
- Raw TMCAD access and licensing must be obtained from the actual dataset
  source. This release does not grant rights to redistribute source solids.

The authors have not yet selected an explicit software or derived-data reuse
license for this collection. No MIT/Apache/Creative Commons license, funding
ownership permission, or archival DOI is inferred from public GitHub visibility.
Before a formal archived release, the authors should confirm the applicable
rights and select the intended code and derived-data licenses.
