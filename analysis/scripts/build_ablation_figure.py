"""Build Figure 4 from saved JEV and completed CLM classification results.

Read saved evaluator metrics to compare representation combinations.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
import scienceplots  # noqa: E402,F401 - registers the SciencePlots style
from clm_manuscript_data import load_matched_summaries  # noqa: E402

baseline, clm = load_matched_summaries()
(ROOT / "evidence").mkdir(parents=True, exist_ok=True)
(ROOT / "figures").mkdir(parents=True, exist_ok=True)
rows = [
    ("JEV Choice probabilities, probed", "jev_choice_probabilities", baseline),
    ("CLM Choice probabilities, probed", "clm_choice_probabilities", clm),
    ("JEV geometric Noul, probed", "jev_geometry_questions", baseline),
    ("CLM geometric Noul, probed", "clm_geometry_questions", clm),
    ("JEV full Noul signature, probed", "jev_all_questions", baseline),
    ("CLM full Noul signature, probed", "clm_all_questions", clm),
    ("Surface + JEV geometric Noul", "geometry_plus_jev_geometry_questions", baseline),
    ("Surface + CLM geometric Noul", "geometry_plus_clm_geometry_questions", clm),
    ("Surface + JEV full Noul", "geometry_plus_jev", baseline),
    ("Surface + CLM full Noul", "geometry_plus_clm", clm),
    ("Descriptor + JEV full Noul", "descriptor_plus_jev", baseline),
    ("Descriptor + CLM full Noul", "descriptor_plus_clm", clm),
]
assert len({source[key]["n"] for _, key, source in rows}) == 1
evidence = [{"label": label, "method": key,
             "accuracy": source[key]["accuracy"], "macro_f1": source[key]["macro_f1"]}
            for label, key, source in rows]
(ROOT / "evidence/typed_output_ablations.json").write_text(
    json.dumps(evidence, indent=2), encoding="utf-8"
)

accuracy = np.array([row["accuracy"] for row in evidence])
macro_f1 = np.array([row["macro_f1"] for row in evidence])
y = np.array([0, .72, 1.75, 2.47, 3.5, 4.22, 5.45, 6.17,
              7.2, 7.92, 9.15, 9.87])

with plt.style.context(["science", "no-latex"]):
    plt.rcParams.update({
        "font.family": "Arial", "font.sans-serif": ["Arial"],
        "pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 9,
        "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 7.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "savefig.bbox": "standard",
    })
    fig, ax = plt.subplots(figsize=(7.2, 5.0), dpi=150)
    for start, end in ((-.4, 1.1), (1.35, 2.85), (3.1, 4.6)):
        ax.axhspan(start, end, color="#F7F4EF", zorder=0)
    for line in (1.35, 3.1, 4.95, 6.85, 8.65):
        ax.axhline(line, color="#B9C4CA", linewidth=.65, zorder=1)
    ax.hlines(y, macro_f1, accuracy, color="#A7B4B9", linewidth=1.1, zorder=2)
    ax.scatter(accuracy, y, s=30, marker="o", color="#4F7862", edgecolor="white",
               linewidth=.5, label="Accuracy", zorder=4)
    ax.scatter(macro_f1, y, s=31, marker="D", color="#B46B31", edgecolor="white",
               linewidth=.5, label="Macro F1", zorder=4)
    for score_acc, score_f1, row_y in zip(accuracy, macro_f1, y):
        ax.annotate(f"{score_f1:.3f}", (score_f1, row_y), xytext=(-4, 0),
                    textcoords="offset points", ha="right", va="center",
                    color="#9A5829", fontsize=6.5, zorder=5)
        ax.annotate(f"{score_acc:.3f}", (score_acc, row_y), xytext=(4, 0),
                    textcoords="offset points", ha="left", va="center",
                    color="#3D6452", fontsize=6.5, zorder=5)
    ax.set_yticks(y, [label for label, _, _ in rows])
    ax.set_ylim(10.35, -.55)
    ax.set_xlim(.30, 1.0)
    ax.set_xticks([.4, .5, .6, .7, .8, .9, 1.0])
    ax.set_xlabel("Held-out score")
    ax.grid(axis="x", color="#B8C5CB", alpha=.35, linewidth=.7)
    ax.set_axisbelow(True)
    ax.minorticks_off()
    ax.tick_params(axis="both", which="both", top=False, right=False)
    ax.legend(frameon=False, ncol=2, loc="lower left", bbox_to_anchor=(0, 1.015),
              columnspacing=1.5, handletextpad=.35, fontsize=8)
    fig.subplots_adjust(left=.34, right=.975, top=.88, bottom=.11)
    for suffix, kwargs in (("pdf", {}), ("png", {"dpi": 300})):
        fig.savefig(ROOT / "figures" / f"typed_output_ablations.{suffix}", **kwargs)
    plt.close(fig)
