"""Build class-recall and paired-error figure from saved matched predictions.

Run from the repository root in the study plotting environment:
    python analysis/scripts/build_class_error_figure.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.append(str(HERE / "_vendor"))
import scienceplots  # noqa: E402,F401 - registers SciencePlots styles
from clm_manuscript_data import IMPORT, load_matched_summaries  # noqa: E402

COLORS = {"geometry": "#2C6E8F", "semantic": "#B46B31",
          "fusion": "#4F7862", "clm": "#80649C"}
PAIR_KEYS = [
    ("laya_ft_direct_choice", "laya_direct_choice"),
    ("descriptor_plus_geometry", "laya_ft_direct_choice"),
    ("geometry_embedding", "geometry_plus_jev"),
    ("descriptor_plus_geometry", "geometry_plus_laya_ft_choice_probabilities"),
    ("geometry_embedding", "geometry_plus_laya_ft_choice_probabilities"),
    ("geometry_embedding", "descriptor_plus_geometry"),
    ("descriptor_plus_jev", "descriptor_only"),
    ("geometry_plus_clm", "geometry_embedding"),
    ("geometry_plus_jev", "geometry_plus_clm"),
]


def saved_pairs() -> list[dict]:
    load_matched_summaries()
    eval_dir = ROOT / "evaluation" / "classification"
    methods = {name for pair in PAIR_KEYS for name in pair}
    frames = {name: pd.read_csv((IMPORT / "evaluation/classification" if name == "geometry_plus_clm"
                                 else eval_dir) / f"{name}.csv", dtype={"sample_id": str})
              .set_index("sample_id").sort_index() for name in methods}
    reference = frames["descriptor_plus_geometry"]
    assert len(reference) == 1960 and reference.index.is_unique
    for name, frame in frames.items():
        assert frame.index.equals(reference.index) and frame.label.equals(reference.label), name
    truth = reference.label
    pairs = []
    for a, b in PAIR_KEYS:
        ac, bc = frames[a].prediction.eq(truth), frames[b].prediction.eq(truth)
        pairs.append({
            "a": a, "b": b,
            "both_correct": int((ac & bc).sum()),
            "a_only": int((ac & ~bc).sum()),
            "b_only": int((~ac & bc).sum()),
            "both_wrong": int((~ac & ~bc).sum()),
        })
    (ROOT / "evidence" / "paired_outcomes.json").write_text(
        json.dumps(pairs, indent=2), encoding="utf-8"
    )
    return pairs


def build(recall: pd.DataFrame, pairs: list[dict], out: Path) -> None:
    assert len(recall) == 10 and len(pairs) == len(PAIR_KEYS)
    order = recall.sort_values("descriptor_plus_geometry")["class"].tolist()
    plotted = recall.set_index("class").loc[order]
    with plt.style.context(["science", "no-latex"]):
        plt.rcParams.update({
            "font.family": "Arial", "font.sans-serif": ["Arial"],
            "pdf.fonttype": 42, "ps.fonttype": 42,
            "font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8,
            "ytick.labelsize": 8, "axes.spines.top": False,
            "axes.spines.right": False, "savefig.bbox": "standard",
        })
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.2), dpi=150,
                                 gridspec_kw={"width_ratios": [1.38, 1]})
        yy = np.arange(len(order))
        series = [
            ("descriptor_plus_geometry", "Descriptor + surface", COLORS["fusion"], "o", -.24),
            ("laya_ft_direct_choice", "FT Laya Choice", COLORS["semantic"], "s", -.08),
            ("jev_all_questions", "JEV Noul probe", COLORS["geometry"], "^", .08),
            ("clm_all_questions", "CLM Noul probe", COLORS["clm"], "D", .24),
        ]
        ax = axes[0]
        for key, title, color, marker, offset in series:
            ax.scatter(plotted[key].to_numpy(), yy + offset, color=color,
                       marker=marker, s=30, label=title, zorder=3,
                       linewidths=.5, edgecolors="white")
        # Select values identify the most informative class-level contrasts.
        point_labels = [
            ("pulley", "clm_all_questions", .24, 6, -9),
            ("pulley", "descriptor_plus_geometry", -.24, 5, -9),
            ("coupling", "laya_ft_direct_choice", -.08, 5, -9),
        ]
        for cls, key, offset, dx, dy in point_labels:
            value = float(plotted.at[cls, key])
            row_y = order.index(cls) + offset
            ax.annotate(f"{value:.3f}", (value, row_y), xytext=(dx, dy),
                        textcoords="offset points", fontsize=6.8,
                        color="#273740", va="top", zorder=5)
        ax.set_xlim(.15, 1.04)
        ax.set_xticks([.2, .4, .6, .8, 1])
        ax.set_xlabel("Class recall")
        ax.set_yticks(yy, order)
        ax.set_ylim(-.6, len(order)-.4)
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=.17)
        ax.grid(axis="y", alpha=.09)
        ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.01),
                  ncol=2, fontsize=7, handletextpad=.35, columnspacing=.8)
        ax.text(-.08, 1.12, "a", transform=ax.transAxes, weight="bold", fontsize=9)
        ax.minorticks_off()
        ax.tick_params(axis="both", which="both", top=False, right=False)

        ax = axes[1]
        lookup = {(p["a"], p["b"]): p for p in pairs}
        comparisons = [
            ("S+JEV\nvs S", lookup[("geometry_embedding", "geometry_plus_jev")]["b_only"],
             lookup[("geometry_embedding", "geometry_plus_jev")]["a_only"]),
            ("S+CLM\nvs S", lookup[("geometry_plus_clm", "geometry_embedding")]["a_only"],
             lookup[("geometry_plus_clm", "geometry_embedding")]["b_only"]),
            ("S+JEV\nvs S+CLM", lookup[("geometry_plus_jev", "geometry_plus_clm")]["a_only"],
             lookup[("geometry_plus_jev", "geometry_plus_clm")]["b_only"]),
            ("D+S\nvs S", lookup[("geometry_embedding", "descriptor_plus_geometry")]["b_only"],
             lookup[("geometry_embedding", "descriptor_plus_geometry")]["a_only"]),
            ("D+S\nvs S+FT", lookup[("descriptor_plus_geometry", "geometry_plus_laya_ft_choice_probabilities")]["a_only"],
             lookup[("descriptor_plus_geometry", "geometry_plus_laya_ft_choice_probabilities")]["b_only"]),
        ]
        yy2 = np.arange(len(comparisons))
        height = .25
        ax.barh(yy2-height/2, [r[1] for r in comparisons], height=height,
                color=COLORS["fusion"], label="First only")
        ax.barh(yy2+height/2, [r[2] for r in comparisons], height=height,
                color=COLORS["semantic"], label="Second only")
        for j, (_, first, second) in enumerate(comparisons):
            ax.text(first+1.4, j-height/2, str(first), va="center", fontsize=7)
            ax.text(second+1.4, j+height/2, str(second), va="center", fontsize=7)
        ax.set_yticks(yy2, [r[0] for r in comparisons])
        ax.invert_yaxis()
        ax.set_xlim(0, 84)
        ax.set_xticks([0, 20, 40, 60, 80])
        ax.set_xlabel("Discordant test parts")
        ax.grid(axis="x", alpha=.17)
        ax.minorticks_off()
        ax.tick_params(axis="both", which="both", top=False, right=False)
        ax.text(-.08, 1.12, "b", transform=ax.transAxes, weight="bold", fontsize=9)
        ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.01),
                  ncol=2, fontsize=7, handletextpad=.35, columnspacing=.8)
        fig.subplots_adjust(left=.135, right=.985, bottom=.14, top=.85, wspace=.37)
        fig.savefig(out / "class_error_structure.pdf")
        fig.savefig(out / "class_error_structure.png", dpi=300)
        plt.close(fig)


if __name__ == "__main__":
    baseline = pd.read_csv(ROOT / "evidence" / "class_recall.csv")
    clm = pd.read_csv(IMPORT / "evidence/class_recall.csv")
    assert baseline[["class", "n"]].equals(clm[["class", "n"]])
    baseline["clm_all_questions"] = clm["clm_all_questions"]
    baseline.to_csv(ROOT / "evidence/class_recall_with_clm.csv", index=False)
    build(baseline, saved_pairs(), ROOT / "figures")
