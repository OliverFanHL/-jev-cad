"""Build the method overview from one real held-out STEP part and its saved mesh.

The two source assets are exported independently for reuse and inspection.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
FIG = ROOT / "figures"
SAMPLE = "2408daa14ba0de294b93"
STEP_SOURCE = PROJECT.parent / "TMCAD" / "bracket" / "556.stp"
RENDER_SOURCE = FIG / "step_candidates" / "bracket_556_A.png"
MESH_SOURCE = PROJECT / "outputs" / "study" / "full" / "mesh" / f"{SAMPLE}.npz"
NAVY = "#18334D"
BLUE = "#2C6E8F"
TEAL = "#287E81"
ORANGE = "#BD7135"
PALE_BLUE = "#F1F6FA"
PALE_TEAL = "#EEF8F6"
PALE_ORANGE = "#FFF7EF"
LINE = "#C7D6E0"


def crop_white(image: Image.Image, margin: int = 16) -> Image.Image:
    rgb = image.convert("RGB")
    diff = ImageChops.difference(rgb, Image.new("RGB", rgb.size, "white"))
    box = diff.point(lambda value: 255 if value > 12 else 0).getbbox()
    assert box is not None
    left, top, right, bottom = box
    return rgb.crop((max(0, left-margin), max(0, top-margin),
                     min(rgb.width, right+margin), min(rgb.height, bottom+margin)))


def build_step_asset() -> Path:
    assert STEP_SOURCE.exists() and RENDER_SOURCE.exists(), "STEP source/render missing"
    with Image.open(RENDER_SOURCE) as im:
        cropped = crop_white(im)
        cropped.save(FIG / "method_step_bracket_556.png", optimize=True)
        cropped.save(FIG / "method_step_bracket_556.pdf", "PDF", resolution=300.0)
    return FIG / "method_step_bracket_556.png"


def build_surface_asset() -> Path:
    with np.load(MESH_SOURCE) as data:
        features = data["features"].copy()
        knn = data["knn"].copy()
    assert features.shape == (512, 6) and knn.shape == (512, 16)
    xyz = features[:, :3]
    assert np.isfinite(xyz).all()
    # Orthographic projection parallel to the STEP renderer's fixed camera.
    view = np.array([.95, -.80, .70], dtype=float)
    view /= np.linalg.norm(view)
    right = np.cross(np.array([0., 0., 1.]), view)
    right /= np.linalg.norm(right)
    up = np.cross(view, right)
    projected = np.column_stack((xyz @ right, xyz @ up))
    depth = xyz @ view
    focus = int(np.argmin(np.linalg.norm(projected - np.median(projected, axis=0), axis=1)))
    neighbors = knn[focus]
    with plt.rc_context({"font.family": "Arial", "savefig.bbox": "standard"}):
        fig, ax = plt.subplots(figsize=(3.2, 2.2), dpi=300)
        ax.scatter(projected[:, 0], projected[:, 1], c=depth, cmap="Blues",
                   s=9, linewidths=0, alpha=.85, zorder=2)
        for j in neighbors:
            ax.plot(projected[[focus, j], 0], projected[[focus, j], 1],
                    color=ORANGE, lw=.55, alpha=.55, zorder=3)
        ax.scatter(projected[neighbors, 0], projected[neighbors, 1],
                   color=ORANGE, s=15, edgecolor="white", linewidth=.35, zorder=4)
        ax.scatter(*projected[focus], color="#A64331", s=30,
                   edgecolor="white", linewidth=.6, zorder=5)
        ax.set_aspect("equal")
        ax.set_axis_off()
        span = np.ptp(projected, axis=0)
        mid = (projected.min(axis=0) + projected.max(axis=0)) / 2
        ax.set_xlim(mid[0] - span[0]*.55, mid[0] + span[0]*.55)
        ax.set_ylim(mid[1] - span[1]*.55, mid[1] + span[1]*.55)
        fig.subplots_adjust(0, 0, 1, 1)
        fig.savefig(FIG / "method_surface_bracket_556.pdf")
        fig.savefig(FIG / "method_surface_bracket_556.png", dpi=300)
        plt.close(fig)
    return FIG / "method_surface_bracket_556.png"


def box(ax, x, y, w, h, title, detail, fc, ec):
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=1.4",
                           fc=fc, ec=ec, lw=1.15)
    ax.add_patch(patch)
    ax.text(x+2.2, y+h-3.2, title, ha="left", va="top", color=NAVY,
            fontsize=8.8, fontweight="bold")
    ax.text(x+2.2, y+3.0, detail, ha="left", va="bottom", color=NAVY,
            fontsize=7.2, linespacing=1.15)


def arrow(ax, start, end, color):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=9,
                                 lw=1.45, color=color, connectionstyle="arc3,rad=0"))


def build_overview(step_path: Path, surface_path: Path) -> None:
    with plt.rc_context({"font.family": "Arial", "font.sans-serif": ["Arial"],
                         "pdf.fonttype": 42, "savefig.bbox": "standard"}):
        fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=300)
        fig.patch.set_facecolor("white")
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 58)
        ax.axis("off")
        fig.subplots_adjust(0, 0, 1, 1)
        ax.text(2, 55.4, "ONE HELD-OUT STEP PART", color=NAVY, fontsize=8.5,
                fontweight="bold", va="center")
        ax.text(28, 55.4, "TWO GEOMETRY REPRESENTATIONS", color=NAVY,
                fontsize=8.5, fontweight="bold", va="center")
        ax.text(78.5, 55.4, "MATCHED READOUTS", color=NAVY,
                fontsize=8.5, fontweight="bold", va="center")
        ax.plot([2, 98], [52.7, 52.7], color=LINE, lw=.85)
        ax.add_patch(FancyBboxPatch((2, 13.2), 22.4, 35.8,
                                    boxstyle="round,pad=0.01,rounding_size=1.4",
                                    fc="white", ec=LINE, lw=1.1))
        step = np.asarray(Image.open(step_path))
        ax.imshow(step, extent=(3.1, 23.3, 21, 43.8), aspect="auto", zorder=2)
        ax.text(13.2, 17.0, "STEP B-Rep", ha="center", color=NAVY,
                fontsize=8.4, fontweight="bold")
        ax.text(13.2, 14.5, "bracket/556", ha="center", color="#526473", fontsize=7.1)
        box(ax, 29, 32.5, 20, 16, "Global geometry", "74 measured descriptors\n$d$", PALE_ORANGE, ORANGE)
        box(ax, 54, 32.5, 19.5, 16, "Typed request", "14-field JSON state\nChoice + 22 Noul", PALE_ORANGE, ORANGE)
        box(ax, 78.5, 32.5, 19.5, 16, "Typed responses", "JEV / base Laya / FT Laya\nChoice $p$, Noul $h$", PALE_ORANGE, ORANGE)
        ax.add_patch(FancyBboxPatch((29, 7.9), 20, 19.5,
                                    boxstyle="round,pad=0.01,rounding_size=1.4",
                                    fc=PALE_TEAL, ec=TEAL, lw=1.1))
        surface = np.asarray(Image.open(surface_path))
        ax.imshow(surface, extent=(31.8, 46.2, 12.0, 24.1), aspect="auto", zorder=2)
        ax.text(39, 10.6, "512 surface points · 16-NN", ha="center",
                color=NAVY, fontsize=7.1)
        box(ax, 54, 7.9, 19.5, 19.5, "Local encoder", "Two EdgeBlocks\n128-D embedding $g$", PALE_TEAL, TEAL)
        box(ax, 78.5, 7.9, 19.5, 19.5, "Evaluation", "Direct Choice; probes\nand feature fusion", PALE_BLUE, BLUE)
        arrow(ax, (24.4, 34.8), (29, 40.5), ORANGE)
        arrow(ax, (24.4, 26), (29, 18), TEAL)
        arrow(ax, (49, 40.5), (54, 40.5), ORANGE)
        arrow(ax, (73.5, 40.5), (78.5, 40.5), ORANGE)
        arrow(ax, (49, 17.5), (54, 17.5), TEAL)
        arrow(ax, (73.5, 17.5), (78.5, 17.5), TEAL)
        arrow(ax, (88.2, 32.5), (88.2, 27.4), ORANGE)
        # The original descriptor vector is also evaluated directly and in fusion.
        ax.plot([46, 46, 76.3, 76.3], [32.5, 29.8, 29.8, 25.0],
                color=ORANGE, lw=1.0, ls=(0, (3, 2)), alpha=.85)
        arrow(ax, (76.3, 25.0), (78.5, 25.0), ORANGE)
        ax.text(61.2, 30.4, "$d$ probe / fusion", ha="center", va="bottom",
                color=ORANGE, fontsize=6.7)
        ax.text(50, 3.2, "Matched part IDs · common train / validation / test split · held-out classification",
                ha="center", va="center", fontsize=7.2, color="#526473")
        fig.savefig(FIG / "method_overview.pdf")
        fig.savefig(FIG / "method_overview.png", dpi=300)
        plt.close(fig)


if __name__ == "__main__":
    build_overview(build_step_asset(), build_surface_asset())
    print(f"Built method overview and real-sample assets for {SAMPLE}.")
