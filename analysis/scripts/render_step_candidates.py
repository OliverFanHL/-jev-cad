"""Render three held-out STEP solids at three reproducible metallic camera views.

Outputs nine individual PNGs and a labeled contact sheet in
analysis/figures/step_candidates/. The STEP geometry is loaded directly;
no generative or schematic substitute is used.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cadquery as cq
import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFont
from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData
from vtkmodules.vtkFiltersCore import vtkPolyDataNormals
from vtkmodules.vtkIOImage import vtkPNGWriter
from vtkmodules.vtkRenderingCore import (vtkActor, vtkLight, vtkPolyDataMapper,
                                         vtkRenderer, vtkRenderWindow,
                                         vtkWindowToImageFilter)
import vtkmodules.vtkRenderingOpenGL2  # noqa: F401 - VTK backend registration


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "step_candidates"
OUT.mkdir(parents=True, exist_ok=True)
PARTS = [
    ("bracket_556", "2408daa14ba0de294b93", ROOT.parent.parent / "TMCAD/bracket/556.stp"),
    ("gear_66", "1dcf73c1bc7182e9f9af", ROOT.parent.parent / "TMCAD/gear/66.stp"),
    ("pulley_328", "0bb26142b84c91d03539", ROOT.parent.parent / "TMCAD/pulley/328.stp"),
]
ALTERNATIVE_PARTS = [
    ("flange_499", "14394e605c4c31cddba3", ROOT.parent.parent / "TMCAD/flange/499.stp"),
    ("shaft_277", "1d13b5c5b3192f424767", ROOT.parent.parent / "TMCAD/shaft/277.stp"),
    ("screw_844", "09129836ea6c689d2934", ROOT.parent.parent / "TMCAD/bolt_screw/screw/844.stp"),
]
FIGURE2_PARTS = [
    ("nut_165", "08b6e4a49adf79f320db", ROOT.parent.parent / "TMCAD/nut/165.stp"),
]
VIEWS = [
    ("A", "three-quarter", (.95, -.80, .70)),
    ("B", "reverse oblique", (-.85, -.90, .95)),
    ("C", "raised axial", (.35, -.25, 1.70)),
]
SIZE = (840, 600)


def load_poly(path: Path, tolerance_fraction: float = .008) -> tuple[vtkPolyData, np.ndarray, float]:
    assert path.exists(), path
    imported = cq.importers.importStep(str(path))
    shape = imported.val()
    if not shape.Faces():
        # Some STEP files import as a sequence of empty compounds followed by
        # the actual solid. Selecting .val() alone would render a blank image.
        visible = [item for item in imported.objects if item.Faces()]
        if not visible:
            raise ValueError(f"No renderable B-Rep faces in {path}")
        shape = visible[0] if len(visible) == 1 else cq.Compound.makeCompound(visible)
    bounds = shape.BoundingBox()
    extent = max(float(bounds.xlen), float(bounds.ylen), float(bounds.zlen), 1e-6)
    verts, faces = shape.tessellate(max(extent * tolerance_fraction, 1e-5))
    xyz = np.asarray([v.toTuple() for v in verts], dtype=np.float64)
    tri = np.asarray(faces, dtype=np.int64)
    assert len(xyz) > 0 and len(tri) > 0
    poly = vtkPolyData()
    pts = vtkPoints()
    pts.SetData(numpy_to_vtk(xyz, deep=True))
    poly.SetPoints(pts)
    cells = vtkCellArray()
    cells.SetCells(len(tri), numpy_to_vtkIdTypeArray(
        np.column_stack((np.full(len(tri), 3, dtype=np.int64), tri)).ravel(), deep=True))
    poly.SetPolys(cells)
    normal = vtkPolyDataNormals()
    normal.SetInputData(poly)
    normal.SplittingOff()
    normal.ConsistencyOn()
    normal.Update()
    return normal.GetOutput(), xyz.mean(axis=0), extent


def render(poly: vtkPolyData, center: np.ndarray, extent: float,
           direction: tuple[float, float, float], path: Path) -> None:
    mapper = vtkPolyDataMapper()
    mapper.SetInputData(poly)
    actor = vtkActor()
    actor.SetMapper(mapper)
    prop = actor.GetProperty()
    prop.SetColor(.73, .77, .81)  # neutral satin steel
    prop.SetAmbient(.31)
    prop.SetDiffuse(.62)
    prop.SetSpecular(.46)
    prop.SetSpecularPower(38)
    renderer = vtkRenderer()
    renderer.SetBackground(1, 1, 1)
    renderer.AddActor(actor)

    key = vtkLight()
    key.SetLightTypeToSceneLight()
    key.SetPosition(*(center + extent * np.array([1.5, -1.1, 2.2])))
    key.SetFocalPoint(*center)
    key.SetIntensity(.75)
    renderer.AddLight(key)
    fill = vtkLight()
    fill.SetLightTypeToSceneLight()
    fill.SetPosition(*(center + extent * np.array([-1.3, .8, .9])))
    fill.SetFocalPoint(*center)
    fill.SetIntensity(.38)
    renderer.AddLight(fill)
    head = vtkLight()
    head.SetLightTypeToCameraLight()
    head.SetPosition(.8, .7, 1.0)
    head.SetFocalPoint(0, 0, 0)
    head.SetIntensity(.30)
    renderer.AddLight(head)

    camera = renderer.GetActiveCamera()
    camera.SetFocalPoint(*center)
    vec = np.asarray(direction, dtype=float)
    vec /= np.linalg.norm(vec)
    camera.SetPosition(*(center + extent * 3.2 * vec))
    camera.SetViewUp(0, 0, 1)
    camera.ParallelProjectionOn()
    renderer.ResetCamera()
    camera.SetParallelScale(camera.GetParallelScale() * 1.12)
    renderer.ResetCameraClippingRange()

    window = vtkRenderWindow()
    window.SetOffScreenRendering(1)
    window.SetSize(*SIZE)
    window.SetMultiSamples(8)
    window.AddRenderer(renderer)
    window.Render()
    capture = vtkWindowToImageFilter()
    capture.SetInput(window)
    capture.Update()
    writer = vtkPNGWriter()
    writer.SetFileName(str(path))
    writer.SetInputConnection(capture.GetOutputPort())
    writer.Write()
    window.Finalize()
    # Keep each candidate as a close-cropped, reusable image rather than a
    # mostly empty renderer canvas; aspect ratio and geometry are preserved.
    with Image.open(path) as raw:
        rgb = raw.convert("RGB")
        diff = ImageChops.difference(rgb, Image.new("RGB", rgb.size, "white"))
        bounds = diff.point(lambda value: 255 if value > 10 else 0).getbbox()
        assert bounds is not None, path
        margin = 24
        x0, y0, x1, y1 = bounds
        rgb.crop((max(0, x0-margin), max(0, y0-margin),
                  min(rgb.width, x1+margin), min(rgb.height, y1+margin))).save(path)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "arialbd.ttf" if bold else "arial.ttf"
    return ImageFont.truetype(str(Path("C:/Windows/Fonts") / name), size)


def contact_sheet(records: list[dict], out: Path, parts: list[tuple]) -> None:
    thumb_w, thumb_h = 520, 372
    gap, left, top = 20, 215, 68
    sheet = Image.new("RGB", (left + 3*thumb_w + 4*gap,
                              top + len(parts)*thumb_h + (len(parts)+1)*gap), "white")
    draw = ImageDraw.Draw(sheet)
    for col, (_, title, _) in enumerate(VIEWS):
        draw.text((left + col*(thumb_w+gap) + 12, 17), title,
                  fill="#20384A", font=font(22, bold=True))
    for row, (slug, _, _) in enumerate(parts):
        label = slug.replace("_", " / ")
        draw.text((20, top + row*(thumb_h+gap) + thumb_h//2 - 15), label,
                  fill="#20384A", font=font(23, bold=True))
        for col, (view_id, _, _) in enumerate(VIEWS):
            record = next(r for r in records if r["part"] == slug and r["view"] == view_id)
            with Image.open(out / record["image"]) as original:
                image = original.convert("RGB")
                image.thumbnail((thumb_w-14, thumb_h-14), Image.Resampling.LANCZOS)
            x = left + col*(thumb_w+gap)
            y = top + row*(thumb_h+gap)
            draw.rounded_rectangle((x, y, x+thumb_w, y+thumb_h), radius=12,
                                   outline="#D5DEE4", width=2)
            sheet.paste(image, (x+(thumb_w-image.width)//2,
                                y+(thumb_h-image.height)//2))
            draw.text((x+14, y+12), view_id, fill="#20384A", font=font(19, bold=True))
    sheet.save(out / "metallic_views_contact_sheet.png", optimize=True)


def render_set(parts: list[tuple], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for slug, sample_id, path in parts:
        poly, center, extent = load_poly(path)
        for view_id, view_name, direction in VIEWS:
            image_name = f"{slug}_{view_id}.png"
            render(poly, center, extent, direction, out / image_name)
            records.append({"part": slug, "sample_id": sample_id,
                            "source_step": str(path), "view": view_id,
                            "view_name": view_name, "camera_direction": direction,
                            "finish": "satin steel, two soft lights",
                            "image": image_name})
            print(image_name, flush=True)
    contact_sheet(records, out, parts)
    (out / "manifest.json").write_text(json.dumps(records, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", choices=("initial", "alternative", "figure2"),
                        default="initial")
    selection = parser.parse_args().set
    groups = {
        "initial": (PARTS, OUT),
        "alternative": (ALTERNATIVE_PARTS, OUT / "alternative_parts"),
        "figure2": (FIGURE2_PARTS, OUT / "figure2_nut"),
    }
    render_set(*groups[selection])
    if selection == "figure2":
        # Same square canvas as the LaTeX input-trace rendering, prepared as a
        # selectable replacement without changing the manuscript itself.
        with Image.open(OUT / "figure2_nut" / "nut_165_A.png") as source:
            candidate = source.convert("RGB")
            candidate.thumbnail((464, 464), Image.Resampling.LANCZOS)
        square = Image.new("RGB", (512, 512), "white")
        square.paste(candidate, ((512-candidate.width)//2, (512-candidate.height)//2))
        square.save(ROOT / "figures/renders/input_trace_render_metallic_candidate.png",
                    optimize=True)


if __name__ == "__main__":
    main()
