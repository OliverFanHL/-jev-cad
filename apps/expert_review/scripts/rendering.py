"""STEP tessellation and fixed-view rendering for the review app."""
from __future__ import annotations

from pathlib import Path

import cadquery as cq
import numpy as np
from PIL import Image, ImageChops
from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData
from vtkmodules.vtkFiltersCore import vtkPolyDataNormals
from vtkmodules.vtkIOImage import vtkPNGWriter
from vtkmodules.vtkRenderingCore import (vtkActor, vtkLight, vtkPolyDataMapper,
                                         vtkRenderer, vtkRenderWindow,
                                         vtkWindowToImageFilter)
import vtkmodules.vtkRenderingOpenGL2  # noqa: F401 - VTK backend registration


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
