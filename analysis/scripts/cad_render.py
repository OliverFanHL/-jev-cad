"""Offscreen VTK rendering of STEP tessellations for paper figures."""
from pathlib import Path

import numpy as np
from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData
from vtkmodules.vtkRenderingCore import vtkActor, vtkPolyDataMapper, vtkRenderer, vtkRenderWindow, vtkWindowToImageFilter
from vtkmodules.vtkIOImage import vtkPNGWriter
from vtkmodules.vtkFiltersCore import vtkPolyDataNormals
from jev_tmcad.geometry.descriptors import _load_shape
import vtkmodules.vtkRenderingOpenGL2  # noqa: F401 - VTK rendering backend


def render_step(path: str | Path, output: str | Path, width=800, height=560):
    shape = _load_shape(str(path))
    bb = shape.BoundingBox()
    extent = max(float(bb.xlen), float(bb.ylen), float(bb.zlen), 1e-6)
    vertices, triangles = shape.tessellate(max(extent * .008, 1e-5))
    xyz = np.asarray([v.toTuple() for v in vertices], dtype=np.float64)
    tri = np.asarray(triangles, dtype=np.int64)
    cells = np.column_stack((np.full(len(tri), 3, dtype=np.int64), tri)).ravel()
    poly = vtkPolyData()
    points = numpy_to_vtk(xyz, deep=True)
    from vtkmodules.vtkCommonCore import vtkPoints
    vtk_points = vtkPoints()
    vtk_points.SetData(points)
    poly.SetPoints(vtk_points)
    faces = vtkCellArray()
    faces.SetCells(len(tri), numpy_to_vtkIdTypeArray(cells, deep=True))
    poly.SetPolys(faces)
    norm = vtkPolyDataNormals()
    norm.SetInputData(poly)
    norm.SplittingOff()
    norm.ConsistencyOn()
    norm.Update()
    mapper = vtkPolyDataMapper()
    mapper.SetInputConnection(norm.GetOutputPort())
    actor = vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(.32, .58, .69)
    actor.GetProperty().SetAmbient(.32)
    actor.GetProperty().SetDiffuse(.65)
    actor.GetProperty().SetSpecular(.12)
    actor.GetProperty().SetSpecularPower(24)
    renderer = vtkRenderer()
    renderer.SetBackground(1, 1, 1)
    renderer.AddActor(actor)
    center = xyz.mean(axis=0)
    span = np.ptp(xyz, axis=0)
    distance = max(span) * 3.0
    camera = renderer.GetActiveCamera()
    camera.SetFocalPoint(*center)
    camera.SetPosition(*(center + distance * np.array([.95, -.80, .70])))
    camera.SetViewUp(0, 0, 1)
    camera.ParallelProjectionOn()
    renderer.ResetCamera()
    renderer.ResetCameraClippingRange()
    window = vtkRenderWindow()
    window.SetOffScreenRendering(1)
    window.SetSize(width, height)
    window.SetMultiSamples(8)
    window.AddRenderer(renderer)
    window.Render()
    image = vtkWindowToImageFilter()
    image.SetInput(window)
    image.Update()
    writer = vtkPNGWriter()
    writer.SetFileName(str(output))
    writer.SetInputConnection(image.GetOutputPort())
    writer.Write()
    window.Finalize()

