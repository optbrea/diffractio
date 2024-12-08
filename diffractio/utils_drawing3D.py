# !/usr/bin/env python3

# ----------------------------------------------------------------------
# Name:        utils_drawing3D.py
# Purpose:     Utility functions for 3D drawing operations. Pyvista has been used for the implementation
#
# Author:      Luis Miguel Sanchez Brea
#
# Created:     2024
# Licence:     GPLv3
# ----------------------------------------------------------------------

# -*- coding: utf-8 -*-


import numpy as np

import pyvista
import pyvista as pv
from pyvista.core.utilities.helpers import wrap

from .config import CONF_DRAWING, Draw_XYZ_Options, Draw_pyvista_Options, video_isovalue_Options
from .utils_drawing import normalize_draw


def load_stl(filename: str, has_draw: bool = False, verbose: bool = False):  # -> tuple[MultiBlock | UnstructuredGrid | DataSet | pyvista_n...:
    """
    load_stl _summary_

    _extended_summary_

    Args:
        filename (str): _description_
        has_draw (bool, optional): _description_. Defaults to False.
        verbose (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: mesh
    """

    mesh = pv.read(filename)

    if has_draw:
        mesh.plot()

    if verbose:
        print(filename)
        print("bounds")
        print(mesh.bounds)
        print("volume")
        print(mesh.volume)
        print("center")
        print(mesh.center)
    return mesh


def voxelize_volume_diffractio(self, mesh, refractive_index, check_surface=True):
    """Voxelize mesh to create a RectilinearGrid voxel volume.

    Creates a voxel volume that encloses the input mesh and discretizes the cells
    within the volume that intersect or are contained within the input mesh.
    ``InsideMesh``, an array in ``cell_data``, is ``1`` for cells inside and ``0`` outside.

    Parameters
    ----------
    mesh : pyvista.DataSet
        Mesh to voxelize.


    check_surface : bool, default: True
        Specify whether to check the surface for closure. If on, then the
        algorithm first checks to see if the surface is closed and
        manifold. If the surface is not closed and manifold, a runtime
        error is raised.

    Returns
    -------
    pyvista.RectilinearGrid
        RectilinearGrid as voxelized volume with discretized cells.

    See Also
    --------
    pyvista.voxelize
    pyvista.DataSetFilters.select_enclosed_points

    """

    mesh = wrap(mesh)

    # check and pre-process input mesh
    surface = mesh.extract_geometry()  # filter preserves topology
    if not surface.faces.size:
        # we have a point cloud or an empty mesh
        raise ValueError('Input mesh must have faces for voxelization.')
    if not surface.is_all_triangles:
        # reduce chance for artifacts, see gh-1743
        surface.triangulate(inplace=True)

    # Create a RectilinearGrid
    voi = pyvista.RectilinearGrid(self.y, self.x, self.z)

    # get part of the mesh within the mesh's bounding surface.
    selection = voi.select_enclosed_points(surface, tolerance=0.0, check_surface=check_surface)
    mask_vol = selection.point_data['SelectedPoints'].view(np.bool_)

    data = np.array(mask_vol.tolist())
    data = data.reshape(len(self.z), len(self.y), len(self.x))

    # Get voxels that fall within input mesh boundaries
    cell_ids = np.unique(voi.extract_points(np.argwhere(mask_vol))["vtkOriginalCellIds"])
    # Create new element of grid where all cells _within_ mesh boundary are
    # given new name 'MeshCells' and a discrete value of 1
    voi['InsideMesh'] = self.n_background*np.ones(voi.n_cells)
    voi['InsideMesh'][cell_ids] = refractive_index

    volume_n = self.n_background + (refractive_index-self.n_background)*data
    self.n = volume_n.reshape(len(self.z), len(self.x), len(self.y))

    self.n = np.transpose(self.n, axes=(2, 1, 0))

    return self, voi  # , selection, mask_vol, cell_ids, volume_n


def draw(
    self,
    kind: Draw_XYZ_Options = "intensity",
    drawing: Draw_pyvista_Options = "volume",
    has_grid: bool = False,
    filename: str = "",
    logarithm: float = 0.,
    **kwargs
):
    """_summary_

    Args:
        kind (str, optional): "intensity" or "refractive_index". Defaults to 'refractive_index'.
        drawing (str, optional): volume, clip, slices, projections. Defaults to 'volume'.
        has_grid (bool, optional): add grid. Defaults to False.
        filename (str, optional): saves images: html, png or svg. Defaults to ''.
    """

    x_center = (self.x[-1]+self.x[0])/2
    y_center = (self.y[-1]+self.y[0])/2
    z_center = (self.z[-1]+self.z[0])/2

    len_x = len(self.x)
    len_y = len(self.y)
    len_z = len(self.z)

    delta_x = self.x[1]-self.x[0]
    delta_y = self.y[1]-self.y[0]
    delta_z = self.z[1]-self.z[0]

    if 'opacity' in kwargs.keys():
        opacity = kwargs["opacity"]
    else:
        opacity = 'sigmoid'

    if 'dimensions' in kwargs.keys():
        dimensions = kwargs["dimensions"]
    else:
        dimensions = (len_y, len_z, len_x)

    if 'scale' in kwargs.keys():
        scale = kwargs["scale"]
    else:
        scale = (len_y, len_z, len_x)

    if 'cmap' in kwargs.keys():
        cmap = kwargs["cmap"]
    else:
        cmap = 'gist_heat'

    if 'spacing' in kwargs.keys():
        spacing = kwargs["spacing"]
    else:
        spacing = np.array((delta_y, delta_x, delta_z))

    if 'cpos' in kwargs.keys():
        cpos = kwargs["cpos"]
    else:
        cpos = [(540, -617, 180),
                (128, 126., 111.),
                (-0, 0, 0)]

    if 'background_color' in kwargs.keys():
        background_color = kwargs["background_color"]
    else:
        background_color = (1., 1., 1.)

    if 'camera_position' in kwargs.keys():
        camera_position = kwargs["camera_position"]
    else:
        camera_position = 'xy'

    grid = pv.ImageData(dimensions=dimensions, spacing=spacing)

    if kind == "intensity":
        intensity = self.intensity()
        intensity = intensity/intensity.max()
        intensity = normalize_draw(intensity, logarithm)

        data = intensity

    elif kind == "refractive_index":
        data = np.abs(self.n)
        cmap = CONF_DRAWING['color_n']

    else:
        print("bad kind in draw_XYZ")

    data = data.reshape((len(self.y), len(self.x), len(self.z)))

    if drawing == "volume":
        data = np.transpose(data, axes=(2, 0, 1))

        pl = pyvista.Plotter()

        vol = pl.add_volume(data, cmap=cmap, opacity=opacity, shade=False)

        pl.set_scale(
            xscale=1/scale[2],
            yscale=1/scale[0],
            zscale=1/scale[1],
            reset_camera=True,
            render=True)
        pl.set_position((1, 1, 1))
        pl.reset_camera(self)

    elif drawing == "clip":

        grid["scalars"] = np.transpose(data, axes=(2, 0, 1)).flatten()

        pl = pyvista.Plotter()
        pl.add_volume_clip_plane(grid, normal="y", opacity=opacity, cmap=cmap)
        pl.add_volume_clip_plane(grid, normal="z", opacity=opacity, cmap=cmap)
        pl.add_volume_clip_plane(grid, normal="x", opacity=opacity, cmap=cmap)
        pl.set_scale(
            xscale=1/scale[2],
            yscale=1/scale[0],
            zscale=1/scale[1],
            reset_camera=True,
            render=True)
        pl.camera_position = camera_position

    elif drawing == "slices":

        grid["scalars"] = np.transpose(data, axes=(2, 0, 1)).flatten()

        pl = pyvista.Plotter()
        slice = grid.slice_orthogonal()
        dargs = dict(cmap=cmap)
        pl.set_scale(
            xscale=1/scale[2],
            yscale=1/scale[0],
            zscale=1/scale[1],
            reset_camera=True,
            render=True,
        )
        pl.add_mesh(slice, **dargs)
        # pl.camera_position = camera_position

    elif drawing == "projections":
        data = np.transpose(data, axes=(0, 1, 2))  # prueba y error - bien en volume
        # data = np.transpose(data, axes=(0,1,2)) # prueba y error - bien en volume

        grid["scalars"] = np.transpose(data, axes=(2, 0, 1)).flatten()

        pl = pyvista.Plotter(shape=(2, 2))
        dargs = dict(cmap=cmap)
        slice1 = grid.slice_orthogonal(x=0, y=0)
        slice2 = grid.slice_orthogonal(x=0, z=0)
        slice3 = grid.slice_orthogonal(y=0, z=0)
        slice4 = grid.slice_orthogonal()

        # XYZ - show 3D scene first
        pl.subplot(1, 1)
        pl.add_mesh(slice4, **dargs)
        pl.set_scale(
            xscale=1/scale[2],
            yscale=1/scale[0],
            zscale=1/scale[1],
            reset_camera=True,
            render=True,
        )
        # XY
        pl.subplot(0, 0)
        pl.add_mesh(slice1, **dargs)
        pl.set_scale(
            xscale=1/scale[2],
            yscale=1/scale[0],
            zscale=1/scale[1],
            reset_camera=True,
            render=True,
        )

        if has_grid is True:
            pl.show_grid()
        pl.camera_position = "xy"
        pl.enable_parallel_projection()
        # ZY
        pl.subplot(0, 1)
        pl.add_mesh(slice2, **dargs)
        pl.set_scale(
            xscale=1/scale[2],
            yscale=1/scale[0],
            zscale=1/scale[1],
            reset_camera=True,
            render=True,
        )
        if has_grid is True:
            pl.show_grid()
        pl.camera_position = "zx"
        pl.enable_parallel_projection()
        # XZ
        pl.subplot(1, 0)
        pl.add_mesh(slice3, **dargs)
        pl.set_scale(
            xscale=1/scale[2],
            yscale=1/scale[0],
            zscale=1/scale[1],
            reset_camera=True,
            render=True,
        )
        if has_grid is True:
            pl.show_grid()
        pl.camera_position = "zy"
        pl.enable_parallel_projection()

    elif drawing == 'video_isovalue':
        # data = np.transpose(data, axes=(0,1,2)) # prueba y error - bien en volume

        grid["scalars"] = np.transpose(data, axes=(2, 0, 1)).flatten()

        pl = pv.Plotter()
        pl.set_scale(
            xscale=scale[1],
            yscale=scale[0],
            zscale=scale[2],
            reset_camera=True,
            render=True,
        )

        vol = pl.add_volume(grid, opacity=opacity, cmap=cmap)
        vol.prop.interpolation_type = "linear"

        values = np.linspace(0.05 * data.max(), data.max(), num=25)
        surface = grid.contour(values[:1])

        surfaces = [grid.contour([v]) for v in values]

        surface = surfaces[0].copy()
        plotter = pv.Plotter(off_screen=True)

        # Open a movie file
        plotter.open_gif(filename)

        # Add initial mesh
        plotter.add_mesh(
            surface,
            opacity=opacity,
            cmap=cmap,
            clim=grid.get_data_range(),
            show_scalar_bar=False,
        )
        # Add outline for reference
        plotter.add_mesh(grid.outline_corners(), color="r")

        # print('Orient the view, then press "q" to close window and produce movie')

        # initial render and do NOT close
        plotter.show(auto_close=True)

        # Run through each frame
        for surf in surfaces:
            surface.copy_from(surf)
            plotter.write_frame()  # Write this frame
        # Run through backwards
        for surf in surfaces[::-1]:
            surface.copy_from(surf)
            plotter.write_frame()  # Write this frame

        # Be sure to close the plotter when finished
        plotter.close()

    # pl.view_isometric()
    # pl.show_axes()
    # pl.show_bounds()
    pl.background_color = background_color

    if has_grid is True:
        pl.show_grid()

    pl.show()

    if filename != "":
        if filename[-3:] == "svg":
            pl.save_graphic(filename)
        elif filename[-4:] == "html":
            pl.export_html(filename)
        elif filename[-3:] == "png":
            pl.screenshot(filename)

    return pl


def video_isovalue(self, filename: str, kind: video_isovalue_Options = "refractive_index", **kwargs):
    """_summary_

    Args:
        filename (str): filename. Defaults to ''.
        kind (str, optional): "intensity" or "refractive_index". Defaults to 'refractive_index'.
    """

    x_center = (self.x[-1]+self.x[0])/2
    y_center = (self.y[-1]+self.y[0])/2
    z_center = (self.z[-1]+self.z[0])/2

    len_x = len(self.x)
    len_y = len(self.y)
    len_z = len(self.z)

    delta_x = self.x[1]-self.x[0]
    delta_y = self.y[1]-self.y[0]
    delta_z = self.z[1]-self.z[0]

    if 'opacity' in kwargs.keys():
        opacity = kwargs["opacity"]
    else:
        opacity = 'sigmoid'

    if 'dimensions' in kwargs.keys():
        dimensions = kwargs["dimensions"]
    else:
        dimensions = (len_y, len_z, len_x)

    if 'scale' in kwargs.keys():
        scale = kwargs["scale"]
    else:
        scale = (len_y, len_z, len_x)

    if 'cmap' in kwargs.keys():
        cmap = kwargs["cmap"]
    else:
        cmap = 'hot'

    if 'spacing' in kwargs.keys():
        spacing = kwargs["spacing"]
    else:
        spacing = np.array((delta_y, delta_x, delta_z))

    if 'cpos' in kwargs.keys():
        cpos = kwargs["cpos"]
    else:
        cpos = [(540, -617, 180),
                (128, 126., 111.),
                (-0, 0, 0)]

    if 'background_color' in kwargs.keys():
        background_color = kwargs["background_color"]
    else:
        background_color = (1., 1., 1.)

    if 'camera_position' in kwargs.keys():
        camera_position = kwargs["camera_position"]
    else:
        camera_position = "xy"

    grid = pv.ImageData(dimensions=dimensions, spacing=spacing)

    if kind == "intensity":
        intensity = self.intensity()
        intensity /= intensity.max()
        data = intensity

        data = intensity
    elif kind == "refractive_index":
        data = self.n
        data = data-data.min()
        data /= data.max()
        cmap = CONF_DRAWING['color_n']
        print("refractive_index")

    print(data.min(), data.max())

    grid["scalars"] = data.flatten()

    pl = pv.Plotter()
    pl.set_scale(xscale=scale[1], yscale=scale[0], zscale=scale[2], reset_camera=True, render=True)

    vol = pl.add_volume(grid, opacity=opacity, cmap=cmap)
    vol.prop.interpolation_type = "linear"

    values = np.linspace(0.05 * data.max(), data.max(), num=25)
    surface = grid.contour(values[:1])

    surfaces = [grid.contour([v]) for v in values]

    surface = surfaces[0].copy()
    print(surface)
    plotter = pv.Plotter(off_screen=True)

    # Open a movie file
    plotter.open_gif(filename)

    # Add initial mesh
    plotter.add_mesh(
        surface,
        opacity=opacity,
        cmap=cmap,
        clim=grid.get_data_range(),
        show_scalar_bar=False,
    )
    # Add outline for reference
    plotter.add_mesh(grid.outline_corners(), color="r")

    # print('Orient the view, then press "q" to close window and produce movie')

    # initial render and do NOT close
    plotter.show(auto_close=True)

    # Run through each frame
    for surf in surfaces:
        surface.copy_from(surf)
        plotter.write_frame()  # Write this frame
    # Run through backwards
    for surf in surfaces[::-1]:
        surface.copy_from(surf)
        plotter.write_frame()  # Write this frame

    # Be sure to close the plotter when finished
    plotter.close()
    pl.close()


def show_stl(filename):

    mesh = pv.read(filename)
    mesh.plot()
