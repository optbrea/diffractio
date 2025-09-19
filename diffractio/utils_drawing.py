# !/usr/bin/env python3


# ----------------------------------------------------------------------
# Name:        utils_drawing.py
# Purpose:     Utility functions for drawing operations
#
# Author:      Luis Miguel Sanchez Brea
#
# Created:     2024
# Licence:     GPLv3
# ----------------------------------------------------------------------


""" Functions for drawing """

# flake8: noqa


import os

from .utils_typing import npt, Any, NDArray,  NDArrayFloat, NDArrayComplex

import matplotlib.animation as manimation
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

from .__init__ import degrees, eps, mm
from .config import bool_raise_exception, CONF_DRAWING
from .utils_optics import field_parameters

percentage_intensity = CONF_DRAWING["percentage_intensity"]


def view_image(filename: str):
    """reproduces image

    Args:
        filename (str): filename
    """
    if filename != "" and filename is not None:
        mpimg.imread(filename)
        plt.show()


def concatenate_drawings(
    kind1: str = "png",
    kind2: str = "png",
    nx: int = 5,
    ny: int = 3,
    geometria_x: int = 256,
    geometria_y: int = 256,
    raiz: str = "fig4_nsensors_1",
    filename: str = "figura2.png",
    directory: str = "",
):
    files_list = os.listdir(directory)
    print(files_list)

    files_text = ""
    for file in sorted(files_list):
        if file[-3:] == kind1 and file[0: len(raiz)] == raiz:
            print(file)
            files_text = files_text + " " + directory + file

    os.system("cd " + directory)
    texto1 = "montage %s -tile %dx%d -geometry %d x %d -5-5 %s" % (
        files_text,
        nx,
        ny,
        geometria_x,
        geometria_y,
        filename,
    )

    print(texto1)
    os.system(texto1)

    print("Finished")


def draw2D(
    image: NDArrayFloat,
    x: NDArrayFloat,
    y: NDArrayFloat,
    xlabel: str = r"$x  (\mu m)$",
    ylabel: str = r"$y  (\mu m)$",
    title: str = "",
    color: str = "YlGnBu",  # YlGnBu  seismic
    interpolation: str = "bilinear",  # 'bilinear', 'nearest'
    scale: str = "scaled",
    reduce_matrix: str = "standard",
    range_scale: str = "um",
    verbose: bool = False,
) -> tuple:
    """makes a drawing of XY

    Args:
        image (numpy.array): image to draw
        x (numpy.array): positions x
        y (numpy.array): positions y
        xlabel (str): label for x
        ytlabel (str): label for y
        title (str): title
        color (str): color
        interpolation (str): 'bilinear', 'nearest'
        scale (str): kind of axis (None, 'equal', 'scaled', etc.)
        range_scale (str): 'um' o 'mm'
        verbose (bool): if True prints information

    Returns:
        id_fig: handle of figure
        IDax: handle of axis
        IDimage: handle of image
    """
    if reduce_matrix in (None, "", []):
        pass
    elif reduce_matrix == "standard":
        num_x = len(x)
        num_y = len(y)
        reduction_x = int(num_x / 500)
        reduction_y = int(num_y / 500)

        if reduction_x == 0:
            reduction_x = 1
        if reduction_y == 0:
            reduction_y = 1

        image = image[::reduction_x, ::reduction_y]
    else:
        image = image[:: reduce_matrix[0], :: reduce_matrix[1]]

    if verbose is True:
        print(("image size {}".format(image.shape)))

    id_fig = plt.figure()
    IDax = id_fig.add_subplot(111)

    if range_scale == "um":
        extension = (x[0], x[-1], y[0], y[-1])
    else:
        extension = (x[0] / mm, x[-1] / mm, y[0] / mm, y[-1] / mm)
        xlabel = "x (mm)"
        ylabel = "y (mm)"

    IDimage = plt.imshow(
        image,
        interpolation=interpolation,
        aspect="auto",
        origin="lower",
        extent=extension,
    )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.suptitle(title)
    plt.axis(extension)
    if scale not in ("", None, []):
        plt.axis(scale)
    IDimage.set_cmap(color)
    plt.tight_layout()
    return id_fig, IDax, IDimage


def draw_several_fields(
        fields: list,
        titles: tuple[str] = "",
        title: str = "",
        figsize: tuple[float, float] | None = None,
        kinds: tuple[str] = "",
        logarithm: tuple[float] | float = False,
        normalize: bool = False):
    """Draws several fields in subplots

    Args:
        fields (tuple): list with several scalar_fields_XY
        titles (tuple): list with titles
        title (str): suptitle
        kinds (tuple): list with kinds of figures (amplitude', 'intensity', 'phase', 'real_field', 'contour')
        logarithm (float): If >0, intensity is scaled in logarithm
        normalize (bool): If True, max(intensity)=1
    """

    order = [[1, 1], [2, 1], [3, 1], [2, 2], [3, 2], [3, 2]]
    length = [(10, 8), (10, 5), (11, 5), (9, 7), (12, 9), (12, 9)]

    num_dibujos = len(fields)
    fil = order[num_dibujos - 1][0]
    col = order[num_dibujos - 1][1]

    if figsize == None:
        figsize = length[num_dibujos - 1]

    id_fig = plt.figure(figsize=figsize, facecolor="w", edgecolor="k")
    num_dibujos = len(fields)

    percentage_intensity = CONF_DRAWING["percentage_intensity"]

    if type(logarithm) in (int, float, bool):
        logarithm = logarithm * np.ones_like(fields)

    for i in sorted(range(num_dibujos)):
        c = fields[i]
        id_fig.add_subplot(col, fil, i + 1)
        extension = (c.x.min(), c.x.max(), c.y.min(), c.y.max())
        amplitude, intensity, phase = field_parameters(c.u, has_amplitude_sign=True)

        if kinds == "":
            image = intensity
            colormap = CONF_DRAWING["color_intensity"]
            kind = "intensity"
        else:
            kind = kinds[i]
        if kind == "intensity":
            image = intensity
            colormap = CONF_DRAWING["color_intensity"]
        elif kind == "phase":
            phase = phase/degrees
            phase[intensity < percentage_intensity * (intensity.max())] = 0
            colormap = CONF_DRAWING["color_phase"]
            image = phase
        elif kind == "amplitude":
            image = amplitude
            colormap = CONF_DRAWING["color_amplitude"]
        elif kind == "real":
            percentage_intensity = CONF_DRAWING["percentage_intensity"]
            rf = np.real(c.u)
            intensity = np.abs(c.u) ** 2
            rf[intensity < percentage_intensity * (intensity.max())] = 0

            image = np.real(c.u)
            colormap = CONF_DRAWING["color_real"]

        if logarithm[i] != 0 and kind in ("intensity", "amplitude", "real"):
            image = np.log(logarithm[i] * image + 1)

        if normalize == "maximum" and kind in ("intensity", "amplitude", "real"):
            image = image / image.max()

        plt.imshow(
            image,
            interpolation="bilinear",
            aspect="auto",
            origin="lower",
            extent=extension,
        )

        plt.set_cmap(colormap)
        if titles != "":
            plt.title(titles[i], fontsize=24)

        plt.suptitle(title, fontsize=26)
        plt.axis("scaled")
        plt.axis(extension)
        plt.colorbar(orientation="horizontal", fraction=0.046)
        if kind == "phase":
            plt.clim(-180, 180)


def change_image_size(
        image_name: str,
        length: str = "800x600",
        final_filename: str = "prueba.png",
        dpi: int = 100):
    """change the size with imageMagick

    Args:
        image_name (str): name of file
        length (str): size of image
        final_filename (str): final filename
        dpi (int): dpi

    Examples:

    convert image_name -resize '1000' -units 300 final_filename.png
        - anchura 1000 - mantiene forma
    convert image_name -resize 'x200' final_filename.png
        - height  200  - mantiene forma
    convert image_name -resize '100x200>' final_filename.png
        - mantiene forma, lo que sea mayor
    convert image_name -resize '100x200<' final_filename.png
        - mantiene forma, lo que sea menor
    convert image_name -resize '@1000000' final_filename.png
        - mantiene la forma, con 1Mpixel
    convert image_name -resize '100x200!' final_filename.png
        - obliga a tener el tamaño, no mantiene escala
    """
    texto = "convert {} -resize {} {}".format(image_name, length, final_filename)
    print(texto)
    os.system(texto)


def extract_image_from_video(nombre_video: str | None = None, num_frame: str = "[0, ]",
                             final_filename: str = "prueba.png"):
    """Extract images form a video using imageMagick.

    convert 'animacion.avi[15,]' animacion_frame.png. Extracts frame 15 (ony 15)
    convert 'animacion.avi[15]' animacion_frame.png. Extracts the first 15
    convert 'animacion.avi[5,10]' animacion_frame.png. Extracts frame 5 and 10
    """

    texto = "convert '%s%s' %s" % (nombre_video, num_frame, final_filename)
    print(texto)
    os.system(texto)


def normalize_draw(u, logarithm: float | bool = False, normalize: bool = False, cut_value: float | None = None):
    """
    Gets a field and changes its caracteristics for drawing

    Args:
        u (field): field
        logarithm (float): logarithm to image: np.log(logarithm*u + 1)
        normalize (str or bool): False, 'mean', 'intensity'
    """

    u = np.real(u)

    if logarithm > 0:
        u_sign = u
        u = np.log(logarithm * np.abs(u) + 1)
        if np.any(u_sign < 0):
            np.putmask(u, u_sign < 0, -1 * u)

    if normalize is False:
        pass
    elif normalize == "maximum":
        u = u / (np.abs(u).max() + eps)
    elif normalize == "mean":
        u = u / u.mean()

    if cut_value not in ([], "", 0, None):
        u[u > cut_value] = cut_value

    return u


def prepare_drawing(u, kind: str = "intensity", logarithm: float | bool = False,
                    normalize: bool = False):
    """It is necessary that figure is previously defined: plt.figure()

    Args:
        u - field
        kind - 'intensity', 'amplitude', 'phase'
        logarithm - True or False
        normalize: False, 'maximum', 'intensity', 'area'

    Returns:
        Returns (numpy.array): I_drawing for direct plotting
    """
    amplitude, intensity, phase = field_parameters(u)

    if kind == "intensity":
        I_drawing = intensity
        I_drawing = normalize_draw(I_drawing, logarithm, normalize)
        # plt.title('Intensity')
    elif kind == "amplitude":
        I_drawing = amplitude
        I_drawing = normalize_draw(I_drawing, logarithm, normalize)
        # plt.title('Amplitude')
    elif kind == "phase":
        I_drawing = phase
        # plt.title('phase')
    else:
        print("bad kind parameter")
        return None
    return I_drawing


def prepare_video(fps: int = 15, title: str = "", artist: str = "", comment: str = ""):
    """_summary_

    Args:
        fps (int, optional): FPS. Defaults to 15.
        title (str, optional): Titles. Defaults to "".
        artist (str, optional): ?. Defaults to "".
        comment (str, optional): comment. Defaults to "".

    Returns:
        _type_: _description_
    """
    FFMpegWriter = manimation.writers["ffmpeg"]  # ffmpeg mencoder
    metadata = dict(title=title, artist=artist, comment=comment)
    writer = FFMpegWriter(fps=fps, metadata=metadata)
    return writer


def make_video_from_file(self, files: list, filename: str = ""):
    """make a video from file

    Args:
        files (tuple): files to add
        filename (bool, optional): final filename. Defaults to "".
    """
    print("Start", files)
    if not (filename) == "":
        print("Making movie animation.mpg - this make take a while")
        texto = (
            "mencoder 'mf://_tmp*.png' -mf kind=png:fps=10 -ovc lavc -lavcopts vcodec=wmv2 -oac copy -o " +
            filename
        )
        # texto = "mencoder 'mf://home/_tmp*.png' -mf kind=png:fps=10 -ovc lavc -lavcopts vcodec=wmv2 -oac copy -o " + filename
        os.system(texto)
        # os.system("convert _tmp*.png animation2.gif")  # este sale muy grande
        # esto podria hacer mas pequeno convert -geometry 400 -loop 5  animation2.gif animation3.gif
        # cleanup
        print(files)
        for fname in files:
            os.remove(fname)
    print("exit", files)


def reduce_matrix_size(reduce_matrix: str | tuple[int], x: NDArrayFloat,
                       y: NDArrayFloat, image: NDArrayFloat,
                       verbose: bool = False):
    """Reduces the size of matrix for drawing purposes. If the matrix is very big, the drawing process is slow.

    Args:
        reduce_matrix (str or (int, int)): if str: 'standard', if (int, int) reduction_factor.
        x (np.array): array with x.
        y (np.array): array with y or z
        image (np.array): image to reduce the size.
        verbose (bool): if True, prints info

    Returns:
        (np.array): reduced image
    """
    image_ini = image.shape
    if reduce_matrix in (None, "", []):
        pass
    elif reduce_matrix == "standard":
        num_x = len(x)
        num_y = len(y)
        reduction_x = int(num_x / 500)
        reduction_y = int(num_y / 500)

        if reduction_x > 2 and reduction_y > 2:
            image = image[::reduction_x, ::reduction_y]
        else:
            pass
    else:
        image = image[:: reduce_matrix[0], :: reduce_matrix[1]]

    if verbose:
        print(
            (
                "reduce_matrix_size: size ini = {}, size_final = {}".format(
                    image_ini, image.shape
                )
            )
        )
    return image


def draw_edges(vector_field, plt, draw_borders: bool=True, color: str='w.', ms: float=0.1) -> None:
    """
    draw_edges _summary_

    _extended_summary_

    Args:
        vector_field (Vector_field_XY | Vector_field_XZ): _description_
        plt (_type_): _description_
        scale (str, optional): _description_. Defaults to 'scaled'.
        draw_borders (bool, optional): _description_. Defaults to True.
        color (str, optional): _description_. Defaults to 'w.'.
        ms (float, optional): _description_. Defaults to 0.05.
    """
    min_incr: float = 0.0005
    
    if draw_borders is True:
        vector_field.borders = np.array(vector_field.borders)
        vector_field.surface_detection(1, min_incr)
        border0 = vector_field.borders[0]
        border1 = vector_field.borders[1]

        plt.plot(border1, border0, color, ms=ms)

        # idea to draw the refractive index with a 2d colormap
        # h1 = plt.imshow(np.abs(vector_field.n.transpose()),
        #         interpolation='bilinear',
        #         aspect='auto',
        #         origin='lower', alpha=0.25, cmap=CONF_DRAWING['color_n'])