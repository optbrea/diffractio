# !/usr/bin/env python3

# ----------------------------------------------------------------------
# Name:        utils_optics.py
# Purpose:     Utility functions for optics operations
#
# Author:      Luis Miguel Sanchez Brea
#
# Created:     2024
# Licence:     GPLv3
# ----------------------------------------------------------------------

# flake8: noqa

""" General purpose optics functions """

import pandas as pd
from scipy.signal import argrelextrema
from scipy.interpolate import CubicSpline, PchipInterpolator, Akima1DInterpolator

from .utils_typing import npt, Any, NDArray,  NDArrayFloat, NDArrayComplex


from .__init__ import degrees, np, plt, um
from .utils_math import fft_convolution1d, fft_convolution2d, find_extrema, nearest


def roughness_1D(x: NDArrayFloat, t: float, s: float, kind: str = "normal"):
    """Rough surface, 1D.

    Args:
        x (numpy.array): array with x positions
        t (float): correlation lens
        s (float): std of roughness
        kind (str): 'normal', 'uniform'

    Returns:
        (numpy.array) Topography of roughnness in microns.

    References:
        JA Oglivy "Theory of wave scattering from random surfaces" Adam Hilger p.224.

    """

    width = x[-1] - x[0]
    dx = x[1] - x[0]

    # Surface parameters

    L_width = width / (2 * dx)
    M = round(4 * t / (np.sqrt(2) * dx))

    N_width = int(np.floor(L_width + M))

    desp_width = np.arange(-M, M + 1)

    desp_width = desp_width * dx
    weigths = np.exp(-2 * (desp_width**2 / t**2))

    weigths = np.abs(weigths / np.sqrt((weigths**2).sum()))

    if kind == "normal":
        h_no_corr = s * np.random.randn(2 * N_width + 2)
        h_corr = fft_convolution1d(h_no_corr, weigths)
        h_corr = h_corr[0: len(x)]
    elif kind == "uniform":
        h_corr = s * (np.random.rand(len(x)) - 0.5)

    return h_corr


def roughness_2D(x: NDArrayFloat, y: tuple[float, float], t: float, s: float):
    """Rough surface, 2D

    Args:
        x (numpy.array): x positions
        y (numpy.array): y positions
        t (float, float): (tx, ty), correlation length of roughness
        s (float): std of heights

    Returns:
        (numpy.array) Topography of roughnness in microns.

    Example:
        roughness(t=(50*um, 25*um), s=1*um)

    References:
        JA Oglivy "Theory of wave scattering from random surfaces" Adam Hilger p.224.
    """

    if isinstance(t, (float, int, complex)):
        t = (t, t)

    tx, ty = t

    width = x[-1] - x[0]
    length = y[-1] - y[0]
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    L_width = width / (2 * dx)
    L_length = length / (2 * dy)
    Mx = round(4 * tx / (np.sqrt(2) * dx))
    My = round(4 * ty / (np.sqrt(2) * dy))

    N_width = int(np.floor(L_width + Mx))
    N_length = int(np.floor(L_length + My))

    desp_width, desp_length = np.meshgrid(np.arange(-Mx, Mx + 1), np.arange(-My, My + 1))
    desp_width = desp_width * dx
    desp_length = desp_length * dy

    weigths = np.exp(-2 * (desp_width**2 / tx**2 + desp_length**2 / ty**2))
    weigths = np.abs(weigths / np.sqrt((weigths**2).sum()))

    h_no_corr = s * np.random.randn(2 * N_width + 2, 2 * N_length + 2)
    h_corr = fft_convolution2d(h_no_corr, weigths)
    h_corr = h_corr[0: len(x), 0: len(y)]

    return h_corr


def beam_width_1D(u: NDArrayComplex, x: NDArrayFloat, remove_background: bool = False):
    """One dimensional beam width, according to D4σ or second moment width.

    Args:
        u (np.array): field (not intensity).
        x (np.array): x

    Returns:
        (float): width
        (float): x_mean

    References:
        https://en.wikipedia.org/wiki/Beam_diameter
    """

    intensity = np.abs(u)**2

    if remove_background is True:
        intensity = intensity - intensity - min()

    P = (intensity).sum()
    x_mean = (intensity * x).sum() / P
    x2_mean = (intensity * (x - x_mean) ** 2).sum() / P
    width_x = 4 * np.sqrt(x2_mean)
    return width_x, x_mean


def width_percentage(x: NDArrayFloat, y: NDArrayFloat, percentage: float = 0.5, verbose: bool = False):
    """beam width (2*sigma) given at a certain height from maximum

    Args:
        x (np.array): x
        y (np.array): y
        percentage (float): percentage of height. For example: 0.5

    Returns:
        (float): width, width of at given %
        (tuple): x_list: (x[i_left], x[i_max], x[i_right])
        (tuple): x_list: (i_left, i_max, i_right)

    Notes:
        y=np.exp(-x**2/(s**2))  percentage=1/e -> width = 2*s
        y=np.exp(-x**2/(s**2))  percentage=1/e**4 -> width = 4*s
        y=np.exp(-x**2/(2*s**2))  percentage=1/e**2 =  -> width = 4*s

    """

    maximum = y.max()
    level = percentage * maximum
    i_max = np.argmax(y)

    if i_max == 0:
        i_left = 0
        print("beam width out of range")
    else:
        i_left, _, _ = nearest(y[0:i_max], level)

    if i_max == len(y) - 1:
        i_right = len(y) - 1
        print("beam width out of range")
    else:
        i_right, _, _ = nearest(y[i_max:-1], level)
        i_right = i_right + i_max

    if verbose is True:
        print(i_left, i_max, i_right)

    width = x[i_right] - x[i_left]

    x_list = (x[i_left], x[i_max], x[i_right])
    i_list = (i_left, i_max, i_right)

    return width, x_list, i_list


def beam_width_2D(x: NDArrayFloat, y: NDArrayFloat, intensity: NDArrayFloat,
                  remove_background: bool = False, has_draw: bool = False):
    """2D beam width, ISO11146 width

    Args:
        x (np.array): 1d x
        y (np.array): 1d y
        intensity (np.array):  intensity

    Returns:
        (float): dx width x
        (float): dy width y
        (float): principal_axis, angle
        (str): (x_mean, y_mean, x2_mean, y2_mean, xy_mean), Moments

    References:

        * https://en.wikipedia.org/wiki/Beam_diameter

        * http://www.auniontech.com/ueditor/file/20170921/1505982360689799.pdf


    """
    X, Y = np.meshgrid(x, y)
    if remove_background is True:
        intensity = intensity - intensity - min()

    P = intensity.sum()
    x_mean = (intensity * X).sum() / P
    y_mean = (intensity * Y).sum() / P
    x2_mean = (intensity * (X - x_mean) ** 2).sum() / P
    y2_mean = (intensity * (Y - y_mean) ** 2).sum() / P
    xy_mean = (intensity * (X - x_mean) * (Y - y_mean)).sum() / P
    # gamma = (x2_mean - y2_mean) / np.abs(x2_mean - y2_mean + 1e-16)
    gamma = np.sign(x2_mean - y2_mean + 0.0000000001)
    rt = np.sqrt((x2_mean - y2_mean) ** 2 + 4 * xy_mean**2)
    dx = 2 * np.sqrt(2) * np.sqrt(x2_mean + y2_mean + gamma * rt)
    dy = 2 * np.sqrt(2) * np.sqrt(x2_mean + y2_mean - gamma * rt)

    # print(gamma)
    # print(rt)
    # print(x2_mean, y2_mean, rt, dx, dy)

    principal_axis = 0.5 * np.arctan2(2 * xy_mean, x2_mean - y2_mean)

    if has_draw is True:
        from matplotlib.patches import Ellipse

        from .scalar_fields_XY import Scalar_field_XY

        u0 = Scalar_field_XY(x, y, 1)
        u0.u = np.sqrt(intensity)
        u0.draw()
        ellipse = Ellipse(
            xy=(x_mean, y_mean), width=dy, height=dx, angle=-principal_axis/degrees
        )

        ax = plt.gca()
        ax.add_artist(ellipse)
        ellipse.set_clip_box(ax.bbox)
        ellipse.set_alpha(0.75)
        ellipse.set_facecolor("none")
        ellipse.set_edgecolor([1, 1, 1])
        ellipse.set_linewidth(3)

    return dx, dy, principal_axis, (x_mean, y_mean, x2_mean, y2_mean, xy_mean)


def refractive_index(filename: str, wavelength: float, raw: bool = False,
                     has_draw: bool = bool):
    """gets refractive index from https://refractiveindex.info .

    * Files has to be converted to xlsx format.
    * n and k checks has to be activated.

    Args:
        filename (str): xlsx file
        wavelength (float): wavelength in microns, example, 0.6328.
        raw (bool): if True returns all the data in file.
        has_draw (bool): draw the data from the file

    Returns:
        if raw is False (float, float): n, k  from each wavelength
        if raw is True  (np.array, np.array): n,k for wavelengths in file


    """
    data = pd.read_excel(filename)

    wavelengths = data["Wavelength, µm"].values.astype(float)
    n = data["n"].values.astype(float)
    kappa = data["k"].values.astype(float)

    if has_draw is True:
        fig, ax1 = plt.subplots()
        ax1.set_xlabel("wavelengths (nm)")
        ax1.plot(wavelengths, n, "r", label="n")
        ax1.set_ylabel("n", color="r")
        ax1.tick_params(axis="y", labelcolor="r")
        ax2 = ax1.twinx()
        ax2.plot(wavelengths, kappa, "b", label=r"$\kappa$")
        ax2.set_ylabel(r"$\kappa$", color="b")
        ax2.tick_params(axis="y", labelcolor="b")
        fig.tight_layout()
        fig.legend()

    if raw is True:
        return wavelengths, n, kappa

    else:
        z_n = np.polyfit(wavelengths, n, 6)
        z_kappa = np.polyfit(wavelengths, kappa, 6)

        f_n = np.poly1d(z_n)
        f_kappa = np.poly1d(z_kappa)

        return f_n(wavelength), f_kappa(wavelength)


def FWHM1D(x: NDArrayFloat, intensity: NDArrayFloat, percentage: float = 0.5,
           remove_background: str | None = None, has_draw: bool = False):
    """ FWHM

    remove_background = 

    Args:
        x (NDArrayFloat): x array
        intensity (NDArrayFloat): intensity array
        percentage (float, optional): heigth of peak to measure. Defaults to 0.5.
        remove_background (str | None, optional): 'min', 'mean', None. Defaults to None.
        has_draw (bool, optional): It draws. Defaults to False.

    Returns:
        float: value of FWHM
    """

    if remove_background == "mean":
        I_background = intensity.mean()
    elif remove_background == "min":
        I_background = intensity.min()
    else:
        I_background = 0

    intensity = intensity - I_background

    if type(remove_background) is float:
        intensity[intensity < remove_background * intensity.max()] = 0

    delta_x = x[1] - x[0]
    amp_max = intensity.max()
    amp_med = amp_max * percentage
    i_max = np.where(intensity == amp_max)
    i_max = int(i_max[0][0])
    left = intensity[0:i_max]
    right = intensity[i_max::]

    i_left, _, distance_left = nearest(left, percentage * amp_max)
    slope_left = (intensity[i_left + 1] - intensity[i_left]) / delta_x

    i_right, _, distance_right = nearest(right, percentage * amp_max)
    slope_right = (
        intensity[i_max + i_right] - intensity[i_max + i_right - 1]
    ) / delta_x

    i_right = i_right + i_max

    x_right = i_right * delta_x - distance_right / slope_right
    x_left = i_left * delta_x - distance_left / slope_left

    FWHM_x = x_right - x_left

    amp_max = amp_max + I_background
    amp_med = amp_med + I_background

    intensity = intensity + I_background

    if has_draw is True:
        if remove_background is True:
            intensity = intensity + intensity.min()
        plt.figure()

        plt.plot(x, intensity, "k", lw=2)
        plt.plot([x[0], x[-1]], [amp_max, amp_max], "r--")
        plt.plot([x[0], x[-1]], [amp_med, amp_med], "r--")

        plt.plot(x[i_max], intensity[i_max], "ro", ms=8)
        plt.plot(x[int(i_right)], intensity[int(i_left)], "ro", ms=8)
        plt.plot(x[int(i_left)], intensity[int(i_right)], "ro", ms=8)
        plt.ylim(ymin=0)
        plt.xlim(x[0], x[-1])

    return np.squeeze(FWHM_x)


def FWHM2D(x: NDArrayFloat, y: NDArrayFloat, intensity: NDArrayFloat, percentage: float = 0.5,
           remove_background: bool = False, has_draw: bool = False, xlim: tuple[float] | None = None):
    """ Get FWHM2D  in x and i direction


    Args:
        x (NDArrayFloat): x array
        y (NDArrayFloat): y array
        intensity (NDArrayFloat): intensity
        percentage (float, optional): heigth of peak to measure. Defaults to 0.5.
        remove_background (bool, optional): 'min', 'mean', None. Defaults to False.
        has_draw (bool, optional): if True it draws. Defaults to False.
        xlim (tuple[float] | None, optional): xlim in drawing. Defaults to None.

    Returns:
        FWHM_x (float): width in x direction
        FWHM_y (float): width in y direction
    TODO: perform profiles at several angles and fit to a ellipse.
    """

    i_pos, _, I_max = find_extrema(intensity.transpose(), x, y, kind="max")

    Ix = intensity[:, i_pos[0, 1]]
    Iy = intensity[i_pos[0, 0], :]

    # print(x.shape, Iy.shape)
    FWHM_x = FWHM1D(x, Ix, percentage, remove_background, has_draw=has_draw)
    if has_draw is True:
        if xlim is not None:
            plt.xlim(xlim)

    # print(y.shape, Iy.shape)
    FWHM_y = FWHM1D(y, Iy, percentage, remove_background, has_draw=has_draw)
    if has_draw is True:
        if xlim is not None:
            plt.xlim(xlim)

    return FWHM_x, FWHM_y


def DOF(z: NDArrayFloat, widths: NDArrayFloat, w_factor: float = np.sqrt(2), w_fixed: float = 0,
        has_draw: bool = False, verbose: bool = False):
    """Determines Depth-of_focus (DOF) in terms of the width at different distances

    Args:

        z (np.array): z positions
        widths (np.array): width at positions z
        w_factor (float): range to determine z where   w = w_factor * w0, being w0 the beam waist
        w_fixed (float): If it is not 0, then it is used as w_min
        has_draw (bool): if True draws the depth of focus
        verbose (bool): if True, prints data

    References:

        B. E. A. Saleh and M. C. Teich, Fundamentals of photonics. john Wiley & sons, 2nd ed. 2007. Eqs (3.1-18) (3.1-22) page 79

    Returns:

        (float): Depth of focus
        (float): beam waist
        (float, float, float): postions (z_min, z_0, z_max) of the depth of focus
    """

    if w_fixed == 0:
        beam_waist = widths.min()
        i_w0 = np.where(widths == beam_waist)
        i_w0 = int(i_w0[0][0])

    else:
        beam_waist = w_fixed
        i_w0, _, _ = nearest(widths, beam_waist)

    left = widths[0:i_w0]
    right = widths[i_w0::]

    i_left, _, _ = nearest(left, w_factor * beam_waist)

    i_right, _, _ = nearest(right, w_factor * beam_waist)

    z_rayleigh = z[i_right + i_w0] - z[i_left]

    if verbose:
        print(i_w0, widths[i_w0])
        print(z_rayleigh)

        print(widths[i_right + i_w0], z[i_right + i_w0])
        print(widths[i_left], z[i_left])

    if has_draw:
        plt.figure()

        plt.plot(z, widths, "k", lw=2)
        plt.plot(z, -widths, "k", lw=2)
        plt.plot(z, np.zeros_like(z), "k-.", lw=2)

        plt.plot([z[i_left], z[i_left]], [-widths[i_left], widths[i_left]], "r--")
        plt.plot(
            [z[i_right + i_w0], z[i_right + i_w0]],
            [-widths[i_right + i_w0], widths[i_right + i_w0]],
            "r--",
        )
        plt.annotate(
            text="",
            xy=(z[i_left], -widths[i_right + i_w0]),
            xytext=(z[i_right + i_w0], -widths[i_right + i_w0]),
            arrowprops=dict(arrowstyle="<->"),
        )
        plt.text(z[i_w0], -widths.mean(), r"$z_{R}$", fontsize=18)
        plt.xlim(z[0], z[-1])
        plt.ylim(-widths.max(), widths.max())

    return z_rayleigh, beam_waist, np.array([z[i_left], z[i_w0], z[i_right + i_w0]])


def detect_intensity_range(x: NDArrayFloat, intensity: NDArrayFloat, percentage: float = 0.95,
                           has_draw: bool = True, logarithm=True):
    """Determines positions x_min, x_max where intensity of the beam is percentage

    Args:
        x (np.array): x positions
        intensity (np.array): Intensity of the 1D beam
        percentage (float): value 0-1 representing the percentage of intensity between area
        has_draw (bool): if True draws the field an the range
        logarithm (float): when has_draw, draws logarithm or normal intensity

    Returns:
        (float, float): positions (x_min, right) where intensity beam is enclosed at %.
    """

    I_cum = intensity.cumsum()

    pc = percentage + (1 - percentage)/2
    Icum_min = (1 - pc) * I_cum.max()
    Icum_max = I_cum.max() * pc

    I_min = intensity.min()
    I_max = intensity.max()

    i_min, _, _ = nearest(I_cum, Icum_min)
    i_max, _, _ = nearest(I_cum, Icum_max)

    x_min = x[i_min]
    x_max = x[i_max]

    if has_draw is True:
        _, ax = plt.subplots()

        if logarithm is True:
            I2 = np.log(intensity + 1)
            I_min2 = np.log(I_min + 1)
            I_max2 = np.log(I_max + 1)

            I2 = I2 / I2.max()
            I_max2 = I_max2 / I_max2.max()
        else:
            I2 = intensity
            I_min2 = I_min
            I_max2 = I_max

        ax.plot(x, I2, c="r", alpha=1, lw=4)

        x_bordes = [x_min, x_max, x_max, x_min, x_min]
        y_bordes = [I_min2, I_min2, I_max2, I_max2, I_min2]

        ax.fill(x_bordes, y_bordes, c="r", alpha=0.25)

    return x_min, x_max


def MTF_ideal(frequencies: NDArrayFloat, wavelength: float, diameter: float, focal: float,
              kind: str, verbose: bool = False, has_draw: bool = False):
    """Determines the ideal MTF of a lens.

    References:
        https://www.edmundoptics.com/resources/application-notes/optics/introduction-to-modulation-transfer-function/

        https://www.optikos.com/wp-content/uploads/2015/10/How-to-Measure-MTF-and-other-Properties-of-Lenses.pdf

    Args:
        frequencies (numpy.array): array with frequencies in *lines/mm*
        wavelength (float): wavelength of incoming light beam
        diameter (float): diameter of lens
        focal (float): focal distance of lens
        kind (float): '1D', '2D'
        verbose (bool): if True displays limit frequency of the lens

    Returns:
        (numpy.array) MTF: Normalized MTF of ideal lens
        (float) frequency_max: maximum frequency of the lens
    """

    F_number = focal / diameter
    frequency_max = 1000.0 / (wavelength * F_number)  # porque mido en micras
    fx_norm = np.abs(frequencies / frequency_max)

    if kind == "1D":
        MTF = 1 - np.abs(fx_norm)
        MTF[fx_norm > 1] = 0

    elif kind == "2D":
        fx2 = np.arccos(fx_norm)
        MTF = np.real(2 / np.pi * (fx2 - np.cos(fx2) * np.sin(fx2)))

        # Another definition: https://www.optikos.com/wp-content/uploads/2015/10/How-to-Measure-MTF-and-other-Properties-of-Lenses.pdf
        # MTF = np.real(2/np.pi*(np.arccos(fx_norm)-fx_norm*np.sqrt(1-fx_norm**2)))

        # isH1 = MTF > 1
        # MTF[isH1] = 2 - MTF[isH1]

    if verbose is True:
        print("frquency = {:4.2f} lines/mm".format(frequency_max))

    if has_draw is True:
        plt.figure()
        plt.plot(frequencies, MTF, "k")
        plt.xlabel(r"$f_x (mm^{-1})$", fontsize=18)
        plt.ylabel("MTF", fontsize=18)

    return MTF, frequency_max


def lines_mm_2_cycles_degree(lines_mm: NDArrayFloat, focal: float):
    """ Converts lines/mm to cycles/degree. JA Gomez-Pedrero 
    Args:
        lines_mm (numpy.array or float): lines_per_mm
        focal (float): focal of lens
    """

    frec_cycles_deg = 180 * focal * lines_mm / np.pi

    return frec_cycles_deg


def MTF_parameters(MTF: NDArrayFloat, MTF_ideal: NDArrayFloat, lines_mm: float = 50, verbose: bool = False):
    """MTF Args: strehl_ratio, mtf_50_ratio, freq_50_real, freq_50_ideal

    References:
        https://www.edmundoptics.com/resources/application-notes/optics/introduction-to-modulation-transfer-function/strehl_ratio

    frequencies of mtf are given since both MTF can have different steps

    Args:
        MTF (N,2 numpy.array): (freq, MTF) of system in lines/mm
        MTF_ideal (M,2 numpy.array): (freq, MTF) of ideal system in lines/mm
        lines_mm (float): (0-1) Height of MTF for ratios

    Returns:
        (float): strehl_ratio
        (float): MTF_ratio at freq_obs height
        (float): frequency at freq_obs of MTF
        (float): frequency at freq_obs of MTF_ideal
    """

    fx_real, mtf_real = MTF
    fx_ideal, mtf_ideal = MTF_ideal

    i_0_real, _, _ = nearest(fx_real, 0)
    i_0_ideal, _, _ = nearest(fx_ideal, 0)

    dx_real = fx_real[1] - fx_real[0]
    dx_ideal = fx_ideal[1] - fx_ideal[0]

    mtf_real = mtf_real[i_0_real::]
    mtf_ideal = mtf_ideal[i_0_ideal::]

    fx_real = fx_real[i_0_real::]
    fx_ideal = fx_ideal[i_0_ideal::]

    # STREHL RATIO
    strehl_ratio = (mtf_real.sum() * dx_real) / (mtf_ideal.sum() * dx_ideal)

    # MTF at 50 (u other) lines/mm
    imenor_ideal, _, _ = nearest(fx_ideal, lines_mm)
    imenor_real, _, _ = nearest(fx_real, lines_mm)

    freq_50_ideal = np.abs(mtf_ideal[imenor_ideal])
    freq_50_real = np.abs(mtf_real[imenor_real])

    mtf_50_ratio = freq_50_real / freq_50_ideal

    if verbose is True:
        print(" MTF Args:")
        print("- Strehl_ratio      = {:2.2f}".format(strehl_ratio))
        print("- MTF_ratio @ {:2.2f}  = {:2.2f}".format(lines_mm, mtf_50_ratio))
        print(
            "- freq @ {:2.2f}  real (lines/mm) = {:2.2f}".format(lines_mm, freq_50_real)
        )
        print(
            "- freq @ {:2.2f}  ideal (lines/mm) = {:2.2f}".format(
                lines_mm, freq_50_ideal
            )
        )

    return strehl_ratio, mtf_50_ratio, freq_50_real, freq_50_ideal


def gauss_spectrum(wavelengths: NDArrayFloat, w_central: float, Dw: float, normalize: bool = True):
    """ 
    Returns weigths for a gaussian spectrum

    Args:
        wavelengths (NDArrayFloat): array with wavelengths
        w_central (float): central wavelength
        Dw (float): width of the spectrum
        normalize (bool, optional): if True sum of weights is 1. Defaults to True.

    Returns:
        weights (float): gaussian spectrum
    """

    weigths = np.exp(-((wavelengths - w_central) ** 2) / (2 * Dw**2))

    if normalize is True:
        weights = weigths / weigths.sum()

    return weights


def lorentz_spectrum(wavelengths: NDArrayFloat, w_central: float, Dw: float, normalize: bool = True):
    """ 
    Returns weigths for a Lorentz spectrum

    Args:
        wavelengths (NDArrayFloat): array with wavelengths
        w_central (float): central wavelength
        Dw (float): width of the spectrum
        normalize (bool, optional): if True sum of weights is 1. Defaults to True.

    Returns:
        weights (float): Lorentz spectrum
    """

    weigths = 1 / (1 + ((wavelengths - w_central) / (Dw/2)) ** 2)

    if normalize is True:
        weights = weigths / weigths.sum()

    return weights


def uniform_spectrum(wavelengths: NDArrayFloat, normalize: bool = True):
    """returns weigths for a gaussian spectrum

    Args:
        wavelengths: array with wavelengths
        w_central: central wavelength
        Dw: width of the spectrum
        normalize: if True sum of weights is 1
    """

    weigths = np.ones_like(wavelengths, dtype=float)

    if normalize is True:
        weights = weigths / weigths.sum()

    return weights


def normalize_field(self, kind='amplitude', new_field: bool = False):
    """Normalize the field to maximum intensity.

    Args:
        kind (str): 'amplitude', or 'intensity'
        new_field (bool): If True returns a field, else returns a matrix

    Returns:
        (np.array): normalized field.
    """

    if self.type[0:6] == "Scalar":

        if kind == 'amplitude':
                maximum = np.sqrt(np.abs(self.u) ** 2).max()
        elif kind == 'intensity':
                maximum = (np.abs(self.u) ** 2).max()


        if new_field is False:
            self.u = self.u / maximum
        else:
            field_new = self.duplicate()
            field_new.u = self.u / maximum
            return field_new

    elif self.type[0:6] == "Vector":

        if kind == 'amplitude':
                maximum  = np.sqrt(np.abs(self.Ex) ** 2 + np.abs(self.Ey) ** 2 + np.abs(self.Ez) ** 2).max()
        elif kind == 'intensity':
                maximum  = (np.abs(self.Ex) ** 2 + np.abs(self.Ey) ** 2 + np.abs(self.Ez) ** 2).max()


        if new_field is False:
            self.Ex = self.Ex / maximum
            self.Ey = self.Ey / maximum
            self.Ez = self.Ez / maximum
        else:
            field_new = self.duplicate()
            field_new.Ex = self.Ex / maximum
            field_new.Ey = self.Ey / maximum
            field_new.Ez = self.Ez / maximum
            return field_new


def field_parameters(u: NDArrayComplex, has_amplitude_sign: bool = False):
    """Determines main parameters of field: amplitude intensity phase. All this parameters have the same dimension as u.

    Args:
        u (numpy.array): optical field (comes usually form field.u)
        has_amplitude_sign (bool): If True - amplitude = np.sign(u) * np.abs(u), Else: amplitude =  np.abs(u)

    Returns:
        amplitude (numpy.array): np.abs(u)
        intensity (numpy.array): np.abs(u)**2
        phase (numpy.array): np.angle(u)

    """

    intensity = np.abs(u) ** 2
    phase = np.angle(u)

    if has_amplitude_sign is True:
        amplitude = np.sign(np.real(u)) * np.abs(u)
    else:
        amplitude = np.abs(u)

    # amplitude = np.abs(u)
    # amplitude = u * np.exp(-1j * phase)
    amplitude = np.real(amplitude)

    return amplitude, intensity, phase


def convert_phase2heigths(phase: NDArrayFloat, wavelength: float, n: float, n_background: float):
    """Phase is converted to a depth. It is useful to convert Scalar_mask_X to Scalar_mask_XZ

    phase(x,z)= k (n-n_0) h(x,z).

    Args:
        phase (np.array): Phases
        wavelength (float): wavelength
        n (float or complex): refractive index of material
        n_background (float): refractive index of background

    Returns:
        (np.array): depths related to phases
    """
    k = 2 * np.pi / wavelength
    n = np.real(n)

    return phase / (k * (n - n_background))


def convert_amplitude2heigths(amplitude: NDArrayComplex, wavelength: float,
                              kappa: float, n_background: float):
    """Amplitude and it is converted to a depth. It is useful to convert Scalar_mask_X to Scalar_mask_XZ.

    Args:
        phase (np.array): Phases
        wavelength (float): wavelength
        kappa (float): refractive index of material.
        n_background (float): refractive index of background

    Returns:
        (np.array): depths related to amplitudes
    """

    eps_depth = 1e-4

    amplitude[amplitude < eps_depth] = eps_depth

    depth = np.log(amplitude) * wavelength / (-2 * np.pi * kappa)

    return depth


def fresnel_equations_kx(kx: NDArrayComplex, wavelength: float, n1: float, n2: float,
                         outputs: tuple[bool, bool, bool, bool] = [True, True, True, True], 
                         has_draw: bool = True,
                         kind: str = "amplitude_phase"):
    """Fresnel_equations where input are kx part of wavevector.

    Args:
        kx (np.array): kx
        wavelength (float): wavelength
        n1 (float): refractive index of first materia
        n2 (float): refractive index of second materia
        outputs (bool,bool,bool,bool): Selects the outputs to compute
        has_draw (bool, optional): if True, it draw. Defaults to False.
        kind (str): It draw 'amplitude_phase' or 'real_imag'

    Returns:
        _type_: t_TM, t_TE, r_TM, r_TE  (TM is parallel and TE is perpendicular)
    """

    outputs = np.array(outputs)

    k0 = 2 * np.pi / wavelength

    kz_1 = np.sqrt((n1 * k0) ** 2 - kx**2)

    alpha = (n2 * k0) ** 2 - kx**2
    normal = alpha >= 0
    reflexion_total = alpha < 0

    kz_2 = np.zeros_like(kx, dtype=complex)
    kz_2[normal] = np.sqrt(alpha[normal])
    kz_2[reflexion_total] = 1.0j * np.sqrt(-alpha[reflexion_total])

    t_TM, t_TE, r_TM, r_TE = None, None, None, None

    if outputs[0]:
        t_TM = 2 * n1 * n2 * kz_1 / (n2**2 * kz_1 + n1**2 * kz_2)  # parallel
    if outputs[1]:
        t_TE = 2 * kz_1 / (kz_1 + kz_2)  # perpendicular
    if outputs[2]:
        r_TM = (n2**2 * kz_1 - n1**2 * kz_2) / (
            n2**2 * kz_1 + n1**2 * kz_2
        )  # parallel
    if outputs[3]:
        r_TE = (kz_1 - kz_2) / (kz_1 + kz_2)  # perpendicular

    if has_draw and outputs.sum() > 0:
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))

        if kind == "amplitude_phase":
            if outputs[0]:
                axs[0].plot(kx, np.abs(t_TM), "r", label=r"$t_{\parallel, TM}$")
            if outputs[1]:
                axs[0].plot(kx, np.abs(t_TE), "b", label=r"$t_{\perp, TE}$")
            if outputs[2]:
                axs[0].plot(kx, np.abs(r_TM), "r-.", label=r"$r_{\parallel, TM}$")
            if outputs[3]:
                axs[0].plot(kx, np.abs(r_TE), "b-.", label=r"$r_{\perp, TE}$")

            axs[0].legend()
            axs[0].grid()

            axs[1].set_xlim(kx[0], kx[-1])
            axs[0].set_xlabel(r"$k_x$")
            axs[0].set_title("amplitude")

            if outputs[0]:
                axs[1].plot(kx, np.angle(t_TM)/degrees, "r", label=r"$t_{\parallel, TM}$")
            if outputs[1]:
                axs[1].plot(kx, np.angle(t_TE)/degrees, "b", label=r"$t_{\perp, TE}$")
            if outputs[2]:
                axs[1].plot(kx, np.angle(r_TM)/degrees, "r-.", label=r"$r_{\parallel, TM}$")
            if outputs[3]:
                axs[1].plot(kx, np.angle(r_TE)/degrees, "b-.", label=r"$r_{\perp, TE}$")

            axs[1].legend()
            axs[1].grid()
            axs[1].set_xlim(kx[0], kx[-1])
            axs[0].set_xlabel(r"$k_x$")
            axs[1].set_title(r"phase $\, (^{\circ})$")
            axs[1].set_ylim(-190, 190)
            axs[1].set_yticks([-180, -90, 0, 90, 180])

        elif kind == "real_imag":
            if outputs[0]:
                axs[0].plot(kx, np.real(t_TM), "r", label=r"$t_{\parallel, TM}$")
            if outputs[1]:
                axs[0].plot(kx, np.real(t_TE), "b", label=r"$t_{\perp, TE}$")
            if outputs[2]:
                axs[0].plot(kx, np.real(r_TM), "r-.", label=r"$r_{\parallel, TM}$")
            if outputs[3]:
                axs[0].plot(kx, np.real(r_TE), "b-.", label=r"$r_{\perp, TE}$")

            axs[0].legend()
            axs[0].grid()

            axs[0].set_xlim(kx[0], kx[-1])
            axs[0].set_xlabel(r"$k_x$")
            axs[0].set_title("real")

            if outputs[0]:
                axs[1].plot(kx, np.imag(t_TM)/degrees, "r", label=r"$t_{\parallel, TM}$")
            if outputs[1]:
                axs[1].plot(kx, np.imag(t_TE)/degrees, "b", label=r"$t_{\perp, TE}$")
            if outputs[2]:
                axs[1].plot(kx, np.imag(r_TM)/degrees, "r-.", label=r"$r_{\parallel, TM}$")
            if outputs[3]:
                axs[1].plot(kx, np.imag(r_TE)/degrees, "b-.", label=r"$r_{\perp, TE}$")

            axs[1].legend()
            axs[1].grid()
            axs[1].set_xlim(kx[0], kx[-1])
            axs[1].set_xlabel(r"$k_x$")
            axs[1].set_title(r"imag")

    return t_TM, t_TE, r_TM, r_TE  # parallel, perpendicular


def transmitances_reflectances_kx(kx: NDArrayComplex, wavelength: float, n1: float, n2: float,
                                  outputs: tuple[bool, bool, bool, bool] = [True, True, True, True], has_draw: bool = True):
    """Transmitances and reflectances, where input are kx part of wavevector.

    Args:
        kx (np.array): kx
        wavelength (float): wavelength
        n1 (float): refractive index of first materia
        n2 (float): refractive index of second materia
        outputs (bool,bool,bool,bool): Selects the outputs to compute
        has_draw (bool, optional): if True, it draw. Defaults to False.
        outputs (bool,bool,bool,bool): Selects the outputs to compute

    Returns:
        _type_: T_TM, T_TE, R_TM, R_TE  (TM is parallel and TE is perpendicular)
    """

    outputs = np.array(outputs)

    k0 = 2 * np.pi / wavelength

    kz_1 = np.sqrt((n1 * k0) ** 2 - kx**2)

    alpha = (n2 * k0) ** 2 - kx**2
    normal = alpha >= 0
    reflexion_total = alpha < 0

    kz_2 = np.zeros_like(kx, dtype=complex)
    kz_2[normal] = np.sqrt(alpha[normal])
    kz_2[reflexion_total] = 1.0j * np.sqrt(-alpha[reflexion_total])

    t_TM, t_TE, r_TM, r_TE = fresnel_equations_kx(
        kx, wavelength, n1, n2, outputs, has_draw=False
    )

    T_TM, T_TE, R_TM, R_TE = None, None, None, None

    if outputs[0]:
        T_TM = np.real(kz_2 / kz_1 * np.abs(t_TM ** 2))
    if outputs[1]:
        T_TE = np.real(kz_2 / kz_1 * np.abs(t_TE) ** 2)
    if outputs[2]:
        R_TM = np.abs(r_TM) ** 2
    if outputs[3]:
        R_TE = np.abs(r_TE) ** 2

    if has_draw:
        plt.figure()
        if outputs[0]:
            plt.plot(kx, T_TM, "r", label=r"$t_{\parallel, TM}$")
        if outputs[1]:
            plt.plot(kx, T_TE, "b", label=r"$t_{\perp, TE}$")
        if outputs[2]:
            plt.plot(kx, R_TM, "r-.", label=r"$r_{\parallel, TM}$")
        if outputs[3]:
            plt.plot(kx, R_TE, "b-.", label=r"$r_{\perp, TE}$")

        plt.xlabel(r"$k_x$")
        plt.legend()
        plt.grid()

        # plt.xlim(theta[0]/degrees, theta[-1]/degrees)

    return T_TM, T_TE, R_TM, R_TE  # parallel, perpendicular


def fresnel_equations(theta: NDArrayFloat, wavelength: float, n1: float, n2: float,
                      outputs: tuple[bool, bool, bool, bool] = [True, True, True, True], has_draw: bool = True,
                      kind="amplitude_phase"):
    """Fresnel equations and reflectances, where input are angles of incidence.

    Args:
        theta (np.array): kx
        wavelength (float): wavelength
        n1 (float): refractive index of first material
        n2 (float): refractive index of second material
        outputs (bool,bool,bool,bool): Selects the outputs to compute
        kind (str): It draw 'amplitude_phase' or 'real_imag'
        has_draw (bool, optional): if True, it draw. Defaults to False.
        kind (str): It draw 'amplitude_phase' or 'real_imag'

    Returns:
        _type_: T_TM, T_TE, R_TM, R_TE  (TM is parallel and TE is perpendicular)
    """

    outputs = np.array(outputs)

    k0 = 2 * np.pi / wavelength
    kx = n1 * k0 * np.sin(theta)

    t_TM, t_TE, r_TM, r_TE = fresnel_equations_kx(kx, wavelength, n1, n2, outputs, has_draw=False)

    if has_draw and outputs.sum() > 0:
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))
        if kind == "amplitude_phase":
            if outputs[0]:
                axs[0].plot(theta/degrees, np.abs(t_TM), "r", label=r"$t_{\parallel, TM}$")
            if outputs[1]:
                axs[0].plot(theta/degrees, np.abs(t_TE), "b", label=r"$t_{\perp, TE}$")
            if outputs[2]:
                axs[0].plot(theta/degrees, np.abs(r_TM), "r-.", label=r"$r_{\parallel, TM}$")
            if outputs[3]:
                axs[0].plot(theta/degrees, np.abs(r_TE), "b-.", label=r"$r_{\perp, TE}$")

            axs[0].legend()
            axs[0].grid()

            axs[0].set_xlim(theta[0]/degrees, theta[-1]/degrees)
            axs[0].set_xlabel(r"$\theta \, (^{\circ})$")
            axs[0].set_title("amplitude")

            if outputs[0]:
                axs[1].plot(
                    theta/degrees,
                    np.angle(t_TM)/degrees,
                    "r",
                    label=r"$t_{\parallel, TM}$",
                )
            if outputs[1]:
                axs[1].plot(
                    theta/degrees,
                    np.angle(t_TE)/degrees,
                    "b",
                    label=r"$t_{\perp, TE}$",
                )
            if outputs[2]:
                axs[1].plot(
                    theta/degrees,
                    np.angle(np.abs(r_TM))/degrees,
                    "r-.",
                    label=r"$r_{\parallel, TM}$",
                )
            if outputs[3]:
                axs[1].plot(
                    theta/degrees,
                    np.angle(np.abs(r_TE))/degrees,
                    "b-.",
                    label=r"$r_{\perp, TE}$",
                )

            axs[1].legend()
            axs[1].grid()
            axs[1].set_xlim(theta[0]/degrees, theta[-1]/degrees)
            axs[1].set_xlabel(r"$\theta \, (^{\circ})$")
            axs[1].set_title(r"phase $\, (^{\circ})$")
            axs[1].set_ylim(-190, 190)
            axs[1].set_yticks([-180, -90, 0, 90, 180])

        elif kind == "real_imag":
            if outputs[0]:
                axs[0].plot(theta/degrees, np.real(t_TM), "r", label=r"$t_{\parallel, TM}$")
            if outputs[1]:
                axs[0].plot(theta/degrees, np.real(t_TE), "b", label=r"$t_{\perp, TE}$")
            if outputs[2]:
                axs[0].plot(
                    theta/degrees, np.real(r_TM), "r-.", label=r"$r_{\parallel, TM}$"
                )
            if outputs[3]:
                axs[0].plot(
                    theta/degrees, np.real(r_TE), "b-.", label=r"$r_{\perp, TE}$"
                )

            axs[0].legend()
            axs[0].grid()
            axs[0].set_xlabel(r"$\theta \, (^{\circ})$")
            axs[0].set_xlim(theta[0]/degrees, theta[-1]/degrees)
            axs[0].set_title("real")

            if outputs[0]:
                axs[1].plot(
                    theta/degrees, np.imag(t_TM)/degrees, "r", label=r"$t_{\parallel, TM}$"
                )
            if outputs[1]:
                axs[1].plot(
                    theta/degrees,
                    np.imag(t_TE)/degrees,
                    "b",
                    label=r"$t_{\perp, TE}$",
                )
            if outputs[2]:
                axs[1].plot(
                    theta/degrees,
                    np.imag(r_TM)/degrees,
                    "r-.",
                    label=r"$r_{\parallel, TM}$",
                )
            if outputs[3]:
                axs[1].plot(
                    theta/degrees,
                    np.imag(r_TE)/degrees,
                    "b-.",
                    label=r"$r_{\perp, TE}$",
                )

            axs[1].legend()
            axs[1].grid()
            axs[1].set_xlim(theta[0]/degrees, theta[-1]/degrees)
            axs[1].set_xlabel(r"$\theta \, (^{\circ})$")
            axs[1].set_title(r"imag")

    return t_TM, t_TE, r_TM, r_TE  # parallel, perpendicular


def transmitances_reflectances(theta: NDArrayFloat, wavelength: float, n1: float, n2: float,
                               outputs: tuple[bool] = [True, True, True, True], has_draw: bool = False):
    """Transmitances and reflectances, where input are angles of incidence.

    Args:
        theta (np.array): angles
        wavelength (float): wavelength
        n1 (float): refractive index of first materia
        n2 (float): refractive index of second materia
        outputs(bool,bool,bool,bool): Selects the outputs to compute
        has_draw (bool, optional): if True, it draw. Defaults to False.

    Returns:
        _type_: T_TM, T_TE, R_TM, R_TE  (TM is parallel and TE is perpendicular)
    """
    outputs = np.array(outputs)

    k0 = 2 * np.pi / wavelength
    kx = k0 * n1 * np.sin(theta)

    T_TM, T_TE, R_TM, R_TE = transmitances_reflectances_kx(
        kx, wavelength, n1, n2, outputs, has_draw=False
    )

    if has_draw:
        plt.figure()
        if outputs[0]:
            plt.plot(theta/degrees, T_TM, "r", label=r"$t_{\parallel, TM}$")
        if outputs[1]:
            plt.plot(theta/degrees, T_TE, "b", label=r"$t_{\perp, TE}$")
        if outputs[2]:
            plt.plot(theta/degrees, R_TM, "r-.", label=r"$r_{\parallel, TM}$")
        if outputs[3]:
            plt.plot(theta/degrees, R_TE, "b-.", label=r"$r_{\perp, TE}$")

        plt.xlim(theta[0]/degrees, theta[-1]/degrees)
        plt.xlabel(r"$\theta \, (^{\circ})$")
        plt.legend()
        plt.grid()

    return T_TM, T_TE, R_TM, R_TE  # parallel, perpendicular



def determine_extrema(I_far: np.array, angles_x: np.array,  is_angles: bool = False, change_order_0: bool = True,  
                      has_draw: bool = True, has_logarithm: bool = True,  verbose: bool = True,
                      **kwargs):
        
    """
     Determine the extrema of a 1D far field diffraction pattern. 
     
     It can be in positions x or angles.
    
    Args:
        I_far (np.array): Intensity distribution of the far field
        angles_x (np.array): angles of the far field.
        is_angles (bool): if True, the far field is in angles, if False, the far field is in x.
        change_order_0 (bool): if True, the central maxima is included in the minima
        has_draw (bool): It draws the far field with the maxima and minima.
        has_logarithm (bool): It draws the far field with logarithm or not.
        verbose (bool): if True, it prints the maxima and minima.
        

    Returns:
        (i_minima, i_maxima): List with indexes of minima and maxima
        (angles[i_minima], angles[i_maxima]): List with angles of minima and maxima
        (I_far[i_minima], I_far[i_maxima]): List with intensities of minima and maxima
    """
        
    i_minima = argrelextrema(I_far, np.less)
    i_minima = np.array(i_minima).flatten()

    i_maxima = argrelextrema(I_far, np.greater)
    i_maxima = np.array(i_maxima).flatten()
    i_central_max = np.argmax(I_far)

    if change_order_0:
        i_minima= np.append(i_minima, i_central_max)
        i_minima = np.sort(i_minima)
        
    
    if verbose:
        if is_angles:
            print("Central maxima: {:2.2f}, angle: {} degrees".format(I_far[i_central_max], angles_x[i_central_max]/degrees))
            print("Angles minima:")
            print(angles_x[i_minima]/degrees)
            print("Angles maxima:")
            print(angles_x[i_maxima]/degrees)
        else:
            print("Central maxima: {:2.2f}, position: {} ".format(I_far[i_central_max], angles_x[i_central_max]))
            print("Positions minima:")
            print(angles_x[i_minima]/um)
            print("Positions maxima:")
            print(angles_x[i_maxima]/um)
    
    if has_draw:
        if has_logarithm:
            function = plt.semilogy
        else:
            function = plt.plot

        plt.figure(**kwargs)
        
        if is_angles:
            function(angles_x/degrees, I_far, 'k')
            function(angles_x[i_maxima]/degrees, I_far[i_maxima], 'ro')
            function(angles_x[i_minima]/degrees, I_far[i_minima], 'bo')

            plt.ylim(I_far.min(),I_far.max())
            plt.xlim(angles_x[0]/degrees, angles_x[-1]/degrees)
            plt.xlabel(r'angles $\,(^{\circ})$')
            plt.grid('on')
        else:
            function(angles_x, I_far, 'k')
            function(angles_x[i_maxima], I_far[i_maxima], 'ro')
            function(angles_x[i_minima], I_far[i_minima], 'bo')

            plt.ylim(I_far.min(),I_far.max())
            plt.xlim(angles_x[0]/um, angles_x[-1]/um)
            plt.xlabel(r'x($\mu$m)')
            plt.grid('on')         
        
    return (i_minima, i_maxima), (angles_x[i_minima], angles_x[i_maxima]), (I_far[i_minima], I_far[i_maxima])


def size_from_diffraction_minima(angles_minima: np.array, wavelength, size_slit: float | None = None, 
                                 has_draw: bool = False, verbose: bool = False):

    """We have the minima of a 1D diffraction pattern and determine the size.

    _extended_summary_

    Returns:
        angles_minima (np.array):
        wavelength (float):
        size_slit (float | None):
        has_draw (bool): It draws the far field with the maxima and minima.
        verbose (bool): if True, it prints the maxima and minima.
    """
    diff_angles = np.diff((angles_minima))
    diff_angles = np.diff(np.sin(angles_minima))
    sizes_slit = wavelength/diff_angles
    
    # i_bad = np.where(sizes_slit<0.6*size_slit)
    # sizes_slit[i_bad] = sizes_slit[i_bad]*2

    # i_good = np.where(sizes_slit<1.2*size_slit)
    # size_slit_measured_center = sizes_slit[i_good].max()

    diameter_fitting = np.polyfit(angles_minima[0:-1], sizes_slit, 2)
    estimated_diameter_fitting = diameter_fitting[2]
    size_slit_measured_center = estimated_diameter_fitting

    if size_slit is not None:
        percent_error_size_slit_center = 100*(size_slit_measured_center-size_slit)/size_slit
        error_size_slit_center = (size_slit_measured_center-size_slit)
    else:
        error_size_slit_center=None

    
    # quadratic fitting to the diffraction minima
    fitting = np.poly1d(diameter_fitting)
    diameter_squared = fitting(angles_minima)

    if has_draw:

        plt.figure(figsize=(20,5))

        plt.plot(angles_minima[0:-1]/degrees, sizes_slit, 'kx', label='local size')
        if size_slit is not None:
            plt.plot(np.array([angles_minima[0], angles_minima[-1]])/degrees, (size_slit, size_slit), 'r--', label='real size')
        plt.plot(np.array([angles_minima[0], angles_minima[-1]])/degrees, (size_slit_measured_center, size_slit_measured_center), 'g--', label='measured size center')
        plt.plot(angles_minima/degrees, diameter_squared, 'k', label='fitting')

        plt.legend()
        
        if size_slit is not None:
            plt.title(f" {size_slit/um:.2f}  Measured slit size:  {size_slit_measured_center/um:.2f}, error:  {error_size_slit_center*1000:.2f} nm = {percent_error_size_slit_center:.2f}%")

        plt.xlim(angles_minima[0]/degrees, angles_minima[-1]/degrees)
        plt.xlabel(r'$\theta$ (degrees)')
        plt.ylabel("Diameter estimation")


    if verbose:
        print("sizes_slit")
        print(sizes_slit)
        print(f"estimated diameter: {estimated_diameter_fitting/um:.2f} um")

        
    return sizes_slit, size_slit_measured_center, error_size_slit_center


def envelopes(angles: np.array, I_far: np.array, has_draw: bool = True, has_logarithm: bool = True):
    """Generates envelopes, and also contrast from these envelopes.

    Args:
        angles (np.array): angles for the observaton
        I_far (np.array): Intensity at far field
        has_draw (bool, optional): If True, makes some drawingT. Defaults to True.
        has_logarithm (bool, optional): If True, drawings are in logarithm scale. Defaults to True.

    Returns:
        I_max_interpolated (np.array)
        I_min_interpolated (np.array))
        Contrast (np.array)
    
    TODO:  
        Improve: The envolvente should be always above or below the diffraction pattern.
        1. Find local minima at difference
        2. Move points at envolvente to this local minima
        3. New interpolation
    """
    
    i_extrema, angles_extrema, I_extrema = determine_extrema(I_far = I_far, angles_x = angles, change_order_0 = False,
                                                             has_draw = False, has_logarithm=has_logarithm, verbose = False)

    angles_minima = angles_extrema[0]
    angles_maxima = angles_extrema[1]
    
    I_minima = I_extrema[0]
    I_maxima = I_extrema[1]
    
    
    spl_max  = CubicSpline(angles_maxima, I_maxima)
    spl_max  = PchipInterpolator(angles_maxima, I_maxima)
    spl_max  = Akima1DInterpolator(angles_maxima, I_maxima)
    I_max_interpolated = spl_max (angles)


    spl_min  = CubicSpline(angles_minima, I_minima)
    spl_min  = PchipInterpolator(angles_minima, I_minima)
    spl_min  = Akima1DInterpolator(angles_minima, I_minima)
    I_min_interpolated  = spl_min (angles)


    differences  = I_far - I_max_interpolated
    i_to_solve = np.where(differences>0)
    
    
    if has_draw:
        plt.figure(figsize=(20,5))
        plt.plot(angles/degrees, differences, 'k')
        plt.plot(angles/degrees, np.zeros_like(angles), 'k-.')
        plt.title('Differences')
        plt.plot(angles[i_to_solve]/degrees, differences[i_to_solve], 'ro')
        plt.plot(angles_maxima/degrees, np.zeros_like(angles_maxima), 'go')
        plt.xlabel(r'$\theta$ (degrees)')
        plt.ylabel("Enolventes")

        
    # angles_maxima_envolvente = angles_maxima
    
    # diff_maxima = np.diff(angles_maxima).max()/10    
    
    if has_logarithm:
        function = plt.semilogy
    else:
        function = plt.plot

    Contrast = (I_max_interpolated - I_min_interpolated)/(I_max_interpolated + I_min_interpolated)

    if has_draw:
        plt.figure(figsize=(20,5))
        plt.plot(angles/degrees, Contrast, 'k')
        plt.ylim(0,1.05)
        plt.xlim(angles[0]/degrees, angles[-1]/degrees)
        plt.xlabel(r'$\theta$ (degrees)')
        plt.ylabel("contrast")


        plt.figure(figsize=(20,5))
        function(angles/degrees, I_far, 'k')
        function(angles_maxima/degrees, I_maxima, 'ro')
        function(angles/degrees, I_max_interpolated, 'r')  # nejor interpolacion por splines ?

        function(angles_minima/degrees, I_minima, 'bo')
        function(angles/degrees, I_min_interpolated , 'b')  # nejor interpolacion por splines ?

        plt.xlim(angles[0]/degrees, angles[-1]/degrees)
        plt.xlabel(r'$\theta$ (degrees)')
        plt.ylabel("envelopes")


    return I_max_interpolated, I_min_interpolated, Contrast