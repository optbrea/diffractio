# !/usr/bin/env python3

# ----------------------------------------------------------------------
# Name:        scalar_sources_X.py
# Purpose:     Define the Scalar_source_X class for unidimensional scalar sources
#
# Author:      Luis Miguel Sanchez Brea
#
# Created:     2024
# Licence:     GPLv3
# ----------------------------------------------------------------------


"""
This module generates Scalar_field_X class for defining sources.
Its parent is Scalar_field_X.

The main atributes are:
    * self.u - field
    * self.x - x positions of the field
    * self.wavelength - wavelength of the incident field. The field is monocromatic

The magnitude is related to microns: `mifcron = 1.`

*Class for unidimensional scalar masks*

*Functions*
    * plane_wave
    * gauss_beam
    * spherical_wave
    * plane_waves_dict
    * plane_waves_several_inclined
    * gauss_beams_several_parallel
    * gauss_beams_several_inclined

*Also*
    * Polychromatic and extendes sources are defined in scalar_fields_X.py for multiprocessing purposes.
"""
# flake8: noqa



from .__init__ import degrees, np, um
from .utils_typing import npt, Any, NDArray,  NDArrayFloat, NDArrayComplex
from .utils_common import check_none
from .config import bool_raise_exception
from .scalar_fields_X import Scalar_field_X


class Scalar_source_X(Scalar_field_X):
    """Class for unidimensional scalar sources.

    Args:
        x (numpy.array): linear array with equidistant positions. The number of data is preferibly :math:`2^n` .
        wavelength (float): wavelength of the incident field
        n_background (float): refractive index of background
        info (str): String with info about the simulation

    Attributes:
        self.x (numpy.array): linear array with equidistant positions. The number of data is preferibly :math:`2^n` .
        self.wavelength (float): wavelength of the incident field.
        self.u (numpy.array): equal size than x. complex field
        self.quality (float): quality of RS algorithm
        self.info (str): description of data
        self.type (str): Class of the field
        self.date (str): date when performed
    """

    def __init__(self, x: NDArrayFloat | None = None, wavelength: float = 0,
                 n_background: float = 1., info: str = ""):
        super().__init__(x, wavelength, n_background, info)
        self.type = 'Scalar_source_X'

    @check_none('x', raise_exception=bool_raise_exception)
    def plane_wave(self, A: float = 1., theta: float = 0., z0: float = 0.):
        """Plane wave. 

        Args:
            A (float): maximum amplitude
            theta (float): angle in radians
            z0 (float): constant value for phase shift
        """
        # Definicion del vector de onda
        k = 2 * np.pi / self.wavelength
        self.u = A * np.exp(1.j * k * (self.x * np.sin(theta) + z0 * np.cos(theta)))


    @check_none('x', raise_exception=bool_raise_exception)
    def gauss_beam(self, x0: float, w0: float, z0: float, A: float = 1, theta: float = 0.):
        """Gauss Beam.

        Args:
            x0 (float): x position of center
            w0 (float): minimum beam width
            z0 (float): position of beam width
            A (float): maximum amplitude
            theta (float): angle in radians
        """
        k = 2 * np.pi / self.wavelength
        # distance de Rayleigh solo para una direccion.
        z_rayleigh = k * w0**2/2

        phaseGouy = np.arctan2(z0, z_rayleigh)

        w = w0 * np.sqrt(1 + (z0 / z_rayleigh)**2)
        if z0 == 0:
            R = 1e10
        else:
            R = -z0 * (1 + (z_rayleigh / z0)**2)
        amplitude = A * w0 / w * np.exp(-(self.x - x0)**2 / (w**2))
        phase1 = np.exp(1j * k * ((self.x - x0) * np.sin(theta)))  # rotation
        phase2 = np.exp(1j * (-k * z0 - phaseGouy + k * (self.x - x0)**2 /
                              (2 * R)))

        self.u = amplitude * phase1 * phase2


    @check_none('x', raise_exception=bool_raise_exception)
    def spherical_wave(self, A: float, x0: float, z0: float, normalize: bool = False):
        """Spherical wave. self.u = amplitude * A * np.exp(-1.j * np.sign(z0) * k * Rz) / Rz

        Args:
            A (float): maximum amplitude
            x0 (float): x position of source
            z0 (float): z position of source
            mask (bool): If true, masks the spherical wave with radius
            normalize (bool): If True, maximum of field is 1
        """
        k = 2 * np.pi / self.wavelength

        Rz = np.sqrt((self.x - x0)**2 + z0**2)

        # Onda esferica
        self.u = A * np.exp(-1.j * np.sign(z0) * k * Rz) / Rz

        if normalize is True:
            self.u = self.u / np.abs(self.u.max() + 1.012034e-12)


    @check_none('x', raise_exception=bool_raise_exception)
    def plane_waves_dict(self, params: dict):
        """Several plane waves with parameters defined in dictionary

        Args:
            params: list with a dictionary:
                A (float): maximum amplitude
                theta (float): angle in radians
                z0 (float): constant value for phase shift
        """
        # Definicion del vector de onda
        k = 2 * np.pi / self.wavelength

        self.u = np.zeros_like(self.x, dtype=complex)
        for p in params:
            self.u = self.u + p['A'] * np.exp(
                1.j * k *
                (self.x * np.sin(p['theta']) + p['z0'] * np.cos(p['theta'])))


    @check_none('x', raise_exception=bool_raise_exception)
    def plane_waves_several_inclined(self, A: float, num_beams: int, max_angle: float):
        """Several paralel plane waves.

        Args:
            A (float): maximum amplitude
            num_beams (int): number of ints
            max_angle (float): maximum angle for beams
        """

        t = np.zeros_like(self.x, dtype=complex)
        angle = max_angle / num_beams
        for i in range(num_beams):
            theta = -max_angle/2 + angle * (i + 0.5)
            self.plane_wave(theta=theta, z0=0)
            t = t + self.u
        self.u = t


    @check_none('x', raise_exception=bool_raise_exception)
    def gauss_beams_several_parallel(self, A: float, num_beams: int, w0: float, z0: float, x_central: float, x_range: float, theta: float = 0.):
        """Several parallel gauss beams

        Args:
            A (float): maximum amplitude
            num_beams (int): number of gaussian beams (equidistintan)
            w0 (float): beam width of the bemas
            z0 (float): constant value for phase shift
            x_central (float): central position of rays
            x_range (float): range of rays
            theta (float): angle of the parallel beams
        """

        t = np.zeros_like(self.x, dtype=complex)
        distancia = x_range / num_beams
        for i in range(num_beams):
            xi = x_central - x_range/2 + distancia * (i + 0.5)
            self.gauss_beam(x0=xi, w0=w0, z0=z0, A=A, theta=theta)
            t = t + self.u
        self.u = t

    @check_none('x', raise_exception=bool_raise_exception)
    def gauss_beams_several_inclined(self, A: float, num_beams: int, w0: float, x0: float, z0: float, max_angle: float):
        """Several inclined gauss beams

        Args:
            A (float): maximum amplitude
            num_beams (int): number of gaussian beams (equidistintan)
            w0 (float): beam width of the bemas
            x0 (fl(float): maximum amplitude
            num_beams (int): number of ints
            maoat): initial position of gauss beam at x
            z0 (float): constant value for phase shift
            max_angle (float): maximum angle for beams
        """

        t = np.zeros_like(self.x, dtype=complex)
        angle = max_angle / num_beams
        for i in range(num_beams):
            thetai = -max_angle/2 + angle * (i + 0.5)
            self.gauss_beam(x0=x0, w0=w0, z0=z0, A=A, theta=thetai)
            t = t + self.u
        self.u = t
