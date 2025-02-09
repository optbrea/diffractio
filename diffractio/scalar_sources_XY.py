# !/usr/bin/env python3

# ----------------------------------------------------------------------
# Name:        scalar_sources_XY.py
# Purpose:     Define the Scalar_source_XY class for generating scalar sources
#
# Author:      Luis Miguel Sanchez Brea
# Number of lines in this file: 500
#
# Created:     2024
# Licence:     GPLv3
# ----------------------------------------------------------------------


"""
This module generates Scalar_source_XY class for defining sources.
Its parent is Scalar_field_XY.

The main atributes are:
    * self.x - x positions of the field
    * self.y - y positions of the field
    * self.u - field XY
    * self.wavelength - wavelength of the incident field. The field is monocromatic

The magnitude is related to microns: `micron = 1.`

*Class for unidimensional scalar masks*

*Functions*
    * plane_wave
    * gauss_beam
    * spherical_wave
    * vortex_beam
    * laguerre_beam
    * hermite_gauss_beam
    * zernike_beam
    * bessel_beam
    * plane_waves_dict
    * plane_waves_several_inclined
    * gauss_beams_several_parallel
    * gauss_beams_several_inclined

*Also*
    * laguerre_polynomial_nk
    * fZernike
    * delta_kronecker
"""
# flake8: noqa

import os
from math import factorial
from scipy.special import eval_hermite, j0, j1, jv


from .__init__ import np
from .utils_typing import NDArrayFloat
from .utils_common import check_none
from .config import bool_raise_exception
from .scalar_fields_XY import Scalar_field_XY
from .utils_math import fZernike, laguerre_polynomial_nk


class Scalar_source_XY(Scalar_field_XY):
    """Class for XY scalar sources.

    Args:
        x (numpy.array): linear array with equidistant positions. The number of data is preferibly :math:`2^n` .
        y (numpy.array): linear array wit equidistant positions for y values
        wavelength (float): wavelength of the incident field
        info (str): String with info about the simulation

    Attributes:
        self.x (numpy.array): linear array with equidistant positions. The number of data is preferibly :math:`2^n` .
        self.y (numpy.array): linear array wit equidistant positions for y values
        self.wavelength (float): wavelength of the incident field.
        self.u (numpy.array): (x,z) complex field
        self.info (str): String with info about the simulation
    """

    def __init__(self, x: NDArrayFloat | None = None, y: NDArrayFloat | None = None,
                 wavelength: float | None = None,
                 n_background: float = 1., info: str = ""):
        super().__init__(x, y, wavelength, n_background, info)
        self.type = 'Scalar_source_XY'


    @check_none('X', 'Y', raise_exception=bool_raise_exception)
    def plane_wave(self, A: float = 1, theta: float = 0., phi: float = 0., z0: float = 0.):
        """Plane wave. self.u = A * np.exp(1.j * k *
                         (self.X * np.sin(theta) * np.cos(phi) +
                          self.Y * np.sin(theta) * np.sin(phi) + z0 * np.cos(theta)))

        According to https://en.wikipedia.org/wiki/Spherical_coordinate_system: physics (ISO 80000-2:2019 convention)

        Args:
            A (float): maximum amplitude
            theta (float): angle in radians
            phi (float): angle in radians
            z0 (float): constant value for phase shift
        """
        k = 2 * np.pi / self.wavelength
        self.u = A * np.exp(1.j * k *
                            (self.X * np.sin(theta) * np.cos(phi) +
                             self.Y * np.sin(theta) * np.sin(phi) + z0 * np.cos(theta)))


    @check_none('X', 'Y', raise_exception=bool_raise_exception)
    def gauss_beam(self,
                   r0: tuple[float, float],
                   w0: tuple[float, float] | float,
                   z0: tuple[float, float] | float,
                   alpha: float = 0.,
                   beta: float = 0.,
                   A: float = 1,
                   theta: float = 0.,
                   phi: float = 0.):
        """Gauss Beam.

        Args:
            r0 (float, float): (x,y) position of center
            w0 (float, float): (wx,wy) minimum beam width
            z0 (float): (z0x, z0y) position of beam width for each axis (could be different)
            alpha (float): rotation angle of the axis of the elliptical gaussian amplitude
            beta (float): rotation angle of the axis of the main directions of the wavefront (it can be different from alpha)
            A (float): maximum amplitude
            theta (float): angle in radians (angle of k with respect to z))

        """

        if isinstance(w0, (float, int, complex)):
            w0 = (w0, w0)

        if isinstance(z0, (float, int, complex)):
            z0 = (z0, z0)

        w0x, w0y = w0
        # w0 = np.sqrt(w0x * w0y)
        x0, y0 = r0
        z0x, z0y = z0
        k = 2 * np.pi / self.wavelength

        # only for x axis.
        z_rayleigh_x = k * w0x**2/2
        z_rayleigh_y = k * w0y**2/2

        phaseGouy_x = np.arctan2(z0x, z_rayleigh_x)
        phaseGouy_y = np.arctan2(z0y, z_rayleigh_y)

        wx = w0x * np.sqrt(1 + (z0x / z_rayleigh_x)**2)
        wy = w0y * np.sqrt(1 + (z0y / z_rayleigh_y)**2)

        if z0x == 0:
            R_x = 1e10
        else:
            R_x = -z0x * (1 + (z_rayleigh_x / z0x)**2)

        if z0y == 0:
            R_y = 1e10
        else:
            R_y = -z0y * (1 + (z_rayleigh_y / z0y)**2)

        amplitude = (A * (w0x / wx) * (w0y / wy) * np.exp(
            -(self.X * np.cos(alpha) + self.Y * np.sin(alpha) - x0)**2 /
            (wx**2)) * np.exp(
                -(-self.X * np.sin(alpha) + self.Y * np.cos(alpha) - y0)**2 /
                (wy**2)))
        phase1 = np.exp(1.j * k * (self.X * np.sin(theta) * np.cos(phi) +
                                   self.Y * np.sin(theta) * np.sin(phi)))
        phase2 = np.exp(1j * (k * z0x - phaseGouy_x + k * ((self.X * np.cos(beta) + self.Y * np.sin(beta))**2) / (2 * R_x))) * \
            np.exp(1j * (k * z0y - phaseGouy_y + k * ((-self.X * np.sin(beta) + self.Y * np.cos(beta))**2) / (2 * R_y)))

        self.u = amplitude * phase1 * phase2


    @check_none('X', 'Y', raise_exception=bool_raise_exception)
    def spherical_wave(self, r0: tuple[float, float], z0: tuple[float, float] | float, A: float = 1, 
                        radius: float =0., normalize: bool = False):
        """Spherical wave.

        Args:
            A (float): maximum amplitude
            r0 (float, float): (x,y) position of source
            z0 (float) or (float, float): z0 or (zx0, zy0) position of source. When two parameters, the source is stigmatic
            radius (float): radius for mask
            normalize (bool): If True, maximum of field is 1
        """

        k = 2 * np.pi / self.wavelength
        x0, y0 = r0
        if isinstance(z0, (int, float)):
            z0 = (z0, z0)
        z0x, z0y = z0

        R2 = (self.X - x0)**2 + (self.Y - y0)**2

        if z0x == 0:
            R_x = 1e10
        else:
            R_x = np.sqrt((self.X - x0)**2 + z0x**2)

        if z0y == 0:
            R_y = 1e10
        else:
            R_y = np.sqrt((self.Y - y0)**2 + z0y**2)

        if radius > 0:
            amplitude = (R2 <= radius**2)
        else:
            amplitude = 1

        self.u = amplitude * A * np.exp(-1.j * np.sign(z0x) * k * R_x)*np.exp(-1.j * np.sign(z0y) * k * R_y) / np.sqrt(R_x*R_y)

        if normalize is True:
            self.u = self.u / np.abs(self.u.max() + 1.012034e-12)


    @check_none('X', 'Y', raise_exception=bool_raise_exception)
    def vortex_beam(self, A: float, r0: tuple[float, float], w0: tuple[float, float] | float, m: int):
        """Vortex beam.

        Args:
            A (float): Amplitude
            r0 (float, float): (x,y) position of source
            w0 (float): width of the vortex beam
            m (int): order of the vortex beam

        Example:
            vortex_beam(r0=(0*um, 0*um), w0=100*um, m=1)
        """

        if isinstance(w0, (float, int, complex)):
            w0x, w0y = w0, w0
        else:
            w0x, w0y = w0

        x0, y0 = r0
        amplitude = ((self.X - x0) + 1.j * np.sign(m) *
                     (self.Y - y0))**np.abs(m) * np.exp(-(
                         (self.X - x0)**2 / w0x**2 +
                         (self.Y - y0)**2 / w0y**2))

        self.u = A * amplitude / np.abs(amplitude).max()


    @check_none('X', 'Y', raise_exception=bool_raise_exception)
    def hermite_gauss_beam(self, r0: tuple[float, float], A: float,
                           w0: tuple[float, float] | float, n: int, m: int, z: float,
                           z0: tuple[float, float] | float):
        """Hermite Gauss beam.

        Args:
            A (float): amplitude of the Hermite Gauss beam.
            r0 (float, float): (x,y) position of the beam center.
            w0 (float, float): Gaussian waist.
            n (int): order in x.
            m (int): order in y.
            z (float): Propagation distance.
            z0 (float, float): Beam waist position at each dimension

        Example:
             hermite_gauss_beam(A: float = 1, r0=(0*um, 0*um), w0=(100*um, 50*um), n=2, m=3, z=0)
        """
        # Prepare space
        X = self.X - r0[0]
        Y = self.Y - r0[1]

        r2 = np.sqrt(2)

        if isinstance(w0, (float, int, complex)):
            w0x, w0y = w0, w0
        else:
            w0x, w0y = w0

        if isinstance(z0, (float, int, complex)):
            z0x, z0y = z0, z0
        else:
            z0x, z0y = z0

        k = 2 * np.pi / self.wavelength

        # Calculate propagation
        zx = z - z0x
        zRx = k * w0x**2/2
        wx = w0x * np.sqrt(1 + zx**2 / zRx**2)
        if zx == 0:
            Rx = np.inf
        else:
            Rx = zx + zRx**2 / zx

        zy = z - z0y
        zRy = k * w0y**2/2
        wy = w0y * np.sqrt(1 + zy**2 / zRy**2)
        if zy == 0:
            Ry = np.inf
        else:
            Ry = zy + zRy**2 / zy

        # Calculate amplitude
        A = A * np.sqrt(2**(1 - n - m) /
                        (np.pi * factorial(n) * factorial(m))) * np.sqrt(w0x * w0y /
                                                                         (wx * wy))
        Ex = eval_hermite(n, r2 * X / wx) * np.exp(-X**2 / wx**2)
        Ey = eval_hermite(m, r2 * Y / wy) * np.exp(-Y**2 / wy**2)

        # Calculate phase
        Ef = np.exp(1j * k * (X**2 / Rx + Y**2 / Ry)) * np.exp(
            -1j * (0.5 + n) * np.arctan(zx / zRx)) * np.exp(
                -1j * (0.5 + m) * np.arctan(zy / zRy)) * np.exp(1j * k *
                                                                (zx + zy)/2)

        self.u = A * Ex * Ey * Ef


    @check_none('X', 'Y', raise_exception=bool_raise_exception)
    def laguerre_beam(self, r0: tuple[float, float], A: float,
                      w0: tuple[float, float] | float, n: int, l: int,
                      z: float, z0: float):
        """Laguerre beam.

        Args:
            A (float): amplitude of the Hermite Gauss beam.
            r0 (float, float): (x,y) position of the beam center.
            w0 (float): Gaussian waist.
            n (int): radial order.
            l (int): angular order.
            z (float): Propagation distance.
            z0 (float): Beam waist position.

        Example:
            laguerre_beam(A: float = 1, r0=(0*um, 0*um),  w0=1*um,  p=0, l=0,  z=0)
        """
        # Prepare space
        X = self.X - r0[0]
        Y = self.Y - r0[1]
        Ro2 = X**2 + Y**2
        Ro = np.sqrt(Ro2)
        Th = np.arctan2(Y, X)

        # Parameters
        r2 = np.sqrt(2)
        z = z - z0
        k = 2 * np.pi / self.wavelength

        # Calculate propagation
        zR = k * w0**2/2
        w = w0 * np.sqrt(1 + z**2 / zR**2)
        if z == 0:
            R = np.inf
        else:
            R = z + zR**2 / z

        # Calculate amplitude
        A = A * w0 / w
        Er = laguerre_polynomial_nk(2 * Ro2 / w**2, n, l) * np.exp(
            -Ro2 / w**2) * (r2 * Ro / w)**l

        # Calculate phase
        Ef = np.exp(1j * (k * Ro2 / R + l * Th)) * \
            np.exp(-1j * (1 + n) * np.arctan(z / zR))

        self.u = A * Er * Ef


    @check_none('X', 'Y', raise_exception=bool_raise_exception)
    def zernike_beam(self, A: float, r0: tuple[float, float], radius: float,
                     n: tuple[int], m: tuple[int], c_nm: tuple[float]):
        """Zernike beam.

        Args:
            A (float): amplitude of the Zernike beam beam
            r0 (float, float): (x,y) position of source
            radius (float): width of the beam
            n (tuple): list of integers with orders
            m (tuple): list of integers with orders
            c_nm (tuple): list of coefficients, in radians

        Example:
             zernike_beam(A: float = 1, r0=(0*um, 0*um), radius=5*mm, n=[1, 3, 3, 5, 5, 5], m=[1, 1, 3, 1, 3, 5], c_nm=[.25, 1, 1, 1, 1, 1])
        """

        # normalizing to radius 1
        x0, y0 = r0
        R = np.sqrt((self.X - x0)**2 + (self.Y - y0)**2) / radius

        # phase as sum of Zernike functions
        phase = np.zeros(self.X.shape, dtype=float)
        for s in range(len(n)):
            phase = phase + c_nm[s] * fZernike(self.X - x0, self.Y - y0, n[s], m[s], radius)

        self.u = A * np.exp(1.j * np.real(phase))


    @check_none('X', 'Y', raise_exception=bool_raise_exception)
    def bessel_beam(self,
                    A: float,
                    r0: tuple[float, float],
                    alpha: float,
                    n: int,
                    theta: float = 0.,
                    phi: float = 0.,
                    z0: float = 0):
        """Bessel beam produced by an axicon. Bessel-beams are generated unp.sing 2D axicons.

        Args:
            A (float): amplitude of the Bessel beam
            r0 (float, float): (x,y) position of source
            alpha (float): angle of the beam generator
            n (int): order of the beam
            theta (float): angle in radians
            phi (float): angle in radians
            z0 (float): constant value for phase shift

        References:
            J. Durnin, J. Miceli, and J. H. Eberly, Phys. Rev. Lett. 58, 1499 (1987).
            F. Courvoisier, et al. "Surface nanoprocesnp.sing with nondiffracting femtosecond Bessel beams" Optics Letters Vol. 34, No. 20 3163 (2009)
        """

        k = 2 * np.pi / self.wavelength
        x0, y0 = r0
        R = np.sqrt((self.X - x0)**2 + (self.Y - y0)**2)
        beta = k * np.cos(alpha)

        if n == 0:
            jbessel = j0(k * np.sin(alpha) * R)
        elif n == 1:
            jbessel = j1(k * np.sin(alpha) * R)
        else:
            jbessel = jv(n, k * np.sin(alpha) * R)

        self.u = A * jbessel * np.exp(1j * beta * z0) * np.exp(
            1.j * k *
            (self.X * np.sin(theta) * np.cos(phi) + self.Y * np.sin(theta) * np.sin(phi)) +
            z0 * np.cos(theta))


    @check_none('X', 'Y', 'u', raise_exception=bool_raise_exception)
    def plane_waves_dict(self, params: dict):
        """Several plane waves with parameters defined in dictionary

        Args:
            params: list with a dictionary:
                A (float): maximum amplitude
                theta (float): angle in radians
                phi (float): angle in radians
                z0 (float): constant value for phase shift
        """
        # Definicion del vector de onda
        k = 2 * np.pi / self.wavelength

        self.u = np.zeros_like(self.u, dtype=complex)
        for p in params:
            self.u = self.u + p['A'] * np.exp(
                1.j * k *
                (self.X * np.sin(p['theta']) * np.cos(p['phi']) + self.Y *
                 np.sin(p['theta']) * np.sin(p['phi']) + p['z0'] * np.cos(p['theta'])))


    @check_none('X', 'Y', 'u', raise_exception=bool_raise_exception)
    def plane_waves_several_inclined(self, A: float, num_beams: tuple[int, int],
                                     max_angle: tuple[float, float], z0: float = 0):
        """Several paralel plane waves.

        Args:
                A (float): maximum amplitude
                num_beams (int, int): number of beams in the x and y directions
                max_angle (float, float): maximum angle of the beams
                z0 (float): position of the beams
        """

        num_beams_x, num_beams_y = num_beams
        max_angle_x, max_angle_y = max_angle

        t = np.zeros_like(self.u, dtype=complex)
        anglex = max_angle_x / num_beams_x
        angley = max_angle_y / num_beams_y

        for i in range(num_beams_x):
            for j in range(num_beams_y):
                theta = np.pi/2 - max_angle_x/2 + anglex * (i + 0.5)
                phi = np.pi/2 - max_angle_y/2 + angley * (j + 0.5)
                self.plane_wave(A, theta, phi, z0)
                t = t + self.u

        self.u = t


    @check_none('X', 'Y', 'u', raise_exception=bool_raise_exception)
    def gauss_beams_several_parallel(self,
                                     r0: tuple[float, float],
                                     A: float,
                                     num_beams: tuple[int, int],
                                     w0: tuple[float, float] | float,
                                     z0: tuple[float, float] | float,
                                     r_range: tuple[float, float],
                                     theta: float = 0.,
                                     phi: float = 0.):
        """Several parallel gauss beams

        Args:
            A (float): maximum amplitude
            num_beams (int, int): number of gaussian beams (equidistintan) in x and y direction.
            w0 (float): beam width of the bemas
            z0 (float): constant value for phase shift
            r0 (float, float): central position of rays (x_c, y_c)
            r_range (float, float): range of rays x, y
            theta (float): angle
            phi (float): angle
        """

        x_range, y_range = r_range
        x_central, y_central = r0
        num_beams_x, num_beams_y = num_beams

        dist_x = x_range / num_beams_x
        dist_y = y_range / num_beams_y

        t = np.zeros_like(self.u, dtype=complex)
        for i in range(num_beams_x):
            xi = x_central - x_range/2 + dist_x * (i + 0.5)
            for j in range(num_beams_y):
                yi = y_central - y_range/2 + dist_y * (j + 0.5)
                self.gauss_beam(r0=(xi, yi),
                                w0=w0,
                                z0=z0,
                                A=A,
                                theta=theta,
                                phi=phi)
                t = t + self.u

        self.u = t


    @check_none('u', raise_exception=bool_raise_exception)
    def gauss_beams_several_inclined(self, A: float, num_beams, w0: tuple[float, float] | float, r0: tuple[float, float], z0: tuple[float, float] | float, max_angle: tuple[float, float]):
        """Several inclined gauss beams

        Args:
            A (float): maximum amplitude
            num_beams (int, int): number of gaussian beams (equidistintan) in x and y direction.
            w0 (float): beam width
            r0 (float, float): central position of rays (x_c, y_c)
            z0 (float): constant value for phase shift
            max_angle (float, float): maximum angles
        """

        num_beams_x, num_beams_y = num_beams
        max_angle_x, max_angle_y = max_angle
        angle_x = max_angle_x / num_beams_x
        angle_y = max_angle_y / num_beams_y

        t = np.zeros_like(self.u, dtype=complex)
        for i in range(num_beams_x):
            thetai = np.pi/2 - max_angle_x/2 + angle_x * (i + 0.5)
            for j in range(num_beams_y):
                phii = np.pi/2 - max_angle_y/2 + angle_y * (j + 0.5)
                self.gauss_beam(r0=r0,
                                w0=w0,
                                z0=z0,
                                A=A,
                                theta=thetai,
                                phi=phii)
                t = t + self.u
