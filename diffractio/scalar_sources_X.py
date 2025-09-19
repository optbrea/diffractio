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
    * wavelets
    * plane_waves_dict
    * plane_waves_several_inclined
    * gauss_beams_several_parallel
    * gauss_beams_several_inclined

*Also*
    * Polychromatic and extendes sources are defined in scalar_fields_X.py for multiprocessing purposes.
"""
# flake8: noqa


from .__init__ import degrees, np, um
from .__init__ import np, plt
from .utils_typing import npt, Any, NDArray,  NDArrayFloat, NDArrayComplex
from .utils_common import check_none
from .utils_optics import roughness_1D
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
    def super_gauss_beam(self, x0: float, w0: float, z0: float,  power: float = 2.,A: float = 1, theta: float = 0.):
        """Supergaus beam. exp(-(|x-x0|/w0)**power).

        Args:
            x0 (float): x position of center
            w0 (float): minimum beam width
            z0 (float): position of beam width
            power (float): power of the super-Gaussian profile
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
        amplitude = A * w0 / w * np.exp(-np.abs((self.x - x0))**power / (np.abs(w)**power))
        phase1 = np.exp(1j * k * ((self.x - x0) * np.sin(theta)))  # rotation
        phase2 = np.exp(1j * (-k * z0 - phaseGouy + k * (self.x - x0)**2 /
                                (2 * R)))

        self.u = amplitude * phase1 * phase2

        return self



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
    def wavelets(self, kind: str, x0s: NDArray, z0s: NDArray, As: NDArray, phases: NDArray = 0, w0: float = 0.): 
        """Generates a beam profile z(x) with wavelets: spherical waves or gaussian beams. Each wavelet is defined by its position (x0, z0), amplitude A, and phase.
        The resulting field is the sum of all wavelets, each contributing a spherical or gaussian wave at its respective position. 
        The phase is useful for generating beams with partially coherence, where each wavelet can have a different phase.
        If the phase is constant, then the resulting field is a coherent superposition of wavelets.

        Args:
            kind (str): 'spherical' or 'gaussian'
            x0s (NDArray): array of x0 positions
            y0s (NDArray): array of y0 positions
            z0s (NDArray): array of z0 positions
            As (NDArray): array of amplitudes at the positions x0s
            phases (NDArray, optional): array of phases at the positions x0s. Defaults to 0.
            w0 (float, optional): beam waist for Gaussian profile. Defaults to 0. If w0=0, then the beam waist is equal to the wavlength.

        Raises:
            ValueError: If kind is not 'spherical' or 'gaussian'.
            ValueError: If x0s, y0s, z0s, and As do not have the same length.  

        Returns:
            Scalar_source_X: Amplitude distribution at the origin for the given light profile.
        """
        if kind not in ['spherical', 'gaussian']:
            raise ValueError("kind must be 'spherical' or 'gaussian'")
        

        
        if isinstance(x0s, (list, tuple)):
            x0s = np.array(x0s)
        if isinstance(z0s, (list, tuple)):
            z0s = np.array(z0s)
        if isinstance(As, (list, tuple)):
            As = np.array(As)
        if isinstance(phases, (list, tuple)):
            phases = np.array(phases)

        if isinstance(phases, (int, float)):
            phases = phases * np.ones_like(x0s)

        if isinstance(As, (int, float)):
            As = As * np.ones_like(x0s)
        
        if isinstance(z0s, (int, float)):
            z0s = z0s * np.ones_like(x0s)

        if isinstance(x0s, (int, float)):
            x0s = x0s * np.ones_like(z0s)

        if len(x0s) != len(z0s) or len(x0s) != len(As) or len(x0s) != len(phases):
            raise ValueError("x0s, z0s, As, and phases must have the same length")

        if w0 == 0:
            w0 = self.wavelength

        # Initialize the field
        self.u = np.zeros_like(self.x, dtype=np.complex128)

        u0 = self.duplicate(clear=True)

        for i, x0, z0, A in zip(range(len(x0s)), x0s, z0s, As):
            if kind == 'spherical':
                u0.spherical_wave(A=A, x0=x0, z0=z0)
            elif kind == 'gaussian':
                u0.gauss_beam(A=A, x0=x0, z0=z0, w0=w0)

            # Add the contribution of this source to the total field
            self.u = self.u + u0.u * np.exp(1j * phases[i])

    @check_none('x', 'u', raise_exception=bool_raise_exception)
    def partial_coherence(self, lc: float, s: float, has_draw: bool = True, verbose: bool = True) -> None:
        """partial_coherence. Apply partial coherence to the optical field. It generates a random phase to the field based on Gaussian distribution with a given correlation length and standard deviation.

        This method modifies the `u` attribute of the instance by multiplying it with a roughness field generated by`roughness_2D`.

        Args:
            lc (float, float) | float: correlation length in micrometers
            s (float): standard deviation in microns, for example $\lambda$/2.
            has_draw (bool, optional): Whether to draw the result. Defaults to True.
            verbose (bool, optional): Prints statistics about the roughness. Defaults to True.

        TODO: change roughness parameters for degree of coherence.

        References:
            JA Oglivy "Theory of wave scattering from random surfaces" Adam Hilger p.224.
        """

        k = 2 * np.pi / self.wavelength

        t_rough=roughness_1D(x=self.x, t=lc, s=s)

        self.u *= np.exp(1j * k * t_rough)  # Apply roughness to the source.

        if has_draw:
            plt.figure()
            plt.plot(self.x, t_rough/(2*np.pi))
            plt.xlabel('x (um)')
            plt.ylabel(r'Roughness (radians/2$\pi$)')

            plt.figure()
            plt.hist((t_rough/(2*np.pi)).flatten(), bins=100)

        if verbose:
            print(r'Mean roughness    : {:.3f} loops'.format(np.mean(t_rough/(2*np.pi))))
            print(r'Standard deviation: {:.3f} loops'.format( np.std(t_rough/(2*np.pi))))


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
