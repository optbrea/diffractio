# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module is related to FOCO project: development of Extended of Focus (EDOF) lenses


Articles and books:

*  K Uno  and I. Shimizu "Dual Focus Diﬀractive Optical Element with Extended Depth of Focus" * OPTICAL REVIEW Vol. 21, No. 5 (2014) 668–675
* A. Sabathyan, M. Golbandi "Petal-like zone plate: long depth bifocal diffractive lens and star-like beam generator" Journal of the Optical Society of America A, 35(7) 1243 (20018)
"""

from numpy import exp, pi, sqrt

from . import degrees, mm, np, plt, sp
from .scalar_fields_XY import Scalar_field_XY
from .scalar_fields_XYZ import Scalar_field_XYZ
from .scalar_masks_XY import Scalar_mask_XY
from .utils_math import binarize as func_binarize
from .utils_math import cart2pol
from .utils_optics import beam_width_2D


class Angular_lens(Scalar_field_XY):
    """Generates examples for star lenses"""

    def __init__(self, x=None, y=None, wavelength=None, info=""):
        """equal than Scalar_field_XY"""
        super(self.__class__, self).__init__(x, y, wavelength, info)

    def angular_general(self,
                        r0,
                        radius,
                        g,
                        f_ini,
                        f_end,
                        num_periods=2,
                        binarize=False,
                        angle=0 * degrees,
                        mask=True):
        """Transparent lens, from f_ini to f_end
           focal=f_ini+(f_end-f_ini)*theta/(2*sp.pi)

        Parameters:
            r0 (float, float): (x0,y0) - center of lens
            radius (float, float): radius of lens mask
            g (function): periodic function  2*pi in range (-1,1)
            f_ini (float): focal length of lens
            f_end (float): focal length of lens
            num_periods (int): num loops in a lens
            angle (float): angle of axis in radians
            mask (bool): if True, mask with size radius

        Example:
            lens(r0=(0 * um, 0 * um), radius=(100 * um, 200 * um), focal=(5 * mm, 10 * mm), angle=0 * degrees, mask=True)
        """
        # si solamente un numero, posiciones y radius son los mismos para ambos
        # Definicion del origen, el radius y la focal

        # Vector de onda
        k = 2 * pi / self.wavelength

        x0, y0 = r0
        # rotation de la lens
        Xrot, Yrot = self.__rotate__(angle)

        # r = sqrt((Xrot - x0)**2 + (Yrot - y0)**2)
        theta = sp.arctan2((Yrot - y0), (Xrot - x0))

        # Definicion de la amplitude y la phase
        if mask is True:
            amplitude = Scalar_mask_XY(self.x, self.y, self.wavelength)
            amplitude.circle(r0, radius, angle)
            t = amplitude.u
        else:
            t = np.ones_like(self.X)

        f_mean = (f_end + f_ini) / 2
        f_incr = (f_end - f_ini)
        F = f_mean + f_incr / 2 * g(theta * num_periods)

        self.u = t * exp(-1.j * k * ((Xrot**2 + Yrot**2) / (2 * F)))
        self.u[t == 0] = 0

        if binarize is True:
            self.u = func_binarize(self.u, -np.pi, np.pi)

        return self.u

    def daisy_lens(self,
                   r0,
                   radius,
                   f_ini,
                   f_end,
                   num_periods=2,
                   binarize=False,
                   angle=0 * degrees,
                   mask=True):
        """Transparent lens, from f_ini to f_end
           focal=f_ini+(f_end-f_ini)*theta/(2*sp.pi)

        Parameters:
            r0 (float, float): (x0,y0) - center of lens
            radius (float, float): radius of lens mask
            f_ini (float): focal length of lens
            f_end (float): focal length of lens
            num_periods (int): num loops in a lens
            angle (float): angle of axis in radians
            mask (bool): if True, mask with size radius

        Example:
            lens(r0=(0 * um, 0 * um), radius=(100 * um, 200 * um), focal=(5 * mm, 10 * mm), angle=0 * degrees, mask=True)
        """
        # si solamente un numero, posiciones y radius son los mismos para ambos
        # Definicion del origen, el radius y la focal

        # Vector de onda
        k = 2 * pi / self.wavelength

        x0, y0 = r0
        # rotation de la lens
        Xrot, Yrot = self.__rotate__(angle)

        # r = sqrt((Xrot - x0)**2 + (Yrot - y0)**2)
        theta = sp.arctan2((Yrot - y0), (Xrot - x0))

        # Definicion de la amplitude y la phase
        if mask is True:
            amplitude = Scalar_mask_XY(self.x, self.y, self.wavelength)
            amplitude.circle(r0, radius, angle)
            t = amplitude.u
        else:
            t = np.ones_like(self.X)

        f_mean = (f_end + f_ini) / 2
        f_incr = (f_end - f_ini)
        F = f_mean + f_incr / 2 * sp.sin(theta * num_periods)

        self.u = t * exp(-1.j * k * ((Xrot**2 + Yrot**2) / (2 * F)))
        self.u[t == 0] = 0

        if binarize is True:
            self.phase2amplitude(matrix=False, new_field=False)
            self.binarize(kind='amplitude', new_field=False)

        return self.u

    def lotus_lens(self,
                   r0,
                   radius,
                   f_ini,
                   f_end,
                   num_periods=2,
                   binarize=False,
                   angle=0 * degrees,
                   mask=True):

        self.angular_general(r0, radius, triangle_periodic, f_ini, f_end,
                             num_periods, binarize, angle, mask)

    def dual_focus_lens(self,
                        r0,
                        radius,
                        f_ini,
                        f_end,
                        num_periods=2,
                        binarize=False,
                        angle=0 * degrees,
                        mask=True):

        k = 2 * pi / self.wavelength

        x0, y0 = r0
        # rotation de la lens
        Xrot, Yrot = self.__rotate__(angle)

        # r = sqrt((Xrot - x0)**2 + (Yrot - y0)**2)
        theta = sp.arctan2((Yrot - y0), (Xrot - x0))

        # Definicion de la amplitude y la phase
        if mask is True:
            amplitude = Scalar_mask_XY(self.x, self.y, self.wavelength)
            amplitude.circle(r0, radius, angle)
            t = amplitude.u
        else:
            t = np.ones_like(self.X)

        f_mean = (f_end + f_ini) / 2
        f_incr = (f_end - f_ini)
        F = f_mean + f_incr / 2 * np.sign(sp.sin(theta * num_periods))

        self.u = t * exp(-1.j * k * ((Xrot**2 + Yrot**2) / (2 * F)))
        self.u[t == 0] = 0

        if binarize is True:
            self.phase2amplitude(matrix=False, new_field=False)
            self.binarize(kind='amplitude', new_field=False)

        return self.u

    def axilens(self,
                r0,
                radius,
                f_mean,
                f_incr,
                binarize=False,
                angle=0 * degrees,
                mask=True):

        k = 2 * pi / self.wavelength

        x0, y0 = r0
        # rotation de la lens
        Xrot, Yrot = self.__rotate__(angle)

        r = sqrt((Xrot - x0)**2 + (Yrot - y0)**2)
        # theta = sp.arctan2((Yrot - y0), (Xrot - x0))

        # Definicion de la amplitude y la phase
        if mask is True:
            amplitude = Scalar_mask_XY(self.x, self.y, self.wavelength)
            amplitude.circle(r0, radius, angle)
            t = amplitude.u
        else:
            t = np.ones_like(self.X)

        F = f_mean + f_incr / 2 * (r / radius)**2

        self.u = t * exp(-1.j * k * ((Xrot**2 + Yrot**2) / (2 * F)))
        self.u[t == 0] = 0

        if binarize is True:
            self.phase2amplitude(matrix=False, new_field=False)
            self.binarize(kind='amplitude', new_field=False)

        return self.u

    def petal_lens(self,
                   r0,
                   radius,
                   focal,
                   N,
                   alpha,
                   binarize=False,
                   angle=0 * degrees,
                   mask=True):
        """Lens according to Sabatyan and Golbandi

        References:
            A. Sabathyan, M. Golbandi "Petal-like zone plate: long depth bifocal diffractive lens and star-like beam generator" Journal of the Optical Society of America A, 35(7) 1243 (20018)


        Parameters:
            r0 (float, float): (x0,y0) - center of lens
            radius (float, float): radius of lens mask
            focal (float): focal length
            N (float): petal frequency
            alpha (float): focal length of lens
            angle (float): angle of axis in radians
            mask (bool): if True, mask with size radius

        Example:
            lens(r0=(0 * um, 0 * um), radius=(100 * um, 200 * um), focal=(5 * mm, 10 * mm), angle=0 * degrees, mask=True)
        """

        R = radius
        f = focal

        k = 2 * pi / self.wavelength

        x0, y0 = r0
        Xrot, Yrot = self.__rotate__(angle)

        r = sqrt((Xrot - x0)**2 + (Yrot - y0)**2)
        theta = sp.arctan2((Yrot - y0), (Xrot - x0))

        # Definicion de la amplitude y la phase
        if mask is True:
            amplitude = Scalar_mask_XY(self.x, self.y, self.wavelength)
            amplitude.circle(r0, radius, angle)
            t = amplitude.u
        else:
            t = sp.ones_like(self.X)

        self.u = t * np.exp(-1j * pi * (r - alpha * R * np.cos(N * theta))**2 /
                            (self.wavelength * f))
        self.u[t == 0] = 0

        if binarize is True:
            self.phase2amplitude(matrix=False, new_field=False)
            self.binarize(kind='amplitude', new_field=False)

        return self.u

    def trifocus_lens(self,
                      r0,
                      radius,
                      f_mean,
                      f_incr,
                      num_periods,
                      binarize=False,
                      angle=0 * degrees,
                      power=3,
                      mask=True):
        """Lens with 3 focuses, defined as sin(N theta)**3

        References:
            Luis Miguel Sanchez Brea

        Parameters:
            r0 (float, float): (x0,y0) - center of lens
            radius (float, float): radius of lens mask
            f_mean (float): focal length
            f_incr (float): incr focal
            binarize (bool): binarizes
            num_periods (int): number of petals
            angle (float): angle of axis in radians
            power (int): odd number
            mask (bool): if True, mask with size radius

        Example:
            lens(r0=(0 * um, 0 * um), radius=(100 * um, 200 * um), focal=(5 * mm, 10 * mm), angle=0 * degrees, mask=True)
        """

        k = 2 * pi / self.wavelength

        x0, y0 = r0
        # rotation de la lens
        Xrot, Yrot = self.__rotate__(angle)

        # r = sqrt((Xrot - x0)**2 + (Yrot - y0)**2)
        theta = sp.arctan2((Yrot - y0), (Xrot - x0))

        # Definicion de la amplitude y la phase
        if mask is True:
            amplitude = Scalar_mask_XY(self.x, self.y, self.wavelength)
            amplitude.circle(r0, radius, angle)
            t = amplitude.u
        else:
            t = np.ones_like(self.X)

        F = f_mean + f_incr / 2 * sp.sin(theta * num_periods)**power

        self.u = t * exp(-1.j * k * ((Xrot**2 + Yrot**2) / (2 * F)))
        self.u[t == 0] = 0

        if binarize is True:
            self.phase2amplitude(matrix=False, new_field=False)
            self.binarize(kind='amplitude', new_field=False)

        return self.u

    # def propagate(self):
    #     uxyz = Scalar_field_XYZ(fieldinicial=u_xy, z=z0)
    #     uxyz.RS(num_processors=num_processors, verbose=verbose)

    def dartboard_lens(self,
                       r0,
                       diameter,
                       focal_min,
                       focal_max,
                       num_focals=4,
                       num_sectors=4,
                       has_random_focals=True,
                       angle=0 * degrees):
        """
        """
        focals = np.linspace(focal_min, focal_max, num_focals)
        sectores_aleatorios = np.random.permutation(num_focals)

        t_final = Scalar_mask_XY(
            x=self.x, y=self.y, wavelength=self.wavelength)
        tf = Scalar_mask_XY(x=self.x, y=self.y, wavelength=self.wavelength)
        mask = Scalar_mask_XY(x=self.x, y=self.y, wavelength=self.wavelength)

        [rho, theta] = cart2pol(t_final.X - r0[0], t_final.Y - r0[1])

        theta = theta + np.pi

        delta_angle = 2 * pi / (num_focals * num_sectors)
        delta_f = 2 * pi / (num_sectors)

        if has_random_focals is True:
            fs_ = focals[sectores_aleatorios]
        else:
            fs_ = focals

        for i_focal, focal in enumerate(fs_):
            mask = Scalar_mask_XY(
                x=self.x, y=self.y, wavelength=self.wavelength)

            tf.fresnel_lens(
                r0=r0,
                radius=diameter / 2,
                focal=focal,
                kind='amplitude',
                phase=0,
                angle=angle,
                mask=True)

            for i_sectors in range(num_sectors):
                ang_0 = i_focal * delta_angle + i_sectors * delta_f
                ang_1 = i_focal * delta_angle + i_sectors * delta_f + delta_angle

                ix = (theta > ang_0) & (theta <= ang_1)
                mask.u[ix] = 1

            t_final.u = t_final.u + mask.u * tf.u
        self.u = t_final.u

    def compute_beam_width(self,
                           u_xy,
                           z0,
                           num_processors=16,
                           has_draw=True,
                           verbose=False):
        """computes beam width for scalar_field_XY. Uses Rayleigh-Sommerfeld Approach

        Parameters:
            u_xy (scalar_field_XY): field (u0*t) at a z=0 plane
            z0 (numpy_array): position z0
            num_processors (int): num processors for computation
            has_draw (bool): if True draws
            verbose (bool): if True returns data
        """

        widths = sp.zeros_like(z0, dtype=float)

        X, Y = u_xy.X, u_xy.Y
        for iz, zi in enumerate(z0):
            uz = sp.squeeze(self.u[:, :, iz])
            dx, dy, principal_axis, moments = beam_width_2D(uz, X, Y)
            print("{:2.2f} ".format(dx))
            widths[iz] = dx

        if has_draw is True:
            plt.figure(figsize=(12, 6))
            plt.plot(z0 / mm, widths / 2, 'k')
            plt.plot(z0 / mm, -widths / 2, 'k')

        return widths

    def get_DOF(self, z, num_processors=1, verbose=False, has_draw=False):
        """Get depth of focus of the lens in a certain interval z measuring the beam width.
        It also determines the field at positions z

        Parameters:
            z (numpy.array): positions z where focus is placed
            num_procesors (int): number of processors for computation.
            verbose (bool): if True, prints interesting information about the computation.
            has_draw (bool): if True, draws the depth of focus.

        Returns:
            width (numpy.array): beam width at positions given at z.
            u_xyz (Scalar_field_XYZ): fields at positions z.
        """

        u_xyz = Scalar_field_XYZ(self.x, self.y, z, self.wavelength)
        u_xyz.incident_field(self)
        u_xyz.RS(verbose, num_processors)

        width, _, _ = u_xyz.beam_widths(False)

        if has_draw is True:
            plt.figure()
            plt.plot(z / mm, width, 'k--', label='lens')
            plt.ylim(bottom=0)

            plt.xlabel("z (mm)")
            plt.ylabel("beam width $\omega$ ($\mu$m)")
            plt.xlim(z[0] / mm, z[-1] / mm)

        return width, u_xyz

    def draw_comparison(self,
                        z0,
                        widths_lens,
                        widths_star,
                        focals,
                        coverture=2,
                        ylims=(-100, 100),
                        filename=''):
        """Draws comparison of daisy_lens with standard lens

        Parameters:
            z0 (numpy.array): positions of z0
            widths_lens (numpy.array): beam width for standard lens
            widths_star (numpy.array): beam width for star lens
            focals (float, float): f0 and f1 focals for definiton of star lens
            coverture (float): coverture in f_incr for drawing
        """
        f0, f1 = focals
        f_mean = (f0 + f1) / 2
        f_incr = (f1 - f0) / 2

        plt.figure(figsize=(12, 6))
        plt.plot(z0 / mm, widths_lens / 2, 'r--', lw=4, label='standard')
        plt.plot(z0 / mm, widths_star / 2, 'k', lw=4, label='star lens')
        plt.plot(z0 / mm, -widths_lens / 2, 'r--', lw=4)
        plt.plot(z0 / mm, -widths_star / 2, 'k', lw=4)
        plt.xlabel("$z (mm)$", fontsize=20)
        plt.ylabel("$\omega (\mu m)$", fontsize=20)
        plt.legend()
        if coverture > 0:
            plt.xlim((f_mean - coverture * f_incr) / mm,
                     (f_mean + coverture * f_incr) / mm)
        plt.ylim(ylims)
        if not filename == '':
            plt.savefig(
                filename, dpi=300, bbox_inches='tight', pad_inches=0.05)


def triangle_periodic(theta):
    # https://en.wikipedia.org/wiki/Triangle_wave
    p = 2 * np.pi
    return 4 * np.abs(theta / p - np.floor(theta / p + 0.5)) - 1