# !/usr/bin/env python3
# ----------------------------------------------------------------------
# Name:        vector_masks_XY.py
# Purpose:     Defines the Vector_mask_XY class for vector masks operations
#
# Author:      Luis Miguel Sanchez Brea
#
# Created:     2024
# Licence:     GPLv3
# ----------------------------------------------------------------------


# flake8: noqa

"""
This module generates Vector_mask_XY class for defining vector masks. Its parent is Vector_field_XY.
The main atributes are:
        self.x (numpy.array): linear array with equidistant positions. The number of data is preferibly 2**n.
        self.y (numpy.array): linear array with equidistant positions. The number of data is preferibly 2**n.
        self.wavelength (float): wavelength of the incident field.
        self.Ex (numpy.array): Electric_x field
        self.Ey (numpy.array): Electric_y field


*Class for bidimensional vector XY masks*

*Functions*
    * unique_masks
    * equal_masks
    * global_mask
    * complementary_masks
    * from_py_pol
    * polarizer_linear
    * quarter_waveplate
    * half_wave
    * polarizer_retarder
"""
import copy
from typing import Literal

from matplotlib import rcParams
from py_pol.jones_matrix import Jones_matrix


from .__init__ import degrees, np, plt
from .config import bool_raise_exception, CONF_DRAWING, number_types
from .utils_typing import npt, Any, NDArray,  NDArrayFloat, NDArrayComplex
from .utils_common import check_none
from .scalar_masks_XY import Scalar_mask_XY
from .utils_optics import field_parameters
from .vector_fields_XY import Vector_field_XY
from .vector_sources_XY import Vector_source_XY


Draw_Options = Literal['amplitudes', 'phases', 'jones', 'jones_ap']

class Vector_mask_XY(Vector_field_XY):

    def __init__(self, x: NDArrayFloat | None = None, y: NDArrayFloat | None = None,
                 wavelength: float | None = None,  n_background: float = 1, info: str = ""):
        super().__init__(x, y, wavelength, n_background, info)
        self.type = 'Vector_mask_XY'

        self.M00 = np.zeros_like(self.X, dtype=complex)
        self.M01 = np.zeros_like(self.X, dtype=complex)
        self.M10 = np.zeros_like(self.X, dtype=complex)
        self.M11 = np.zeros_like(self.X, dtype=complex)



        del self.Ex, self.Ey, self.Ez


    def __str__(self):
        """Represents data from class."""


        print("{}\n - x:  {},   y:  {},   M00:  {}".format(
            self.type, self.x.shape, self.y.shape, self.M00.shape))
        print(
            " - xmin:       {:2.2f} um,  xmax:      {:2.2f} um,  Dx:   {:2.2f} um"
            .format(self.x[0], self.x[-1], self.x[1] - self.x[0]))
        print(
            " - ymin:       {:2.2f} um,  ymay:      {:2.2f} um,  Dy:   {:2.2f} um"
            .format(self.y[0], self.y[-1], self.y[1] - self.y[0]))

        print(" - wavelength: {:2.2f} um".format(self.wavelength))
        print(" - date:       {}".format(self.date))
        if self.info != "":
            print(" - info:       {}".format(self.info))
        return ""
    

    @check_none('x', 'y')
    def __add__(self, other, kind: str = 'standard'):
        """adds two Vector_mask_XY. For example two  masks

        Args:
            other (Vector_mask_XY): 2nd field to add
            kind (str): instruction how to add the fields:

        Returns:
            Vector_mask_XY: `M3 = M1 + M2`
        """

        if other.type in ('Vector_mask_XY'):
            m3 = Vector_mask_XY(self.x, self.y, self.wavelength)

            m3.M00 = other.M00 + self.M00
            m3.M01 = other.M01 + self.M01
            m3.M10 = other.M10 + self.M10
            m3.M11 = other.M11 + self.M11

        return m3

    @check_none('x', 'y')
    def __mul__(self, other):
        """
        Multilies the Vector_mask_XY matrix by another Vector_mask_XY.

        Args:
            other (Vector_mask_XY): 2nd object to multiply.

        Returns:
            v_mask_XY (Vector_mask_XY): Result.
        """

        if isinstance(other, number_types):
            m3 = Vector_mask_XY(self.x, self.y, self.wavelength)
            m3.M00 = self.M00 * other
            m3.M01 = self.M01 * other
            m3.M10 = self.M10 * other
            m3.M11 = self.M11 * other

        elif other.type in ('Vector_mask_XY', 'Vector_field_XY'):
            m3 = Vector_mask_XY(self.x, self.y, self.wavelength)

            m3.M00 = other.M00 * self.M00 + other.M01 * self.M10
            m3.M01 = other.M00 * self.M01 + other.M01 * self.M11
            m3.M10 = other.M10 * self.M00 + other.M11 * self.M10
            m3.M11 = other.M10 * self.M01 + other.M11 * self.M11

        else:
            raise ValueError('other thype ({}) is not correct'.format(
                type(other)))

        return m3

    @check_none('x', 'y')
    def __rmul__(self, other):
        """
        Multilies the Vector_mask_XY matrix by another Vector_mask_XY.

        Args:
            other (Vector_mask_XY): 2nd object to multiply.

        Returns:
            v_mask_XY (Vector_mask_XY): Result.
        """
        if isinstance(other, number_types):
            m3 = Vector_mask_XY(self.x, self.y, self.wavelength)
            m3.M00 = self.M00 * other
            m3.M01 = self.M01 * other
            m3.M10 = self.M10 * other
            m3.M11 = self.M11 * other
            # print("numero * matriz")

        elif other.type in ('Vector_source_XY', 'Vector_field_XY'):
            m3 = Vector_source_XY(self.x, self.y, self.wavelength)
            m3.Ex = self.M00 * other.Ex + self.M01 * other.Ey
            m3.Ey = self.M10 * other.Ex + self.M11 * other.Ey

        return m3


    def duplicate(self, clear: bool = False):
        """Duplicates the instance"""
        new_field = copy.deepcopy(self)
        if clear is True:
            new_field.clear_field()
        return new_field


    @check_none('x', 'y')
    def apply_circle(self, r0: tuple[float, float] | None = None,
                     radius: tuple[float, float] | None = None):
        """The same circular mask is applied to all the Jones Matrix.

        Args:
            r0 (float, float): center, if None it is generated
            radius (float, float): radius, if None it is generated
        """
        if radius is None:
            x_min, x_max = self.x[0], self.x[-1]
            y_min, y_max = self.y[0], self.y[-1]

            x_radius, y_radius = (x_max - x_min)/2, (y_max - y_min)/2

            radius = (x_radius, y_radius)

        if r0 is None:
            x_center, y_center = (x_min + x_max)/2, (y_min + y_max)/2
            r0 = (x_center, y_center)

        u_pupil = Scalar_mask_XY(self.x, self.y, self.wavelength)
        u_pupil.circle(r0=r0, radius=radius)

        self.M00 = self.M00 * u_pupil.u
        self.M01 = self.M01 * u_pupil.u
        self.M10 = self.M10 * u_pupil.u
        self.M11 = self.M11 * u_pupil.u


    @check_none('x', 'y')
    def pupil(self, r0: tuple[float, float] | None = None,
              radius: tuple[float, float] | None = None, angle: float = 0*degrees):
        """place a pupil in the mask. If r0 or radius are None, they are computed using the x,y parameters.

        Args:
            r0 (float, float): center of circle/ellipse
            radius (float, float) or (float): radius of circle/ellipse
            angle (float): angle of rotation in radians

        Example:

            pupil(r0=(0*um, 0*um), radius=(250*um, 125*um), angle=0*degrees)
        """

        if r0 is None:
            x0 = (self.x[-1] + self.x[0])/2
            y0 = (self.y[-1] + self.y[0])/2
            r0 = (x0, y0)

        if radius is None:
            radiusx = (self.x[-1] - self.x[0])/2
            radiusy = (self.y[-1] - self.y[0])/2
            radius = (radiusx, radiusy)

        x0, y0 = r0

        if isinstance(radius, (float, int, complex)):
            radiusx, radiusy = (radius, radius)
        else:
            radiusx, radiusy = radius

        Xrot, Yrot = self.__rotate__(angle, (x0, y0))

        pupil0 = np.zeros(np.shape(self.X))
        ipasa = (Xrot)**2 / (radiusx + 1e-15)**2 + \
            (Yrot)**2 / (radiusy**2 + 1e-15) < 1
        pupil0[ipasa] = 1
        self.M00 = self.M00 * pupil0
        self.M01 = self.M01 * pupil0
        self.M10 = self.M10 * pupil0
        self.M11 = self.M11 * pupil0


    def scalar_to_vector_mask(self, mask: Scalar_mask_XY, pol_state:  None |Jones_matrix = None, is_intensity: bool = True):
        """The same mask (binary) is applied to all the Jones Matrix.

        Args:
            mask (scalar_mask_XY): mask to apply.
            pol_state (Jones Matrix or Jones_matrix): Polarization state to apply
            is intensity (bool): If True, abs(mask)**2 is applied.
        """

        if pol_state is None:
            pol_state = Jones_matrix().vacuum()

        if isinstance(pol_state, Jones_matrix):
            pol_state = pol_state.M.squeeze()

        t = mask.u      

        self.M00 = pol_state[0, 0] * t
        self.M01 = pol_state[0, 1] * t
        self.M10 = pol_state[1, 0] * t
        self.M11 = pol_state[1, 1] * t

        # Change elements = -0 to 0, to represent correctly phases.
        self.M01.real = np.where(np.real(self.M01) == -0, 0, np.real(self.M01))
        self.M10.real = np.where(np.real(self.M10) == -0, 0, np.real(self.M10))
        self.M00.real = np.where(np.real(self.M00) == -0, 0, np.real(self.M00))
        self.M11.real = np.where(np.real(self.M11) == -0, 0, np.real(self.M11))


    @check_none('x', 'y')
    def complementary_masks(self, mask: Scalar_mask_XY, pol_state_0: Jones_matrix, pol_state_1: Jones_matrix,           is_binarized: bool=True):
        """Creates a vector mask from a scalar mask. It assign an pol_state_0 to 0 values and a pol_state_1 to 1 values..
        For generality, ik mask is a decimal number between 0 and 1, it takes the linear interpolation.

        Args:
            mask (scalar_mask_XY): Mask preferently binary. if not, it is binarized
            pol_state_0 (2x2 numpy.array or Jones_matrix): Jones matrix for 0s.
            pol_state_1 (2x2 numpy.array or Jones_matrix): Jones matrix for 1s.

        Warning:
            TODO: Mask should be binary. Else the function should binarize it.
        """

        if isinstance(pol_state_0, Jones_matrix):
            pol_state_0 = pol_state_0.M.squeeze()
        if isinstance(pol_state_1, Jones_matrix):
            pol_state_1 = pol_state_1.M.squeeze()

        t = np.abs(mask.u)**2
        if is_binarized:
            t = t / t.max()
            t[t < 0.5] = 0
            t[t >= 0.5] = 1

        self.M00 = t * pol_state_1[0, 0] + (1 - t) * pol_state_0[0, 0]
        self.M01 = t * pol_state_1[0, 1] + (1 - t) * pol_state_0[0, 1]
        self.M10 = t * pol_state_1[1, 0] + (1 - t) * pol_state_0[1, 0]
        self.M11 = t * pol_state_1[1, 1] + (1 - t) * pol_state_0[1, 1]


    def multilevel_mask(self, mask: Scalar_mask_XY, states: Jones_matrix, discretize: bool=True, normalize: bool=True):
        """Generates a multilevel vector mask, based in a scalar_mask_XY. The levels should be integers in amplitude (0,1,..., N).
            If it is not like this, discretize generates N levels.
            Usually masks are 0-1. Then normalize generates levels 0-N.

            Args:
                mask (scalar_mask_XY): 0-N discrete scalar mask.
                states (np.array or Jones_matrix): Jones matrices to assign to each level
                discretize (bool): If True, a continuous mask is converted to N levels.
                normalize (bool): If True, levels are 0,1,.., N.

        """
        mask_new = mask.duplicate()

        num_levels = len(states)

        if discretize is True:
            mask_new.discretize(num_levels=num_levels, new_field=False)

        if normalize is True:
            mask_new.u = mask_new.u / mask_new.u.max()
            mask_new.u = mask_new.u * num_levels - 0.5

        mask_new.u = np.real(mask_new.u)
        mask_new.u = mask_new.u.astype(int)

        for i, pol_state in enumerate(states):
            i_level = (mask_new.u == i)

            self.M00[i_level] = pol_state.M[0, 0, 0]
            self.M01[i_level] = pol_state.M[0, 1, 0]
            self.M11[i_level] = pol_state.M[1, 1, 0]
            self.M10[i_level] = pol_state.M[1, 0, 0]


    def from_py_pol(self, polarizer: Jones_matrix):
        """Generates a constant polarization mask from py_pol Jones_matrix.
        This is the most general function to obtain a polarizer.

        Args:
            polarizer (2x2 numpy.matrix): Jones_matrix
        """

        if isinstance(polarizer, Jones_matrix):
            M = polarizer.M
        else:
            M = polarizer

        uno = np.ones_like(self.X, dtype=complex)
        M = np.asarray(M)
        
        self.M00 = uno * M[0, 0]
        self.M01 = uno * M[0, 1]
        self.M10 = uno * M[1, 0]
        self.M11 = uno * M[1, 1]


    def vacuum(self):
        """No polarizing structure.

        Args:
        """
        PL = Jones_matrix('vacuum')
        PL.vacuum()
        self.from_py_pol(PL)


    def polarizer_linear(self, azimuth: float=0*degrees):
        """Generates an XY linear polarizer.

        Args:
            angle (float): linear polarizer angle
        """
        PL = Jones_matrix('m0')
        PL.diattenuator_perfect(azimuth=azimuth)
        self.from_py_pol(PL)


    def quarter_waveplate(self, azimuth: float=0*degrees):
        """Generates an XY quarter wave plate.

        Args:
            azimuth (float): polarizer angle
        """
        PL = Jones_matrix('m0')
        PL.quarter_waveplate(azimuth=azimuth)
        self.from_py_pol(PL)


    def half_waveplate(self, azimuth:float=0*degrees):
        """Generates an XY half wave plate.

        Args:
            azimuth (float): polarizer angle
        """
        PL = Jones_matrix('m0')
        PL.half_waveplate(azimuth=azimuth)
        self.from_py_pol(PL)

    def polarizer_retarder(self, R: float=0*degrees, p1: float=1, p2: float=1, azimuth: float=0*degrees):
        """Generates an XY retarder.

        Args:
            R (float): retardance between Ex and Ey components.
            p1 (float): transmittance of fast axis.
            p2 (float): transmittance of slow axis.
            azimuth (float): linear polarizer angle
        """
        PL = Jones_matrix('m0')
        PL.diattenuator_retarder_linear(R=R, p1=p1, p2=p2, azimuth=azimuth)
        self.from_py_pol(PL)


    def radial_polarizer(self, r0: tuple[float, float]=(0.,0.)):
        """Radial polarizer.

        Args:
            r0 (tuple[float, float]): position of center
         """

        x0, y0 = r0

        R = np.sqrt((self.X-x0)**2 + (self.Y-y0)**2)
        THETA = np.arctan2((self.Y-y0),(self.X-x0))

        
        self.M00 = np.cos(THETA)**2
        self.M01 = np.cos(THETA)*np.sin(THETA)
        self.M10 = np.cos(THETA)*np.sin(THETA)
        self.M11 = np.sin(THETA)**2


    def azimuthal_polarizer(self, r0: tuple[float, float]=(0.,0.)):
        """Generates an azimuthal-polarizer

        Args:
            r0 (tuple[float, float]): position of center
        """

        x0, y0 = r0

        R = np.sqrt((self.X-x0)**2 + (self.Y-y0)**2)
        THETA = np.arctan2((self.Y-y0),(self.X-x0))

        
        self.M00 = np.sin(THETA)**2
        self.M01 = -np.cos(THETA)*np.sin(THETA)
        self.M10 = -np.cos(THETA)*np.sin(THETA)
        self.M11 = np.cos(THETA)**2



    def RCP(self):
        """Right circular polarizer
        """

        ones = np.ones_like(self.X)

        self.M00 = 0.5*ones
        self.M01 = 0.5j*ones
        self.M10 = -0.5j*ones
        self.M11 = 0.5j*ones


    def LCP(self):
        """Left circular polarizer.
        """

        ones = np.ones_like(self.X)

        self.M00 = 0.5*ones
        self.M01 = -0.5j*ones
        self.M10 = 0.5j*ones
        self.M11 = 0.5j*ones



    def RCP2LCP(self):
        """Rght circular polarizer to Left circular polarizer
        """

        ones = np.ones_like(self.X)

        self.M00 = 0.5*ones
        self.M01 = -0.5j*ones
        self.M10 = -0.5j*ones
        self.M11 = -0.5j*ones


    def LCP2RCP(self):
        """Left circular polarizer to Right circular polarizer
        """


        ones = np.ones_like(self.X)

        self.M00 = 0.5*ones
        self.M01 = 0.5j*ones
        self.M10 = 0.5j*ones
        self.M11 = -0.5j*ones


    def q_plate(self, r0: tuple[float, float],  q: float, phi: float = np.pi, theta = 0*degrees):
        """Generates a q_plate. 

        Args:
            r0 (tuple[float, float]): position of 0.
            q (float): _description_
            phi (float, optional): _description_. Defaults to np.pi.
            theta (_type_, optional): angle of the q_plate. Defaults to 0*degrees.


        Reference:
            J.A. Davis et al. "Analysis of a segmented q-plate tunable retarder for the generation of first-order vector beams" Applied Optics Vol. 54, No. 32 p. 9583 (2015) http://dx.doi.org/10.1364/AO.54.009583
        """


        x0, y0 = r0

        R = np.sqrt((self.X-x0)**2 + (self.Y-y0)**2)
        THETA = np.arctan2((self.Y-y0),(self.X-x0))

        self.M00 = np.cos(phi/2)-1j*np.sin(phi/2)*np.cos(2*q*(THETA-theta))
        self.M01 = -1j*np.sin(phi/2)*np.sin(2*q*(THETA-theta))
        self.M10 = -1j*np.sin(phi/2)*np.sin(2*q*(THETA-theta))
        self.M11 = np.cos(phi/2)+1j*np.sin(phi/2)*np.cos(2*q*(THETA-theta))



    def SLM(self, mask: Scalar_mask_XY, states_jones: Jones_matrix) -> None:
        """
        Mask for an Spatial Light Modulator (SLM). 
        
        Each pixel of the mask is converted to a Jones_matrix, according to its value

        Args:
            mask (Scalar_mask_XY): (0-1) Scalar_mask_XY to send to the SLM. It is a 2D array with values between 0 and 1. The function discretizes the values in the number of levels equal to the lenght of states_jones.
            states_jones (Jones_matrix): Jones matrix calibration of the SLM. It is a LUT with the Jones matrices of the SLM for each level of the mask.

        Returns:
                Vector_mask_XY: Vector simulation of the Spatial Light Modulator.
 
        """
        
        mask_level = np.int_(mask.u*(len(states_jones)-1))

        for i, matrix in enumerate(states_jones):
            index = np.where(mask_level == i)
            self.M00[index] = matrix.M[0,0]
            self.M01[index] = matrix.M[0,1]
            self.M10[index] = matrix.M[1,0]
            self.M11[index] = matrix.M[1,1]
            

    def draw(self, kind: Draw_Options ='amplitudes', range_scale: str='um', cmap_max=1.):
        """Draws the mask. It must be different to sources.

        Args:
            kind (str): 'amplitudes', 'phases', 'jones', 'jones_ap'.

            'jones' is for real and imaginary parts.
            'jones_ap' is for amplitud and phase.
        """
        xsize, ysize = rcParams['figure.figsize']

        extension = np.array([self.x[0], self.x[-1], self.y[0], self.y[-1]])
        if range_scale == 'mm':
            extension = extension / 1000.

        a00, int00, phase00 = field_parameters(self.M00, has_amplitude_sign=False)

        a01, int01, phase01 = field_parameters(self.M01, has_amplitude_sign=False)
        a10, int10, phase10 = field_parameters(self.M10, has_amplitude_sign=False)
        a11, int11, phase11 = field_parameters(self.M11, has_amplitude_sign=False)

        if cmap_max == 1.:
            a_max = 1
        else:
            a_max = np.abs((a00, a01, a10, a11)).max()

        if kind in ('amplitudes', 'jones_ap'):
            plt.set_cmap(CONF_DRAWING['color_intensity'])
            fig, axs = plt.subplots(2,  2,  sharex='col',  sharey='row',  gridspec_kw={'hspace': 0.25, 'wspace': 0.025  })
            fig.set_figwidth(xsize)
            fig.set_figheight(1.25 * ysize)

            im1 = axs[0, 0].imshow(a00, extent=extension, origin='lower')
            im1.set_clim(0, a_max)
            axs[0, 0].set_title("J00")

            im1 = axs[0, 1].imshow(a01, extent=extension, origin='lower')
            im1.set_clim(0, a_max)
            axs[0, 1].set_title("J01")

            im1 = axs[1, 0].imshow(a10, extent=extension, origin='lower')
            im1.set_clim(0, a_max)
            axs[1, 0].set_title("J10")

            im1 = axs[1, 1].imshow(a11, extent=extension, origin='lower')
            im1.set_clim(0, a_max)
            axs[1, 1].set_title("J11")

            plt.suptitle("amplitudes", fontsize=15)
            cax = plt.axes([0.89, 0.2, 0.03, 0.6])
            cbar = plt.colorbar(im1, cax=cax, shrink=0.66)
            if cmap_max == 1.:
                cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])

            if range_scale == 'um':
                axs[1, 0].set_xlabel(r'x ($\mu$m)')
                axs[1, 0].set_ylabel(r'y($\mu$m)')
            elif range_scale == 'mm':
                axs[1, 0].set_xlabel(r'x (mm)')
                axs[1, 0].set_ylabel(r'y (mm)')

        if kind in ('phases', 'jones_ap'):
            plt.set_cmap(CONF_DRAWING['color_phase'])
            fig, axs = plt.subplots(2, 2, sharex='col', sharey='row', gridspec_kw={'hspace': 0.25, 'wspace': 0.025  })
            fig.set_figwidth(xsize)
            fig.set_figheight(1.25 * ysize)
            im1 = axs[0, 0].imshow(np.angle(self.M00)/degrees, extent=extension, origin='lower')
            im1.set_clim(-180, 180)
            axs[0, 0].set_title("J00")

            im1 = axs[0, 1].imshow(np.angle(self.M01)/degrees, extent=extension, origin='lower')
            im1.set_clim(-180, 180)
            axs[0, 1].set_title("J01")

            im1 = axs[1, 0].imshow(np.angle(self.M10)/degrees, extent=extension, origin='lower')
            im1.set_clim(-180, 180)
            axs[1, 0].set_title("J10")

            im1 = axs[1, 1].imshow(np.angle(self.M11)/degrees, extent=extension, origin='lower')
            im1.set_clim(-180, 180)
            axs[1, 1].set_title("J11")

            plt.suptitle("phases", fontsize=15)
            cax = plt.axes([.89, 0.2, 0.03, 0.6])
            cbar = plt.colorbar(im1, cax=cax, shrink=0.66)
            if cmap_max == 1.:
                cbar.set_ticks([-180, -135, -90, -45, 0, 45, 90, 135, 180])

            if range_scale == 'um':
                axs[1, 0].set_xlabel(r'x ($\mu$m)')
                axs[1, 0].set_ylabel(r'y($\mu$m)')
            elif range_scale == 'mm':
                axs[1, 0].set_xlabel(r'x (mm)')
                axs[1, 0].set_ylabel(r'y (mm)')

        if kind in ('jones'):
            plt.set_cmap(CONF_DRAWING['color_stokes'])

            fig, axs = plt.subplots(2,  2,  sharex='col',  sharey='row',  gridspec_kw={'hspace': 0.25, 'wspace': 0.025  })
            fig.set_figwidth(xsize)
            fig.set_figheight(1.25 * ysize)

            im1 = axs[0, 0].imshow(np.real(self.M00), extent=extension, origin='lower')
            im1.set_clim(-1, 1)
            axs[0, 0].set_title("J00")

            im1 = axs[0, 1].imshow(np.real(self.M01), extent=extension, origin='lower')
            im1.set_clim(-1, 1)
            axs[0, 1].set_title("J01")

            im1 = axs[1, 0].imshow(np.real(self.M10), extent=extension, origin='lower')
            im1.set_clim(-1, 1)
            axs[1, 0].set_title("J10")

            im1 = axs[1, 1].imshow(np.real(self.M11), extent=extension, origin='lower')
            im1.set_clim(-1, 1)
            axs[1, 1].set_title("J11")

            plt.tight_layout()
            plt.suptitle(r"$\Re$ (Jones)", fontsize=15)
            cax = plt.axes([0.89, 0.2, 0.03, 0.6])
            cbar = plt.colorbar(im1, cax=cax, shrink=0.66)

            fig, axs = plt.subplots(2,  2,  sharex='col',  sharey='row',  gridspec_kw={'hspace': 0.25, 'wspace': 0.025  })
            fig.set_figwidth(xsize)
            fig.set_figheight(1.25 * ysize)

            im1 = axs[0, 0].imshow(np.imag(self.M00), extent=extension, origin='lower')
            im1.set_clim(-1, 1)
            axs[0, 0].set_title("J00")

            im1 = axs[0, 1].imshow(np.imag(self.M01), extent=extension, origin='lower')
            im1.set_clim(-1, 1)
            axs[0, 1].set_title("J01")

            im1 = axs[1, 0].imshow(np.imag(self.M10), extent=extension, origin='lower')
            im1.set_clim(-1, 1)
            axs[1, 0].set_title("J10")

            im1 = axs[1, 1].imshow(np.imag(self.M11), extent=extension, origin='lower')
            im1.set_clim(-1, 1)
            axs[1, 1].set_title("J11")

            plt.tight_layout()
            plt.suptitle(r"$\Im$ (Jones)", fontsize=15)
            cax = plt.axes([0.89, 0.2, 0.03, 0.6])
            cbar = plt.colorbar(im1, cax=cax, shrink=0.66)


def rotation_matrix_Jones(angle: float):
    """Creates an array of Jones 2x2 rotation matrices.

    Args:
        angle (np.array): array of angle of rotation, in radians.

    Returns:
        numpy.array: 2x2 matrix
    """
    M = np.array([[np.cos(angle), np.sin(angle)],
                  [-np.sin(angle), np.cos(angle)]])
    return M
