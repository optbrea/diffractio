# !/usr/bin/env python3

# ----------------------------------------------------------------------
# Name:        scalar_fields_XZ.py
# Purpose:     Class for working with XZ scalar fields

# Author:      Luis Miguel Sanchez Brea
#
# Created:     2024
# Licence:     GPLv3
# ----------------------------------------------------------------------


"""
This module generates Scalar_field_XZ class.

It includes multiprocessing for RS and BPM polychromatic

It can be considered an extension of Scalar_field_X for visualizing XZ fields

For the case of Rayleigh sommefeld it is not necessary to compute all z positions but the final.

Nevertheless, for BPM method, intermediate computations are required. In this class, intermediate results are stored.

X,Z fields are defined using ndgrid (not with meshgrid, it is different).

It is required also for generating masks and fields.
The main atributes are:
    * self.x - x positions of the field
    * self.z - z positions of the field
    * self.u - field XZ
    * self.n - refractive index XZ
    * self.wavelength - wavelength of the incident field. The field is monochromatic

The magnitude is related to microns: `micron = 1.`

*Class for XZ scalar fields*

*Definition of a scalar field*
    * instatiation, duplicate, clean_refractive_index
    * save, load data
    * rotate_field, cut_resample,

*Illumination*
    * incident_field

*Operations*
    * surface_detection
    * search focus

*Propagation*
    * RS, RS_polychormatic,
    * BPM, BPM_poychromatic, BPM_inverse, BPM_back_propagation

*Drawing functions*
    * draw
    * draw_refractive_index
    * draw_incident_field
    * draw_profiles_interactive
    * video

*Parameters*
    * final_field
    * profile_longitudinal
    * profile_transversal
"""

import copy
import copyreg
import time
import types
from copy import deepcopy
from multiprocessing import Pool

import matplotlib.animation as animation
import matplotlib.cm as cm
from numpy import array, concatenate, gradient, pi, sqrt, zeros
from numpy.lib.scimath import sqrt as csqrt
from scipy.fftpack import fft, fft2, fftshift, ifft, ifft2
from scipy.interpolate import RectBivariateSpline

from .__init__ import np, plt
from .__init__ import num_max_processors, degrees, mm, seconds, um
from .config import (bool_raise_exception, Draw_X_Options, Draw_XZ_Options, Draw_interactive_Options, 
                     Draw_refractive_index_Options, CONF_DRAWING, get_scalar_options)
from .utils_typing import NDArrayFloat
from .utils_common import add, get_date, load_data_common, save_data_common, check_none, oversampling, get_scalar, rmul
from .utils_common import get_instance_size_MB
from .utils_drawing import normalize_draw, prepare_drawing
from .utils_math import get_k, nearest, reduce_to_1, rotate_image
from .utils_multiprocessing import _pickle_method, _unpickle_method
from .utils_optics import FWHM1D, beam_width_1D, field_parameters, normalize_field
from .scalar_fields_X import (PWD_kernel, Scalar_field_X, WPM_schmidt_kernel,
                            kernelRS, kernelRSinverse)
from .scalar_masks_X import Scalar_mask_X
from .scalar_sources_X import Scalar_source_X

copyreg.pickle(types.MethodType, _pickle_method, _unpickle_method)

percentage_intensity_config = CONF_DRAWING['percentage_intensity']

class Scalar_field_XZ():
    """Class for working with XZ scalar fields.

    Args:
        x (numpy.array): linear array with equidistant positions.
            The number of data is preferibly :math:`2^n` .
        z (numpy.array): linear array wit equidistant positions for z values
        wavelength (float): wavelength of the incident field
        n_background (float): refractive index of background
        info (str): String with info about the simulation

    Attributes:
        self.x (numpy.array): linear array with equidistant positions.
            The number of data is preferibly :math:`2^n` .
        self.z (numpy.array): linear array wit equidistant positions for z values
        self.wavelength (float): wavelength of the incident field.
        self.u0 (numpy.array): (x) size x - field at the last z position
        self.u (numpy.array): (x,z) complex field
        self.n_background (numpy.array): (x,z) refractive index
        self.fast (bool): if True fast algoritm (approx. Hankle function)
        self.info (str): String with info about the simulation
    """

    def __init__(self, x: NDArrayFloat | None = None, z: NDArrayFloat | None = None,
                 wavelength: float | None = None, n_background: float = 1., info: str = ""):
        self.x = x
        self.z = z
        self.wavelength = wavelength
        self.n_background = n_background
        self.fast = False
        self.quality = 0
        self.borders = None  
        self.CONF_DRAWING = CONF_DRAWING

        if x is not None and z is not None:
            self.X, self.Z = np.meshgrid(x, z)
            self.u0 = Scalar_field_X(x, wavelength)
            self.u = np.zeros_like(self.X, dtype=complex)
            self.n = n_background * np.ones(np.shape(self.X), dtype=complex)
        else:
            self.X = None
            self.Z = None
            self.u0 = None
            self.u = None
            self.n = None
            
        self.borders = None
            
        self.info = info
        self.reduce_matrix = 'standard'  # 'None, 'standard', (5,5)
        self.type = 'Scalar_field_XZ'
        self.date = get_date()


    @check_none('x','z','u',raise_exception=bool_raise_exception)
    def __str__(self):
        """Represents main data of the atributes"""

        Imin = (np.abs(self.u)**2).min()
        Imax = (np.abs(self.u)**2).max()
        phase_min = (np.angle(self.u)).min()/degrees
        phase_max = (np.angle(self.u)).max()/degrees
        nmin = self.n.min()
        nmax = self.n.max()
        print("{}\n - x:  {},   z:  {},   u:  {}".format(
            self.type, self.x.shape, self.z.shape, self.u.shape))
        print(
            " - xmin:       {:2.2f} um,  xmax:      {:2.2f} um,  Dx:   {:2.2f} um"
            .format(self.x[0], self.x[-1], self.x[1] - self.x[0]))
        print(
            " - zmin:       {:2.2f} um,  zmax:      {:2.2f} um,  Dz:   {:2.2f} um"
            .format(self.z[0], self.z[-1], self.z[1] - self.z[0]))
        print(" - nmin:       {:2.2f},     nmax:      {:2.2f}".format(
            nmin, nmax))
        print(" - Imin:       {:2.2f},     Imax:      {:2.2f}".format(
            Imin, Imax))
        print(" - phase_min:  {:2.2f} deg, phase_max: {:2.2f} deg".format(
            phase_min, phase_max))

        print(" - wavelength: {:2.2f} um".format(self.wavelength))
        print(" - date:       {}".format(self.date))
        if self.info != "":
            print(" - info:       {}".format(self.info))
        return ""
        return ""

    @check_none('x','z','u',raise_exception=bool_raise_exception)
    def __add__(self, other, kind: str = 'standard'):
        """Adds two Scalar_field_XZ. For example two light sources or two masks.

        Args:
            other (Vector_field_X): 2nd field to add
            kind (str): instruction how to add the fields: ['source', 'mask', 'phases', 'no_overlap', 'distances'].
                - 'source': adds the fields as they are
                - 'mask': adds the fields as complex numbers and then normalizes so that the maximum amplitude is 1.
                - 'phases': adds the phases and then normalizes so that the maximum amplitude is 1.
                - 'np_overlap': adds the fields as they are. If the sum of the amplitudes is greater than 1, an error is produced
                - 'distances': adds the fields as they are. If the fields overlap, the field with the smallest distance is kept.

        Returns:
            Scalar_field_XZ: `u3 = u1 + u2`
        """


        u = add(self, other, kind='source')
        u.n = self.n + other.n

        return u


    @check_none('x','z','u',raise_exception=bool_raise_exception)
    def __sub__(self, other):
        """Substract two Scalar_field_x. For example two light sources or two masks.

        Args:
            other (Scalar_field_X): field to substract

        Returns:
            Scalar_field_X: `u3 = u1 - u2`

        TODO: It can be improved for maks (not having less than 1)
        """

        u3 = Scalar_field_XZ(self.x, self.z, self.wavelength,
                             self.n_background)
        u3.n = self.n
        u3.u = self.u - other.u
        return u3



    @check_none('x','z','u',raise_exception=bool_raise_exception)
    def __rmul__(self, number: float | complex | int):
        """Multiply a field by a number.  For example  :math: `u_1(x)= m * u_0(x)`.

        Args:
            number (float | complex | int): number to multiply the field.
            kind (str): instruction how to add the fields: ['intensity', 'amplitude', 'phase'].
                - 'intensity': Multiply the intensity of the field by the number.
                - 'amplitude': Multiply the amplitude of the field by the number.
                - 'phase': Multiply the phase of the field by the number.

        Returns:
            Scalar_field_XYZ:
        """

        if self.type == 'Scalar_mask_XZ':
            t = rmul(self, number, kind='intensity')
        elif self.type == 'Scalar_field_XZ':
            t = rmul(self, number, kind='amplitude')
            
        return t


    @check_none('x','z','u',raise_exception=bool_raise_exception)
    def rmul(self, number, kind):
        """Multiply a field by a number.  For example  :math: `u_1(x)= m * u_0(x)`.

        This function is general for all the SCALAR modules of the package. After, this function is called by the rmul method of each class. 
        When module is for sources, any value for the number is valid. When module is for masks, the modulus is <=1.

        The kind parameter is used to specify how to multiply the field. The options are:
        - 'intensity': Multiply the intensity of the field by the number.
        - 'amplitude': Multiply the amplitude of the field by the number.
        - 'phase': Multiply the phase of the field by the number.
        
        Args:
            number (float | complex | int): number to multiply the field.
            kind (str): instruction how to add the fields: ['intensity', 'amplitude', 'phase'].
                - 'intensity': Multiply the intensity of the field by the number.
                - 'amplitude': Multiply the amplitude of the field by the number.
                - 'phase': Multiply the phase of the field by the number.

        Returns:
            The field multiplied by the number.
        """

        t = rmul(self, number, kind)
           
        return t


    @check_none('x','z',raise_exception=bool_raise_exception)
    def __rotate__(self, angle: float, position=None):
        """Rotation of X,Z with respect to position

        Args:
            angle (float): angle to rotate, in radians
            position (float, float): position of center of rotation
        """

        if position is None:
            x0 = (self.x[-1] + self.x[0])/2
            z0 = (self.z[-1] + self.z[0])/2
        else:
            # Definicion de la rotation
            x0, z0 = position

        Xrot = x0 + (self.X - x0) * np.cos(angle) + (self.Z -
                                                     z0) * np.sin(angle)
        Zrot = z0 - (self.X - x0) * np.sin(angle) + (self.Z -
                                                     z0) * np.cos(angle)
        return Xrot, Zrot


    def size(self, verbose: bool = False):
        """returns the size of the instance in MB.

        Args:
            verbose (bool, optional): prints size in Mb. Defaults to False.

        Returns:
            float: size in MB
        """

        return get_instance_size_MB(self, verbose)

        

    def reduce_to_1(self):
        """All the values greater than 1 pass to 1. This is used for Scalar_masks when we add two masks.
        """

        self = reduce_to_1(self)


    def duplicate(self, clear: bool = False):
        """Duplicates the instance.

        Args:
            clear (bool): if True, it clears the field.

        Returns: 
            Scalar_field_XZ: duplicated instance
        """

        new_field = copy.deepcopy(self)
        
        if clear is True:
            new_field.clear_field()
            
        return new_field


    def refractive_index_from_scalar_mask_XY(self, mask_XY, refractive_index_max: float):
        """Transforms XY mask into XZ mask.
            - Areas with value 0 pass to n_background.
            - When transmittance of mask_XY  is 1, pass to refractive_index_max.

        Args:
            mask_XY (diffractio.Scalar_mask_XY): mask
            refractive_index_max (float): real and imaginary part of maximum refractive index.

        Returns:
            _type_: _description_
        """

        self.x = mask_XY.x
        self.z = mask_XY.y
        self.X, self.Z = np.meshgrid(self.x, self.z)
        self.u = np.zeros_like(self.X, dtype=complex)

        self.n = mask_XY.u * refractive_index_max
        self.n[self.n < 1] = self.n_background


    @check_none('x','z','u','n',raise_exception=bool_raise_exception)
    def rotate_field(self, angle: float, center_rotation: tuple[float, float],
                     kind: str = 'all', n_background: float = 1.):
        """Rotate all the image a certain angle

        Args:
            angle (float): angle to rotate, in radians
            center_rotation (float, float): (z,x) position for rotation
            kind (str): 'all', 'n', 'field'
            n_background (float): refractive index of zone incoming
        """
        angle = -angle

        if kind in ('n', 'all'):
            n_real_rotate = rotate_image(self.z, self.x, np.real(self.n),
                                         angle * 180 / pi, center_rotation)
            n_imag_rotate = rotate_image(self.z, self.x, np.imag(self.n),
                                         angle * 180 / pi, center_rotation)
            n_rotate = n_real_rotate + 1j * n_imag_rotate
            n_rotate[n_rotate < n_background] = n_background
            self.n = n_rotate
            self.n[self.n < 1.2] = self.n_background

        self.surface_detection(mode=1, min_incr=0.1, has_draw=False)

        if kind in ('field', 'all'):
            u_real_rotate = rotate_image(self.z, self.x, np.real(self.u),
                                         angle * 180 / pi, center_rotation)
            u_imag_rotate = rotate_image(self.z, self.x, np.imag(self.u),
                                         angle * 180 / pi, center_rotation)
            u_rotate = u_real_rotate + 1j * u_imag_rotate
            self.u = u_rotate


    @check_none('u',raise_exception=bool_raise_exception)
    def clear_field(self):
        """clear field"""
        self.u = np.zeros(np.shape(self.u), dtype=complex)


    @check_none('X',raise_exception=bool_raise_exception)
    def clear_refractive_index(self):
        """clear refractive index n(x,z)=n_background"""

        self.n = self.n_background * np.ones_like(self.X, dtype=complex)


    def normalize(self, kind='amplitude', new_field: bool = False):
        """Normalizes the field so that intensity.max()=1.

        Args:
            kind (str): 'amplitude', or 'intensity'
            new_field (bool): If False the computation goes to self.u. If True a new instance is produced

        Returns
            u (numpy.array): normalized optical field
        """
        return normalize_field(self, kind, new_field)


    @check_none('x',raise_exception=bool_raise_exception)
    def mask_field(self, size_edge: float = 0):
        """
        mask the incident field at the edges, each edge is masked size_edge

        Args:
            size_edge (float): size of edges
        """

        L = self.x[-1] - self.x[0]
        x_center = (self.x[-1] + self.x[0])/2
        mask = Scalar_mask_X(x=self.x, wavelength=self.wavelength)
        mask.slit(x0=x_center, size=L - 2 * size_edge)
        self.u0.u = self.u0.u * mask.u


    @check_none('n',raise_exception=bool_raise_exception)
    def smooth_refractive_index(self,
                                type_filter: int = 2,
                                pixels_filtering: int = 10,
                                max_diff_filter: float = 0.1,
                                draw_check: bool = False):
        """
        Technique to remove artifacts in BPM propagation.

        Args:
            type_filter (int): 1 - 2D, 2 - 1D z (selective), 3 - 1D x (selective)
            pixels_filtering (int): num_pixels used for filtering
            max_diff_filter (float): maximum difference of n in profile between two adjancted pixels to use selective filtering
            draw_check (bool): draw the differences.

        Returns:
            (float): percentage_filtered
            (np.array): lineas_filtradas

        References:
            Robert McLeod "Numerical Methods in Photonics Lecture Notes"  University of Colorado at Boulder, pag 204 (15/54)
        """

        if draw_check is True:
            indice_sin_variar = deepcopy(self.n)

        num_filtrados = 0

        if type_filter == 1:
            # Filtro 2D, pero solo ejecuta en una dirección
            lineas_filtradas = np.ones_like(self.z)

            filtro1 = np.zeros_like(self.n)
            sizex, sizez = self.n.shape
            centerx, centerz = int(sizex/2), int(sizez/2)
            filtro1[centerx - pixels_filtering:centerx + pixels_filtering,
                    centerz - 1:centerz + 1] = 1
            filtro1 = filtro1 / sum(sum(filtro1))
            self.n = fftshift(ifft2(fft2(self.n) * fft2(filtro1)))
            percentage_filtered = 0

        elif type_filter == 2:
            # Filtro 1D, solo ejecuta cuando hay diferencias de índice eje x
            lineas_filtradas = np.zeros_like(self.z)
            filtro1 = np.zeros_like(self.x)
            sizex = len(filtro1)
            centerx = (self.x[-1] + self.x[0])/2

            filtro1 = np.exp(-(self.x - centerx)**2 /
                             (2 * pixels_filtering**2))
            filtro1 = filtro1 / sum(filtro1)
            for i in range(len(self.z)):
                max_diff = np.abs(np.diff(self.n[i, :])).max()
                if max_diff > max_diff_filter:
                    lineas_filtradas[i] = 1
                    self.n[i, :] = fftshift(
                        ifft(fft(self.n[i, :]) * fft(filtro1)))
                    num_filtrados = num_filtrados + 1
            percentage_filtered = 100 * num_filtrados / len(self.z)

        elif type_filter == 3:
            lineas_filtradas = np.zeros_like(self.x)
            filtro1 = np.zeros_like(self.z)
            sizez = len(filtro1)
            centerz = int(sizez/2)
            filtro1[centerz - pixels_filtering:centerz + pixels_filtering] = 1
            filtro1 = filtro1 / sum(filtro1)
            for i in range(len(self.x)):
                max_diff = np.abs(np.diff(self.n[:, i])).max()
                if max_diff > max_diff_filter:
                    lineas_filtradas[i] = 1
                    self.n[:, i] = fftshift(
                        ifft(fft(self.n[:, i]) * fft(filtro1)))
                    num_filtrados = num_filtrados + 1
            percentage_filtered = 100 * num_filtrados / len(self.x)

        if draw_check is True:
            plt.figure()
            plt.plot(self.z, lineas_filtradas)
            plt.xlabel(r'z ($\mu m$)')
            plt.ylabel('filtered zone)')
            plt.title("detection of edges", fontsize=24)

            diferencias_indice = np.abs(self.n - indice_sin_variar)

            extension = [self.z[0], self.z[-1], self.x[0], self.x[-1]]

            plt.figure()

            h1 = plt.imshow(diferencias_indice,
                            interpolation='bilinear',
                            aspect='auto',
                            origin='lower',
                            extent=extension)
            plt.xlabel(r'z ($\mu m$)')
            plt.ylabel(r'x ($\mu m$)')

            plt.axis(extension)
            h1.set_cmap(cm.gray_r)
            plt.title("smooth_refractive_index: n variations", fontsize=24)

        return percentage_filtered, lineas_filtradas


    def save_data(self, filename: str, add_name: str = "",
                  description: str = "", verbose: bool = False):
        """Common save data function to be used in all the modules.
        The methods included are: npz, matlab

        Args:
            filename (str): filename
            add_name= (str): sufix to the name, if 'date' includes a date
            description (str): text to be stored in the dictionary to save.
            verbose (bool): If verbose prints filename.

        Returns:
            (str): filename. If False, file could not be saved.
        """
        try:
            final_filename = save_data_common(self, filename, add_name, description, verbose)
            return final_filename
        except:
            return False


    def load_data(self, filename: str, verbose: bool = False):
        """Load data from a file to a Scalar_field_XZ.
            The methods included are: npz, matlab

        Args:
            filename (str): filename
            verbose (bool): shows data process by screen
        """
        dict0 = load_data_common(self, filename, verbose)

        if dict0 is not None:
            if isinstance(dict0, dict):
                self.__dict__ = dict0
            else:
                raise Exception('no dictionary in load_data')


    @check_none('x','z','u')
    def oversampling(self, factor_rate: int | tuple):
        """Overfample function has been implemented in scalar X, XY, XZ, and XYZ frames reduce the pixel size of the masks and fields. 
        This is also performed with the cut_resample function. However, this function oversamples with integer factors.
        
        Args:
            factor_rate (int | tuple, optional): factor rate. Defaults to 2.
        """

        self = oversampling(self, factor_rate)

    @check_none('x','z','u',raise_exception=bool_raise_exception)
    def cut_resample(self,
                     x_limits: tuple[float, float] | str = '',
                     z_limits: tuple[float, float] | str = '',
                     num_points: int | None = None,
                     new_field: bool = False,
                     interp_kind: tuple[int, int] = (3, 1)):
        """it cut the field to the range (x0,x1). if one of this x0,x1 positions is out of the self.x range it do nothing. It is also valid for resampling the field, just write x0,x1 as the limits of self.x

        Args:
            x_limits (float,float): (x0,x1) starting and final points to cut. if '' - takes the current limit x[0] and x[-1]
            z_limits (float,float): (z0,z1) - starting and final points to cut. if '' - takes the current limit z[0] and z[-1]
            num_points (int): it resamples x, z and u. ([],'',0,None) -> it leave the points as it is
            new_field (bool): it returns a new Scalar_field_XZ
            interp_kind: numbers between 1 and 5
        """
        if x_limits == '':
            x0 = self.x[0]
            x1 = self.x[-1]
        else:
            x0, x1 = x_limits

        if z_limits == '':
            z0 = self.z[0]
            z1 = self.z[-1]
        else:
            z0, z1 = z_limits

        if x0 < self.x[0]:
            x0 = self.x[0]
        if x1 > self.x[-1]:
            x1 = self.x[-1]

        if z0 < self.z[0]:
            z0 = self.z[0]
        if z1 > self.z[-1]:
            z1 = self.z[-1]

        i_x0, _, _ = nearest(self.x, x0)
        i_x1, _, _ = nearest(self.x, x1)

        i_z0, _, _ = nearest(self.z, z0)
        i_z1, _, _ = nearest(self.z, z1)

        kxu, kxn = interp_kind

        if num_points not in ([], '', 0, None):
            num_points_x, num_points_z = num_points
            x_new = np.linspace(x0, x1, num_points_x)
            z_new = np.linspace(z0, z1, num_points_z)
            X_new, Z_new = np.meshgrid(x_new, z_new)

            f_interp_abs = RectBivariateSpline(self.z,
                                               self.x,
                                               np.abs(self.u),
                                               kx=kxu,
                                               ky=kxu,
                                               s=0)
            f_interp_phase = RectBivariateSpline(self.z,
                                                 self.x,
                                                 np.angle(self.u),
                                                 kx=kxu,
                                                 ky=kxu,
                                                 s=0)
            u_new_abs = f_interp_abs(z_new, x_new)
            u_new_phase = f_interp_phase(z_new, x_new)
            u_new = u_new_abs * np.exp(1j * u_new_phase)

            n_interp_real = RectBivariateSpline(self.z,
                                                self.x,
                                                np.real(self.n),
                                                kx=kxn,
                                                ky=kxn,
                                                s=0)
            n_interp_imag = RectBivariateSpline(self.z,
                                                self.x,
                                                np.imag(self.n),
                                                kx=kxn,
                                                ky=kxn,
                                                s=0)
            n_new_real = n_interp_real(z_new, x_new)
            n_new_imag = n_interp_imag(z_new, x_new)
            n_new = n_new_real + 1j * n_new_imag

        else:
            iz_s = slice(i_z0, i_z1)
            jx_s = slice(i_x0, i_x1)
            x_new = self.x[jx_s]
            z_new = self.z[iz_s]
            X_new, Z_new = np.meshgrid(x_new, z_new)
            u_new = self.u[iz_s, jx_s]
            n_new = self.n[iz_s, jx_s]

        if new_field is False:
            self.x = x_new
            self.z = z_new
            self.u = u_new
            self.n = n_new
            self.X = X_new
            self.Z = Z_new
        else:
            field = Scalar_field_XZ(x=x_new,
                                    z=z_new,
                                    wavelength=self.wavelength)
            field.u = u_new
            field.n = n_new

            return field

    @check_none('x','z','u',raise_exception=bool_raise_exception)
    def incident_field(self, u0, z0: float | None = None):
        """Incident field for the experiment. It takes a Scalar_source_X field

        Args:
            u0 (Scalar_source_X): field produced by Scalar_source_X (or a X field)
            z0 (float): position of the incident field. if None, '', [], is at the beginning
        """

        if z0 in (None, '', []):
            self.u0 = u0
        else:
            iz, _, _ = nearest(self.z, z0)
            self.u[iz, :] = self.u[iz, :] + u0.u

    @check_none('x','z','u')
    def final_field(self):
        """Returns the final field as a Scalar_field_X."""

        u_final = Scalar_field_X(x=self.x,
                                 wavelength=self.wavelength,
                                 n_background=self.n_background,
                                 info="from final_field at z0= {} um".format(
                                     self.z[-1]))
        u_final.u = self.u[-1,:]
        return u_final
    

    @check_none('u',raise_exception=bool_raise_exception)
    def get(self, kind: get_scalar_options):
        """Get parameters from Scalar field.

        Args:
            kind (str): 'intensity', 'phase', 'field'

        Returns:
            matrices with required values
        """

        data = get_scalar(self, kind)
        return data
 

    @check_none('x','z','n')
    def __BPM__(self,
                has_edges: bool = True,
                pow_edge: int = 80,
                matrix: bool = False,
                verbose: bool = False):
        """Beam propagation method (BPM).

        Args:
            has_edges (bool): If True absorbing edges are used.
            pow_edge (float): If has_edges, power of the supergaussian
            matrix (bool): if True returns matrix, else goes to self.u
            verbose (bool): shows data process by screen

        References:
            Algorithm in "Engineering optics with matlab" pag 119

        """
        dn = np.abs(np.diff(self.n).max())
        dz = self.z[1] - self.z[0]

        q1 = (0.25 * self.wavelength/2 * dn / dz,
              0.25 * (self.x[-1] - self.x[0])**2 / self.wavelength / dz)
        self.quality = q1

        k0 = 2 * np.pi / self.wavelength

        numz = len(self.z)  # distance en z
        numx = len(self.x)  # distance en x
        deltaz = self.z[1] - self.z[0]  # Tamaño del sampling
        rangox = self.x[-1] - self.x[0]

        pixelx = np.linspace(-int(numx/2), int(numx/2), numx)
        # initial field
        field_z = self.u0.u
        
        # Calculo de la phase 1 normalizada -------------------
        kx1 = np.linspace(0, int(numx/2) + 1, int(numx/2))
        kx2 = np.linspace(-int(numx/2), -1, int(numx/2))
        # Número de ondas del material en una dimensión
        kx = (2 * np.pi / rangox) * np.concatenate((kx1, kx2))
        # Función de transferencia para la propagación que es identica
        # a la respuesta de frecuencia espacial en óptica de Fourier
        # incorporando el termino (-j k0 z).
        phase1 = np.exp((-1j * deltaz * kx**2) / (2 * k0))
        # Campo en el índice de refracción
        field = np.zeros(np.shape(self.n), dtype=complex)

        if has_edges is False:
            has_filter = np.zeros_like(self.z)
        elif has_edges is True:
            has_filter = np.ones_like(self.z)
        elif isinstance(has_edges, (int, float)):
            has_filter = np.zeros_like(self.z)
            iz, _, _ = nearest(self.z, has_edges)
            has_filter[iz:] = 1
        else:
            has_filter = has_edges

        width_edge = 0.9*(self.x[-1]-self.x[0])/2
        x_center = (self.x[-1]+self.x[0])/2

        filter_function = np.exp(-(np.abs(self.x -
                                 x_center) / width_edge)**pow_edge)

        field[0, :] = field_z
        for k in range(0, numz):
            
            if has_filter[k] == 0:
                filter_edge = 1
            else:
                filter_edge = filter_function

            if verbose is True:
                print("BPM: {}/{}".format(k, numz), sep="\r", end="\r")

            phase2 = np.exp(1j * self.n[k, :] * k0 * deltaz)
            field_z = ifft((fft(field_z) * phase1)) * phase2
            self.u[k, :] = self.u[k, :] + field_z * filter_edge

        if matrix is True:
            return self.u



    @check_none('x','z','n')
    def BPM(self, has_edges: bool = True, pow_edge: int = 80, division: bool = False, matrix: bool = False,
            verbose: bool = False):
        """Beam propagation method (BPM).

            Args:
                has_edges (bool or np.array): If True absorbing edges are used. If np.array, they are 0 or 1 depending if at this z position filtering is performed.
                pow_edge (float): If has_edges, power of the supergaussian
                division (False, int): If False nothing, else divides the BPM algorithm in several different executions. To avoid RAM problems
                matrix (bool): if True returns a matrix else
                verbose (bool): shows data process by screen

            References:
            Algorithm in "Engineering optics with matlab" pag 119.
        """

        t1 = time.time()

        if division is False:
            self.__BPM__(has_edges, pow_edge, matrix, verbose)

        else:
            # Here is the division of self.z in parts
            num_executions = int(np.ceil(len(self.z) / division))
            uf = self.u0
            for i in range(num_executions):
                if verbose is True:
                    print("{}/{}".format(i, num_executions),
                          sep="\r",
                          end="\r")

                sl = slice(i * division, (i + 1) * division)
                ui = Scalar_field_XZ(x=self.x,
                                     z=self.z[sl],
                                     wavelength=self.wavelength,
                                     n_background=self.n_background)
                ui.n = self.n[sl, :]
                ui.u0 = uf

                ui = self.__BPM__(ui, has_edges, pow_edge, matrix, verbose)  # BPM -> __BPM__ (240925)
                uf = ui.final_field().u
                self.u[sl, :] = ui.u

            if matrix is True:
                return self.u
        if verbose is True:
            t2 = time.time()
            print("Time = {:2.2f} s, time/loop = {:2.4} ms".format(
                t2 - t1, (t2 - t1) / len(self.z) * 1000))

    @check_none('x','z','n')
    def BPM_inverse(self, verbose: bool = False):
        """
        Beam propagation method (BPM) in inverse mode.

        Args:
            verbose (bool): shows data process by screen

        References:
            Algorithm in "Engineering optics with matlab" pag 119
        """

        c_inverse = Scalar_field_XZ(x=self.x,
                                    z=self.z,
                                    wavelength=self.wavelength,
                                    n_background=self.n_background)
        c_inverse.n = np.fliplr(self.n)
        c_inverse.u0.u = np.conjugate(self.u[-1, :])
        c_inverse.u = np.zeros_like(self.u)
        c_inverse.BPM(verbose)

        # ATENCIÓN, HAGO LA INVERSA Y LA REPRESENTACIÓN ES IGUAL QUE LA DIRECTA
        c_inverse.n = (np.fliplr(c_inverse.n))
        c_inverse.u = (np.fliplr(c_inverse.u))
        return c_inverse

    def BPM_back_propagation(self, verbose: bool = False):
        """
        Beam propagation method (BPM). The field that generates the final field is obtained.

        Args:
            verbose (bool): shows data process by screen

        References:
            Algorithm in "Engineering optics with matlab" pag 119
        """

        c_backpropagation = Scalar_field_XZ(x=self.x,
                                            z=self.z,
                                            wavelength=self.wavelength,
                                            n_background=self.n_background)
        c_backpropagation.n = np.fliplr(self.n)
        c_backpropagation.u = np.fliplr(self.u)
        c_backpropagation.BPM(verbose)
        c_backpropagation.n = (np.fliplr(c_backpropagation.n))
        c_backpropagation.u = (np.fliplr(c_backpropagation.u))
        return c_backpropagation


    def __RS_multiprocessing__(self, i: int):
        """Internal for multiprocessing

        """
        if self.z.min() > 0:
            H = kernelRS(self.xtemp,
                         self.wavelength,
                         self.z[i],
                         self.n_background,
                         kind='z',
                         fast=self.fast)
        else:
            H = kernelRSinverse(self.xtemp,
                                self.wavelength,
                                self.z[i],
                                self.n_background,
                                kind='z',
                                fast=self.fast)

        dx = self.x[1] - self.x[0]

        S = ifft(fft(self.Utemp) * fft(H)) * dx
        nx = len(self.x)
        return S[nx - 1:]

    def RS(self, xout: float | None = None, verbose: bool = False,
           num_processors: int = num_max_processors):
        """Rayleigh Sommerfeld propagation algorithm

        Args:
            xout (float): init position of output position
            verbose (bool): shows the quality of algorithm (>1 good)
            num_processors (int): number of processors for multiprocessing

        Returns:
            Processing time
        """

        time1 = time.time()

        xin = self.x
        x1 = self.x[0]
        xin1 = self.x[0]

        if xout is None:
            xout = self.x
        nx = len(xout)
        dx = xout[1] - xout[0]

        # parametro de quality
        dr_real = sqrt(dx**2)
        rmax = sqrt((xout**2).max())
        dr_ideal = sqrt((self.wavelength / self.n_background)**2 + rmax**2 +
                        2 * (self.wavelength / self.n_background) *
                        sqrt(rmax**2 + self.z.min()**2)) - rmax
        self.quality = dr_ideal / dr_real

        # when computation is performed: quality is determined
        if (self.quality < 1):
            print('- Needs denser sampling: factor: {:2.2f}'.format(
                self.quality))
        else:
            if verbose is True:
                print('Good result: factor {:2.2f}'.format(
                    self.quality), sep="\r", end="\r")

        # matrix W para integracion simpson
        a = [2, 4]
        num_rep = int(round((nx)/2) - 1)
        # print(num_rep)

        b = array(a * num_rep)
        W = concatenate(((1, ), b, (2, 1))) / 3.

        if float(nx)/2 == round(nx/2):  # es par
            i_central = num_rep + 1
            W = concatenate((W[:i_central], W[i_central + 1:]))

        xext = x1 - xin[::-1]  # da la vuelta
        xext = xext[0:-1]
        xext = concatenate((xext, self.x - xin1))

        # field
        U = zeros(2 * nx - 1, dtype=complex)
        U[0:nx] = W * self.u0.u
        self.Utemp = U

        self.xtemp = xext

        if num_processors == 1:
            for i_z, z in enumerate(self.z):

                if z >= 0:
                    H = kernelRS(xext,
                                 self.wavelength,
                                 z,
                                 self.n_background,
                                 kind='z',
                                 fast=self.fast)
                else:
                    H = kernelRSinverse(xext,
                                        self.wavelength,
                                        z,
                                        self.n_background,
                                        kind='z',
                                        fast=self.fast)

                # calculo de la transformada de Fourier
                S = ifft(fft(U) * fft(H)) * dx
                self.u[i_z,:] = S[nx - 1:]  # hasta el final
        else:
            pool = Pool(num_processors)
            t = pool.map(self.__RS_multiprocessing__, list(range(len(self.z))))
            pool.close()
            pool.join()
            for i in range(len(self.z)):
                self.u[i, :] = t[i]
        time2 = time.time()

        if verbose is True:
            print("time in RS= {}. num proc= {}".format(
                time2 - time1, num_processors), sep="\r", end="\r")

        return self.u


    @check_none('x','z','u')
    def PWD(self, n: float | None = None, matrix: bool = False, verbose: bool = False):
        """
        Plane wave decomposition algorithm (PWD).

        Args:
            n (np. array): refractive index, If None, it is n_background
            matrix (bool): if True returns a matrix else
            verbose (bool): If True prints state of algorithm

        Returns:
            numpy.array(): Field at at distance dz from the incident field

        References:
            1. Schmidt, S. et al. Wave-optical modeling beyond the thin-element-approximation. Opt. Express 24, 30188 (2016).

        """
        dx = self.x[1] - self.x[0]
        dz = self.z[1] - self.z[0]
        k0 = 2 * np.pi / self.wavelength

        if n is None:
            n = self.n_background

        kx = get_k(self.x, '+')

        K_perp2 = kx**2
        num_steps = len(self.z)
        self.clear_field()
        self.u[0,:] = self.u0.u
        for i, zi in enumerate(self.z[0:-1]):
            result = self.u[i,:]
            result_next = PWD_kernel(result, n, k0, K_perp2, dz)
            self.u[i + 1,:] = result_next
            if verbose:
                print("{}/{}".format(i, num_steps), sep='\r', end='\r')

        if matrix is True:
            return self.u

    @check_none('x','z')
    def WPM(self, kind: str = 'schmidt', has_edges: bool = True, pow_edge: int = 80,
            matrix: bool = False, verbose: bool = False):
        """
        WPM Method. 'schmidt method is very fast, only needs discrete number of refractive indexes'


        Args:
            kind (str): 'schmidt', 'scalar'
            has_edges (bool): If True absorbing edges are used. 
                It can be a float, indicanting z value when edges start
            pow_edge (float): If has_edges, power of the supergaussian
            matrix (bool): if True returns a matrix else
            verbose (bool): If True prints information

        References:

            1. M. W. Fertig and K.-H. Brenner, “Vector wave propagation method,” J. Opt. Soc. Am. A, vol. 27, no. 4, p. 709, 2010.

            2. S. Schmidt et al., “Wave-optical modeling beyond the thin-element-approximation,” Opt. Express, vol. 24, no. 26, p. 30188, 2016.

        """

        k0 = 2 * np.pi / self.wavelength
        x = self.x
        z = self.z
        dx = x[1] - x[0]
        dz = z[1] - z[0]

        self.u[0, :] = self.u0.u
        kx = get_k(x, flavour='+')
        k_perp2 = kx**2
        k_perp = kx

        X, KP = np.meshgrid(x, k_perp)

        if has_edges is False:
            has_filter = np.zeros_like(self.z)
        elif has_edges is True:
            has_filter = np.ones_like(self.z)
        elif isinstance(has_edges, (int, float)):
            has_filter = np.zeros_like(self.z)
            iz, _, _ = nearest(self.z, has_edges)
            has_filter[iz:] = 1
        else:
            has_filter = has_edges

        width_edge = 0.95*(self.x[-1]-self.x[0])/2
        x_center = (self.x[-1]+self.x[0])/2

        filter_function = np.exp(-(np.abs(self.x -
                                 x_center) / width_edge)**pow_edge)

        t1 = time.time()

        num_steps = len(self.z)
        for j in range(1, num_steps):

            if has_filter[j] == 0:
                filter_edge = 1
            else:
                filter_edge = filter_function

            if kind == 'schmidt':
                self.u[j, :] = self.u[j, :] + WPM_schmidt_kernel(
                    self.u[j - 1, :], self.n[j - 1, :], k0, k_perp2,
                    dz) * filter_edge
            elif kind == 'scalar':

                Nj, KP = np.meshgrid(self.n[j - 1, :], k_perp)
                X, UJ = np.meshgrid(x, fftshift(fft(self.u[j - 1, :])))

                kz = csqrt((Nj * k0)**2 - KP**2)
                Pj = np.exp(1j * kz * dz)

                WXj = UJ * np.exp(1j * KP * (X - x[0]))

                uj = np.mean(WXj * Pj, 0)
                self.u[j, :] = self.u[j, :] + uj * filter_edge

            if verbose is True:
                print("WPM: {}/{}".format(j,num_steps), sep='\r', end='\r')

        t2 = time.time()
        if verbose is True:
            print("Time = {:2.2f} s, time/loop = {:2.4} ms".format(
                t2 - t1, (t2 - t1) / len(self.z) * 1000))
            get_instance_size_MB(self, verbose)

        if matrix is True:
            return self.u


    def RS_polychromatic(self,
                         initial_field,
                         wavelengths: NDArrayFloat,
                         spectrum: NDArrayFloat or None = None,
                         verbose: bool = False,
                         num_processors: int = num_max_processors):
        """Rayleigh Sommerfeld propagation algorithm for polychromatic light.

        Args:
            initial_field (Scalar_field_X): function with only input variable wavelength
            wavelengths (numpy.array): array with wavelengths
            spectrum (numpy.array): array with spectrum. if '' then uniform_spectrum
            verbose (bool): shows the quality of algorithm (>1 good)
            num_processors (int): number of processors for multiprocessing

        Returns:
            Scalar_field_XZ: self.u=sqrt(Intensities) - no phase is stored, only intensity
        """
        if isinstance(spectrum, np.ndarray):
            pass
        elif spectrum in ('', None, [], 0):
            spectrum = np.ones_like(wavelengths)

        I_final = np.zeros_like(self.u, dtype=float)
        u_temp = Scalar_field_XZ(self.x, self.z, self.wavelength,
                                 self.n_background)
        for i, wavelength in enumerate(wavelengths):
            if verbose:
                print("{}/{}".format(i, len(wavelengths)), end='\r')
            u_temp.wavelength = wavelength
            self.u = np.zeros_like(self.X, dtype=complex)
            u_ini = initial_field(wavelength)
            u_temp.clear_field()
            u_temp.incident_field(u_ini)
            u_temp.RS(verbose=verbose, num_processors=num_processors)
            I_final = I_final + spectrum[i] * np.abs(u_temp.u)**2

        u_temp.u = np.sqrt(I_final)
        return u_temp


    @check_none('x','z','u')
    def BPM_polychromatic(self,
                          initial_field,
                          wavelengths: NDArrayFloat,
                          spectrum: NDArrayFloat,
                          verbose: bool = False,
                          num_processors: int = 4):
        """Rayleigh Sommerfeld propagation algorithm for polychromatic light

        Args:
            initial_field (Scalar_field_X): function with only input variable wavelength
            wavelengths (numpy.array): array with wavelengths
            spectrum (numpy.array): array with spectrum. if '' then uniform_spectrum
            verbose (bool): shows the quality of algorithm (>1 good)
            num_processors (int): number of processors for multiprocessing

        Returns:
            Scalar_field_XZ: self.u=sqrt(Intensities) - no phase is stored, only intensity
        """

        if isinstance(spectrum, np.ndarray):
            pass
        elif spectrum in ('', None, [], 0):
            spectrum = np.ones_like(wavelengths)

        I_final = np.zeros_like(self.u, dtype=float)
        for i, wavelength in enumerate(wavelengths):
            u_temp = initial_field(wavelength)
            u_temp.BPM(verbose=False)
            I_final = I_final + spectrum[i] * np.abs(u_temp.u)**2
        u_temp.u = np.sqrt(I_final)
        return u_temp
    

    @check_none('x','z','u')
    def WPM_polychromatic(self,
                          initial_field,
                          wavelengths: NDArrayFloat,
                          spectrum: NDArrayFloat,
                          verbose: bool = False,
                          num_processors: int = 4):
        """WPM propagation algorithm for polychromatic light

        Args:
            initial_field (Scalar_field_X): function with only input variable wavelength
            wavelengths (numpy.array): array with wavelengths
            spectrum (numpy.array): array with spectrum. if '' then uniform_spectrum
            verbose (bool): shows the quality of algorithm (>1 good)
            num_processors (int): number of processors for multiprocessing

        Returns:
            Scalar_field_XZ: self.u=sqrt(Intensities) - no phase is stored, only intensity
        """

        if isinstance(spectrum, np.ndarray):
            pass
        elif spectrum in ('', None, [], 0):
            spectrum = np.ones_like(wavelengths)

        I_final = np.zeros_like(self.u, dtype=float)
        for i, wavelength in enumerate(wavelengths):
            u_temp = initial_field(wavelength)
            u_temp.WPM(verbose=False)
            I_final = I_final + spectrum[i] * np.abs(u_temp.u)**2
        u_temp.u = np.sqrt(I_final)
        return u_temp
    

    @check_none('x','z','u')
    def fast_propagation(self, mask_xz, num_pixels_slice: int = 1024, verbose: bool = False):
        """combines RS and BPM"" to generate the final field

        Args:
            mask_xz (Scalar_mask_XZ): function that returns Scalar_mask_XZ
            num_pixels_slice (int): num of slices for each BPM propagation
            verbose (bool): If True prints info.

        Returns:
            u_current, fields_BPM, transitions
        """

        # check which parts are constant
        z_transitions, algorithm, refr_index_RS = self._detect_transitions_()

        if verbose:
            print(("z_transitions={}".format(z_transitions)))
            print(("algorithm={}".format(algorithm)))
            print(("refr_index_RS={}".format(refr_index_RS)))
        transitions = (z_transitions, algorithm, refr_index_RS)

        fields_BPM = []
        u_current = Scalar_source_X(x=self.x, wavelength=self.wavelength)
        u_current.u = self.u0.u

        for i_zone in range(len(z_transitions) - 1):
            if algorithm[i_zone] == 'RS':  # jump
                distance = z_transitions[i_zone + 1] - z_transitions[i_zone]
                if verbose:
                    print(("i={}, distance = {:2.2f} um".format(
                        i_zone, distance)))
                u_current.RS(z=distance,
                             n=refr_index_RS[i_zone],
                             kind='z',
                             fast=self.fast,
                             new_field=False,
                             verbose=False)
                # u_current.draw()
            else:  # genero experimento para BPM
                x0 = self.x
                z0 = np.linspace(z_transitions[i_zone],
                                 z_transitions[i_zone + 1], num_pixels_slice)
                u1 = mask_xz(x0, z0, u_current, self.wavelength,
                             self.n_background)
                # u1.draw_refractive_index(scale='equal')
                u1.BPM()
                u1.draw(draw_borders=False)
                fields_BPM.append(u1)
                u_current = u1.final_field()
        return u_current, fields_BPM, transitions


    @check_none('u')
    def intensity(self):
        """Returns the intensity of the field

        Returns:
            (np.array): intensity of the field.
        """

        return np.abs(self.u)**2

    @check_none('x','u')
    def average_intensity(self, has_draw: bool = False):
        """Returns average intensity as: (np.abs(self.u)**2).mean()

        Args:
            has_draw(bool): If True draws data.

        Returns:
            intensity_mean (np.array): z array with average intensity at each plane z.

        """
        intensity_mean = (np.abs(self.u)**2).mean(axis=0)
        if has_draw is True:
            plt.figure()
            plt.plot(self.x, intensity_mean)

        return intensity_mean


    @check_none('x','z','u')
    def check_intensity(self, has_draw: bool = True, normalized: bool = True):
        """
        Checks that intensity distribution is not lost by edges. It can be executed after a RS or BPM propagation.

        Args:
            has_draw (bool): Draws the intensity
            normalized (bool): Draws it normalized

        Returns:
            (np.array): array with intensity I(z)
        """

        intensity_prof = np.sum((np.abs(self.u)**2), axis=0)
        I_max = intensity_prof[0]
        if normalized is True:
            intensity_prof = intensity_prof / I_max
        if has_draw is True:
            plt.figure()
            plt.plot(self.z / mm, intensity_prof, 'k')
            plt.grid()
            plt.xlabel(r"$z\,(mm)$")
            plt.ylabel(r"$I(z)$")
            plt.ylim(bottom=0)

        return intensity_prof


    @check_none('x','z','n')
    def detect_index_variations(self, n_edge: float, incr_n: float = 0.1):
        """In a XZ masks, detects refractive index variations.

        Parameteres:
            n_edge (float):
            incr_n (float): refractive index variation to detect

        Returns:
            x_lens_l (np.array): x for left edge.
            h_lens_l (np.array): h for left edge.
            x_lens_r (np.array): x for right edge.
            h_lens_r (np.array): h for right edge.
        """
        z_new = self.z
        x_new = self.x

        iborders = np.real(self.n) > n_edge
        iborders = iborders + 0.0000001

        # surface detection
        diff1a = np.diff(iborders, axis=1)

        # each size
        ix_l, iz_l = (diff1a > incr_n).nonzero()
        ix_r, iz_r = (diff1a < -incr_n).nonzero()

        x_lens_l = x_new[ix_l]
        h_lens_l = z_new[iz_l]

        x_lens_r = x_new[ix_r]
        h_lens_r = z_new[iz_r]
        return x_lens_l, h_lens_l, x_lens_r, h_lens_r


    @check_none('x','z','n')
    def _detect_transitions_(self, min_variation: float = 1e-10):
        """Detects transitions areas and algorithms between RS and BPM.

        Args:
            min_variation (float): min index variation to detect

        Returns:
            (list floats) : z_transitions, positions z of transitions
            (list str) : algorithms, "RS" or "BPM"
            (list floats) : refr_index_RS, refractive indexes for RS
        """
        # para estar seguros que cogemos bien BPM y no empezamos tarde
        dz_bpm = 25*um

        variation = np.std(np.abs(self.n), axis=0)

        z_transitions = [self.z[0]]
        num_transition = 0

        # comprobamos inicio si es RS o BPM
        if variation[0] < 1e-12:
            # print(("init {} - {}".format(variation[0], self.z[0])))
            algorithm = ['RS']
            refr_index_RS = [self.n[0, 0]]
        else:
            # print(("init {} - {}".format(variation[0], self.z[0])))
            algorithm = ['BPM']
            refr_index_RS = [-1]

        # print((self.z[1:]))
        for i, zi in enumerate(self.z[1:]):

            if i == len(self.z) - 2:  # for the last one if RS. store last one
                # print("a, the last one")
                # print((algorithm[-1]))
                # hay que calcular de todas formas, al menos una línea nueva
                num_transition = num_transition + 1
                z_transitions.append(zi)
                algorithm.append('RS')
                refr_index_RS.append(self.n[0, i])

            elif algorithm[
                    num_transition] == 'RS' and variation[i] > min_variation:
                # create new transition
                # print(("c {} - {} -> BPM".format(variation[i], self.z[i])))
                num_transition = num_transition + 1
                z_transitions.append(self.z[i] - dz_bpm)
                algorithm.append('BPM')
                refr_index_RS.append(-1)

            elif algorithm[
                    num_transition] == 'BPM' and variation[i] < min_variation:
                # create new transition
                # print(("d {} - {} -> RS".format(variation[i], self.z[i])))
                num_transition = num_transition + 1
                z_transitions.append(self.z[i] + dz_bpm)
                algorithm.append('RS')
                refr_index_RS.append(self.n[0, i])

        return z_transitions, algorithm, refr_index_RS


    @check_none('x','z','n')
    def surface_detection(self,
                          mode: int = 1,
                          min_incr: float = 0.1,
                          has_draw: bool = False):# -> tuple[ndarray[Any, dtype[float[Any]]] | Any, ndarray[A...:
        """detect edges of variation in refractive index.

        Args:
            mode (int): 1 or 2, algorithms for surface detection: 1-gradient, 2-diff
            min_incr (float): minimum incremental variation to detect
            has_draw (bool): If True draw.
        """
        n_new = self.n
        z_new = self.z
        x_new = self.x

        diff1 = gradient(np.abs(n_new), axis=0)
        diff2 = gradient(np.abs(n_new), axis=1)

        # if np.abs(diff1 > min_incr) or np.abs(diff2 > min_incr):
        t = np.abs(diff1) + np.abs(diff2)

        ix, iz = (t > min_incr).nonzero()

        self.borders = x_new[iz], z_new[ix]

        if has_draw:
            plt.figure()
            extension = [self.z[0], self.z[-1], self.x[0], self.x[-1]]
            plt.imshow(t.transpose(), extent=extension,
                       aspect='auto', alpha=0.5, cmap='gray')

        return self.borders


    @check_none('x','z','u')
    def draw(self,
             kind: Draw_XZ_Options = 'intensity',
             logarithm: float = 0.,
             normalize: bool = False,
             draw_borders: bool = False,
             filename: str = '',
             scale: str = '',
             min_incr: float = 0.0005,
             reduce_matrix: str = 'standard',
             colorbar_kind: str | None = None,
             colormap_kind: str = "",
             z_scale: str = 'um',
             edge_matrix: NDArrayFloat | None = None,
             interpolation: str = 'bilinear',
             percentage_intensity: float | None = None,
             **kwargs):
        """Draws  XZ field.

        Args:
            kind (str): type of drawing: 'amplitude', 'intensity', 'phase', 'real'
            logarithm (float): If >0, intensity is scaled in logarithm
            normalize (bool): If True, max(intensity)=1
            draw_borders (bool): If True draw edges of objects
            filename (str): if not '' stores drawing in file,
            scale (str): '', 'scaled', 'equal', scales the XY drawing
            min_incr: incrimum increment in refractive index for detecting edges
            reduce_matrix (int, int), 'standard' or False: when matrix is enormous, we can reduce it only for drawing purposes. If True, reduction factor
            z_scale (str): 'mm', 'um'
            edge_matrix (numpy.array): positions of borders
            interpolation(str): methods = [None, 'none', 'nearest', 'bilinear', 'bicubic', 'spline16', 'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric', 'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos']
            percentage_intensity (None or number): If None it takes from CONF_DRAWING['percentage_intensity'], else uses this value
        """

        if reduce_matrix is False:
            amplitude, intensity, phase = field_parameters(self.u, True)

        elif reduce_matrix == 'standard':
            num_x = len(self.x)
            num_z = len(self.z)
            reduction_x = int(num_x/2000)
            reduction_z = int(num_z/2000)

            if reduction_x == 0:
                reduction_x = 1
            if reduction_z == 0:
                reduction_z = 1
            u_new = self.u[::reduction_x, ::reduction_z]
            amplitude, intensity, phase = field_parameters(u_new, True)
        else:
            u_new = self.u[::reduce_matrix[0], ::reduce_matrix[1]]
            amplitude, intensity, phase = field_parameters(u_new, True)

        if z_scale == 'um':
            extension = [self.z[0], self.z[-1], self.x[0], self.x[-1]]
        elif z_scale == 'mm':
            extension = [
                self.z[0] / mm, self.z[-1] / mm, self.x[0], self.x[-1]
            ]

        if percentage_intensity is None:
            percentage_intensity = percentage_intensity_config

        plt.figure()

        if kind == 'intensity':
            I_drawing = intensity
            I_drawing = normalize_draw(I_drawing, logarithm, normalize)
        elif kind == 'amplitude':
            I_drawing = amplitude
            I_drawing = normalize_draw(I_drawing, logarithm, normalize)
        elif kind == 'phase':
            phase = phase/degrees
            phase[intensity < percentage_intensity * (intensity.max())] = 0

            I_drawing = phase
        elif kind == 'real':
            I_drawing = np.real(self.u)
        else:
            print("bad kind parameter")
            return

        h1 = plt.imshow(I_drawing.transpose(),
                        interpolation=interpolation,
                        aspect='auto',
                        origin='lower',
                        extent=extension,
                        **kwargs)

        if z_scale == 'um':
            plt.xlabel(r'z ($\mu m$)')
        elif z_scale == 'mm':
            plt.xlabel(r'z (mm)')

        plt.ylabel(r'x ($\mu m$)')

        if colormap_kind in ('', [], None, True):
            if kind == 'intensity':
                colormap_kind = self.CONF_DRAWING["color_intensity"]
            if kind == 'amplitude':
                colormap_kind = self.CONF_DRAWING["color_amplitude"]
            if kind == 'phase':
                colormap_kind = self.CONF_DRAWING["color_phase"]
            if kind == 'real':
                colormap_kind = self.CONF_DRAWING["color_real"]

        if kind == 'intensity':
            climits = I_drawing.min(), I_drawing.max()
        if kind == 'amplitude':
            climits = -I_drawing.max(), I_drawing.max()
        if kind == 'phase':
            climits = -180, 180
        if kind == 'real':
            climits = -I_drawing.max(), I_drawing.max()

        plt.axis(extension)

        if colorbar_kind not in (False, '', [], None):
            plt.colorbar(orientation=colorbar_kind, shrink=0.66)

        h1.set_cmap(colormap_kind)  # OrRd # Reds_r gist_heat
        plt.clim(climits)

        if scale != '':
            plt.axis(scale)

        if draw_borders is True:
            if edge_matrix is None:
                self.surface_detection(1, min_incr, has_draw=False)
                border0 = self.borders[0]
                border1 = self.borders[1]
            else:
                border0, border1 = edge_matrix

            plt.plot(border1, border0, 'w.', ms=.5)

        if filename != '':
            plt.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0.1)

        return h1


    @check_none('x','z','n')
    def draw_refractive_index(self,
                              kind: Draw_refractive_index_Options = 'all',
                              draw_borders: bool = False,
                              title: str = '',
                              filename: str = '',
                              scale: str = '',
                              min_incr: float = 0.01,
                              reduce_matrix: str = 'standard',
                              colorbar_kind: str | None = None,
                              colormap_kind: str | str = cm.Blues,
                              edge_matrix: NDArrayFloat | None = None
                              ):
        """Draws refractive index.

        Args:
            kind (str): 'all', 'real', 'imag'
            draw_borders (bool): If True draw edges of objects
            filename (str): if not '' stores drawing in file,
            title (str): title of drawing
            scale (str): '', 'scaled', 'equal', scales the XY drawing
            min_incr: minimum increment in refractive index for detecting edges
            reduce_matrix (int, int), 'standard' or False: when matrix is enormous, we can reduce it only for drawing purposes. If True, reduction factor
            edge_matrix (numpy.array): positions of borders
        """

        plt.figure()
        extension = [self.z[0], self.z[-1], self.x[0], self.x[-1]]

        if kind == 'all':
            n_draw = np.abs(self.n)
        elif kind == 'real':
            n_draw = np.real(self.n)
            n_draw[np.abs(np.imag(self.n)) > 0] = self.n_background
        elif kind == 'imag':
            n_draw = np.imag(self.n)

        if reduce_matrix is False:
            h1 = plt.imshow(n_draw.transpose(),
                            interpolation='bilinear',
                            aspect='auto',
                            origin='lower',
                            extent=extension)
        elif reduce_matrix == 'standard':
            num_x = len(self.x)
            num_z = len(self.z)
            reduction_x = int(num_x/2000)
            reduction_z = int(num_z/2000)

            if reduction_x == 0:
                reduction_x = 1
            if reduction_z == 0:
                reduction_z = 1
            n_new = n_draw[::reduction_z, ::reduction_x]
            h1 = plt.imshow(n_new.transpose(),

                            # if self.borders is None or edge_matrix is None:
                            #     self.surface_detection(1, min_incr, reduce_matrix)
                            #     border0 = self.borders[0]
                            #     border1 = self.borders[1]
                            # if edge_matrix is not None:
                            #     border0, border1 = edge_matrix          interpolation='bilinear',
                            aspect='auto',
                            origin='lower',
                            extent=extension)
        else:
            n_new = n_draw[::reduce_matrix[0], ::reduce_matrix[1]]
            h1 = plt.imshow(n_new.transpose(),
                            interpolation='bilinear',

                            # if self.borders is None or edge_matrix is None:
                            #     self.surface_detection(1, min_incr, reduce_matrix)
                            #     border0 = self.borders[0]
                            #     border1 = self.borders[1]
                            # if edge_matrix is not None:
                            #     border0, border1 = edge_matrix          aspect='auto',
                            origin='lower',
                            extent=extension)

        plt.xlabel(r'z ($\mu m$)')
        plt.ylabel(r'x ($\mu m$)')
        plt.title(title)

        plt.axis(extension)
        h1.set_cmap(colormap_kind)  # flag OrRd # Reds_r gist_heat # gist_heat

        if colorbar_kind not in (False, '', None):
            plt.colorbar(orientation=colorbar_kind, shrink=0.66)

        if scale != '':
            plt.axis(scale)

        if draw_borders is True:
            if self.borders is None or edge_matrix is None:
                self.surface_detection(1, min_incr, has_draw=False)
                border0 = self.borders[0]
                border1 = self.borders[1]
            if edge_matrix is not None:
                border0, border1 = edge_matrix
            plt.plot(border1, border0, 'c.', ms=.25)

        if filename != '':
            plt.savefig(filename, dpi=100, bbox_inches='tight', pad_inches=0.1)

        return h1


    @check_none('x')
    def draw_incident_field(self,
                            kind: Draw_X_Options = 'intensity',
                            logarithm: float = 0.,
                            normalize: bool = False,
                            filename: str = ''):
        """Draws incident field self.u0

        Args:
            kind (str): type of drawing: 'amplitude', 'intensity', 'field', 'phase', 'fill', 'fft'
            logarithm (float): If >0, intensity is scaled in logarithm
            normalize (bool): If True, max(intensity)=1
            filename (str): if not '' stores drawing in file,
        """

        u_inc = Scalar_field_X(x=self.x,
                               wavelength=self.wavelength,
                               n_background=1,
                               info="incident_field")
        u_inc.u = self.u0.u
        u_inc.draw(kind, logarithm, normalize, None, filename)


    @check_none('x','z','u')
    def profile_longitudinal(self,
                             kind: str = 'intensity',
                             x0: float = 0*um,
                             logarithm: float = 0.,
                             normalize: bool = False,
                             z_scale: str = 'um',
                             has_draw: bool = True,
                             filename: str = ''):
        """Determine and draws longitudinal profile

        Args:
            kind (str): type of drawing: 'amplitude', 'intensity', 'phase', 'refractive_index'
            x0 (float): profile that passes through x=x0
            logarithm (float): If >0, intensity is scaled in logarithm
            normalize (str):  False, 'maximum', 'intensity', 'area'
            has_draw (bool): If True, draws, False only returns profile
            filename (str): if not '' stores drawing in file

        Returns:
            numpy.array: profile
        """

        imenor, _, _ = nearest(vector=self.x, number=x0)

        if z_scale == 'um':
            factor = 1
            xlabel = 'z (um)'

        elif z_scale == 'mm':
            factor = 1000
            xlabel = 'z (mm)'

        if kind == 'refractive_index':
            n_profile = np.abs(self.n[:, imenor])
            I_drawing = n_profile
        else:
            u = np.squeeze(self.u[:, imenor])
            I_drawing = prepare_drawing(u, kind, logarithm, normalize)
            amplitude, intensity, phase = field_parameters(u)

        if has_draw is True:
            plt.figure(facecolor='w', edgecolor='k')
            plt.plot(self.z / factor, I_drawing, 'k', linewidth=2)  # 'k-o'
            plt.axis([
                self.z[0] / factor, self.z[-1] / factor,
                I_drawing.min(),
                I_drawing.max()
            ])

            plt.xlabel(xlabel)
            plt.ylabel('I(z, x = {:2.2f} $\mu$m)'.format(x0))

            if filename != '':
                plt.savefig(filename,
                            dpi=100,
                            bbox_inches='tight',
                            pad_inches=0.1)

        if kind == 'intensity':
            output = intensity
        elif kind == 'amplitude':
            output = amplitude
        elif kind == 'phase':
            output = phase
        elif kind == 'refractive_index':
            output = n_profile
        else:
            output = None
        return output


    @check_none('x','z','u')
    def profile_transversal(self,
                            kind: str = 'intensity',
                            z0: float = 0*um,
                            logarithm: float = 0.,
                            normalize: bool = False,
                            has_draw: bool = True,
                            filename: str = ''):
        """Determine and draws transversal profile.

        Args:
            kind (str): type of drawing:  'amplitude', 'intensity', 'phase', 'refractive_index'
            z0 (float): profile that passes through z=z0
            logarithm (float): If >0, intensity is scaled in logarithm
            normalize (str):  False, 'maximum', 'intensity', 'area'
            draw (bool): If True, draws, False only returns profile
            filename (str): if not '' stores drawing in file,

        Returns:
            numpy.array: profile

        TODO: Include interpolation
        """

        imenor, _, _ = nearest(vector=self.z, number=z0)

        if kind == 'refractive_index':
            n_profile = np.abs(self.n[imenor, :])
            I_drawing = n_profile
        else:
            u = np.squeeze(self.u[imenor, :])
            I_drawing = prepare_drawing(u, kind, logarithm, normalize)
            amplitude, intensity, phase = field_parameters(u)

        if has_draw is True:
            plt.figure(facecolor='w', edgecolor='k')
            plt.plot(self.x, I_drawing, 'k', linewidth=2)  # 'k-o'
            plt.axis([self.x[0], self.x[-1], I_drawing.min(), I_drawing.max()])
            texto = 'I(z=%d um, x)' % (z0)
            plt.xlabel(r'x ($\mu$m)')
            plt.ylabel(texto)

            if filename != '':
                plt.savefig(filename,
                            dpi=100,
                            bbox_inches='tight',
                            pad_inches=0.1)

        if kind == 'intensity':
            output = intensity
        elif kind == 'amplitude':
            output = amplitude
        elif kind == 'phase':
            output = phase
        elif kind == 'refractive_index':
            output = n_profile
        else:
            output = None
        return output


    @check_none('x','z','u')
    def search_focus(self, verbose=True):
        """Search for location of maximum.

        Args:
            kind (str): type of drawing: 'amplitude', 'intensity', 'phase', 'refractive_index'
            x0 (float): profile that passes through x=x0
            logarithm (float): If >0, intensity is scaled in logarithm
            normalize (str):  False, 'maximum', 'intensity', 'area'
            draw (bool): If True, draws, False only returns profile
            filename (str): if not '' stores drawing in file,

        Returns:
            (x,z): positions of focus
        """
        intensity = np.abs(self.u)**2

        iz, ix = np.unravel_index(intensity.argmax(), intensity.shape)
        if verbose is True:
            print(("x = {:2.3f} um, z = {:2.3f} um".format(
                self.x[ix], self.z[iz])))
        return self.x[ix], self.z[iz]


    @check_none('x','z','u')
    def beam_widths(self,
                    kind='FWHM1D',
                    has_draw: tuple[bool] = [True, False],
                    z_scale: str = 'um',
                    percentage: float = 0.5,
                    remove_background: str | None = None,
                    verbose: bool = False):
        """Computes the beam width for all the distances z.
        Args:
            kind (str): kind of algorithm: 'sigma4', 'FWHM1D'
            has_draw (bool, bool): First for complete analysis, second for all FWHM2D computations
            percentage (float): (0-1) percentage for the beam detection
            remove_background=None,
            verbose (bool): prints info

        Returns:
            (numpy.array) widths:  for each distance z
            (numpy.array) positions_center: positions of centers for each z
        """

        if z_scale == 'um':
            factor = 1
            xlabel = 'z (um)'

        elif z_scale == 'mm':
            factor = 1000
            xlabel = 'z (mm)'

        widths = np.zeros_like(self.z)
        positions_center = np.zeros_like(self.z)

        for i, zi in enumerate(self.z):
            field = np.abs(self.u[i, :])
            if kind == 'sigma4':
                widths[i], positions_center[i] = beam_width_1D(field, self.x)
            elif kind == 'FWHM1D':
                intensity = np.abs(field)**2
                widths[i] = FWHM1D(self.x,
                                   intensity,
                                   percentage=percentage,
                                   remove_background=remove_background,
                                   has_draw=has_draw[1])

        if has_draw[0] is True:
            plt.figure()
            plt.plot(self.z / factor, widths, 'r', label='axis')
            plt.xlabel(xlabel)
            plt.ylabel("widths ($\mu$m)")
            plt.legend()

        if verbose:
            print("width = {:2.2f} um".format(widths))

        return widths, positions_center


    @check_none('x','z','u')
    def video(self,
              kind: str = 'intensity',
              z_min: float | None = None,
              z_max: float | None = None,
              logarithm: float = 0.,
              normalize: bool = False,
              time_video: float = 10 * seconds,
              frames_reduction: int = 5,
              filename: str = 'video.avi',
              dpi: int = 100):
        """Generates a video in the z dimension.

        Args:
            kind (str):
            z_min (float):
            z_max (float):
            logarithm (float):
            normalize (bool):
            time_video (float):
            frames_reduction (int):
            filename (str):
            dpi (int):
        """

        I_drawing = prepare_drawing(self.u, kind, logarithm, normalize)
        if z_min is None:
            z_min = self.z[0]
        if z_max is None:
            z_max = self.z[-1]

        imin, _, _ = nearest(self.z, z_min)
        imax, _, _ = nearest(self.z, z_max)

        fig = plt.figure()
        ax = fig.add_subplot(111, autoscale_on=False)
        ax.grid()
        hdl_line, = ax.plot([], [], 'k', lw=2)
        ax.set_title('', transform=ax.transAxes)
        plt.xlim(self.x[0], self.x[-1])
        plt.ylim(I_drawing.min(), I_drawing.max())

        def init():
            hdl_line.set_data([], [])
            ax.set_title('')
            return hdl_line

        def animate(i):

            hdl_line.set_data(self.x, I_drawing[i, :])
            ax.set_title("$z = {:2.0f} \mu m$".format(self.z[i]))
            return i

        ani = animation.FuncAnimation(fig,
                                      animate,
                                      list(range(imin, imax,
                                                 frames_reduction)),
                                      interval=25,
                                      blit=False,
                                      init_func=init)

        fps = int(len(self.z) / (time_video * frames_reduction))

        ani.save(filename, fps=fps, dpi=dpi)
        plt.close()


    @check_none('x','z','u')
    def draw_profiles_interactive(self,
                                  kind: Draw_interactive_Options = 'intensity',
                                  logarithm: float = 0.,
                                  normalize: bool = False):
        """Draws profiles interactivey. Only transversal

        Args:
            kind (str): 'intensity', 'amplitude', 'phase'
            logarithm (float): If >0, intensity is scaled in logarithm
            normalize (bool): If True, max(intensity)=1
        """

        global l2a, zZ, I_drawing, z, h1, x, log1, norm1
        plt.figure()
        h1, = plt.plot([self.z[0], self.z[0]], [self.x[0], self.x[-1]],
                       lw=2,
                       color='w')

        I_drawing = prepare_drawing(self.u, kind, logarithm, normalize)

        log1 = logarithm
        norm1 = normalize

        z_actual = self.z[0]
        x = self.x
        ix, iz = np.unravel_index(I_drawing.argmax(), I_drawing.shape)

        extension = [self.x[0], self.x[-1], I_drawing.min(), I_drawing.max()]

        imenor, value, distance = nearest(vector=self.z, number=z_actual)
        I_drawing_actual = np.squeeze(I_drawing[imenor, :])

        z = self.z

        plt.subplots_adjust(left=0.15, bottom=0.25)
        plt.axis(extension)
        l2a, = plt.plot(self.x, I_drawing_actual, lw=2, color='k')
        axcolor = 'lightgoldenrodyellow'
        axS = plt.axes([0.15, 0.15, 0.75, 0.03], facecolor=axcolor)
        zZ = plt.Slider(axS,
                        'z (um)',
                        self.z[0],
                        self.z[-1],
                        valinit=self.z.min())
        zZ.on_changed(__update__)


def __update__(val: float):
    """for making videos.
    """
    zz = zZ.val
    i_z_min, value, distance = nearest(vector=z, number=zz)
    I_drawing_profile = np.squeeze(I_drawing[i_z_min, :])

    I_drawing_profile = normalize_draw(I_drawing_profile, log1, norm1)
    l2a.set_ydata(I_drawing_profile)
    plt.draw()
