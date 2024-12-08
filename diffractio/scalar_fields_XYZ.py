# !/usr/bin/env python3


# ----------------------------------------------------------------------
# Name:        scalar_fields_XYZ.py
# Purpose:     Defines the Scalar_field_XYZ class and related functions
#
# Author:      Luis Miguel Sanchez Brea
#
# Created:     2024
# Licence:     GPLv3
# ----------------------------------------------------------------------


"""
This module generates Scalar_field_XYZ class and several functions for multiprocessing.
It is required also for generating masks and fields.
The main atributes are:
    * self.u - field XYZ
    * self.x - x positions of the field
    * self.y - y positions of the fieldalgorithm
    * self.z - z positions of the field
    * self.wavelength - wavelength of the incident field. The field is monochromatic
    * self.u0 - initial field at z=z0
    * self.n_background - background refractive index
    * self.n - refractive index


The magnitude is related to microns: `micron = 1.`

*Class for unidimensional scalar fields*

*Definition of a scalar field*
    * instantiation, duplicate
    * load and save data
    * to_Scalar_field_XY
    * xy_2_xyz
    * cut_function
    * __rotate__
    * __rotate_axis__

*Propagation*
    * RS - Rayleigh Sommerfeld
    * RS_amplification
    * BPM - Beam Propagation method

*Drawing functions*
    * draw_XYZ
    * draw_XY
    * draw_XZ
    * draw_volume
    * draw_refractive_index
    * video

"""

# flake8: noqa

import copy
import copyreg
import os
import sys
import time
import types
from multiprocessing import Pool

import matplotlib.animation as anim
from numpy import cos, diff, gradient, sin
from scipy.fftpack import fft2, ifft2
from scipy.interpolate import RegularGridInterpolator

from .__init__ import degrees, mm, np, plt, num_max_processors
from .config import bool_raise_exception, CONF_DRAWING, Draw_pyvista_Options, video_isovalue_Options, get_scalar_options
from .config import  Draw_XZ_Options, Draw_XY_Options, Draw_XYZ_Options
from .utils_common import get_date, load_data_common, save_data_common, check_none, oversampling, get_scalar, add, sub, rmul
from .utils_drawing import normalize_draw
from .utils_math import get_k, nearest, reduce_to_1
from .utils_multiprocessing import _pickle_method, _unpickle_method
from .utils_optics import FWHM2D, beam_width_2D, field_parameters, normalize_field
from .utils_drawing3D import draw, video_isovalue
from .utils_typing import NDArrayFloat
from .scalar_fields_XY import PWD_kernel, Scalar_field_XY, WPM_schmidt_kernel
from .scalar_fields_XZ import Scalar_field_XZ
from .vector_fields_XZ import Vector_field_XZ
from .vector_fields_XY import Vector_field_XY

copyreg.pickle(types.MethodType, _pickle_method, _unpickle_method)

class Scalar_field_XYZ():
    """Class for 3D scalar fields.

    Args:
        u0 (Scalar_field_XY): Initial scalar field. wavelength, and x, y arrays are obtained from this field.
        z (numpy.array): linear array with equidistant positions.
        n_background (float): refractive index of background
        info (str): String with info about the simulation

    Attributes:
        self.x (numpy.array): linear array with equidistant positions. The number of data is preferibly :math:`2^n` .
        self.y (numpy.array): linear array with equidistant positions. The number of data is preferibly :math:`2^n` .
        self.z (numpy.array): linear array with equidistant positions. The number of data is preferibly :math:`2^n` .
        self.u (numpy.array): equal size than X. complex field
        self.wavelength (float): wavelength of the incident field.
        self.u0 (Scalar_field_XY): Initial XY field
        self.n_background (float): background refractive index.
        self.n (numpy.array): refractive index. Same dimensions than self.u.
    """

    def __init__(self, x: NDArrayFloat | None = None, y: NDArrayFloat | None = None,
                 z: NDArrayFloat | None = None, wavelength: float | None = None,
                 n_background: float = 1., info: str = ""):
        self.x = x
        self.y = y
        self.z = z
        self.wavelength = wavelength
        self.n_background = n_background
        self.fast = True
        self.quality = 0
        self.borders = None
        self.CONF_DRAWING = CONF_DRAWING

        if x is not None and z is not None:
            self.X, self.Y, self.Z = np.meshgrid(x, y, z)
            self.u0 = Scalar_field_XY(x, y, wavelength)
            self.u = np.zeros_like(self.X, dtype=complex)
            self.n = n_background * np.ones_like(self.X, dtype=complex)
        else:
            self.X = None
            self.Y = None
            self.Z = None
            self.u0 = None
            self.u = None
            self.n = None
        self.info = info
        self.reduce_matrix = 'standard'  # 'None, 'standard', (5,5)
        self.type = 'Scalar_field_XYZ'
        self.date = get_date()


    def xy_2_xyz(self, u0_XY, z: NDArrayFloat):
        """Similar to Init. send a Scalarfield_XY and passes to XYZ.

        Args:
            u0_XY (Scalar_field_XY): init field
            z (numpy.array): array with z positions
        """
        u0 = u0_XY[0]
        self.x = u0.x
        self.y = u0.y
        self.z = z

        self.wavelength = u0.wavelength
        self.u0 = u0
        self.amplification = 1
        self.X, self.Y, self.Z = np.meshgrid(self.x, self.y, self.z)

        self.u = np.zeros_like(self.X, dtype=complex)
        self.n = np.ones(np.shape(self.X), dtype=float)

        for i in range(len(self.z)):
            self.u[:, :, i] = u0_XY[i].u

    @check_none('x','y','z','u',raise_exception=bool_raise_exception)
    def __str__(self):
        """Represents main data."""

        Imin = (np.abs(self.u)**2).min()
        Imax = (np.abs(self.u)**2).max()
        phase_min = (np.angle(self.u)).min()/degrees
        phase_max = (np.angle(self.u)).max()/degrees
        print("{}\n - x:  {},   y:  {},  z:  {},   u:  {}".format(
            self.type, self.x.shape, self.y.shape, self.z.shape, self.u.shape))
        print(
            " - xmin:       {:2.2f} um,  xmax:      {:2.2f} um,  Dx:   {:2.2f} um"
            .format(self.x[0], self.x[-1], self.x[1] - self.x[0]))
        print(
            " - ymin:       {:2.2f} um,  ymax:      {:2.2f} um,  Dy:   {:2.2f} um"
            .format(self.y[0], self.y[-1], self.y[1] - self.y[0]))
        print(
            " - zmin:       {:2.2f} um,  zmax:      {:2.2f} um,  Dz:   {:2.2f} um"
            .format(self.z[0], self.z[-1], self.z[1] - self.z[0]))
        print(" - Imin:       {:2.2f},     Imax:      {:2.2f}".format(
            Imin, Imax))
        print(" - phase_min:  {:2.2f} deg, phase_max: {:2.2f} deg".format(
            phase_min, phase_max))
        print(" - wavelength: {:2.2f} um".format(self.wavelength))
        print(" - date:       {}".format(self.date))
        if self.info != "":
            print(" - info:       {}".format(self.info))
        return ("")
    
    
    @check_none('x','y','z','u',raise_exception=bool_raise_exception)
    def __add__(self, other):
        """Adds two Scalar_field_x. For example two light sources or two masks.

        Args:
            other (Scalar_field_X): 2nd field to add
            kind (str): instruction how to add the fields: ['source', 'mask', 'phases', 'distances', 'no_overlap].

        Returns:
            Scalar_field_X: `u3 = u1 + u2`
        """

        if self.type == 'Scalar_mask_XYZ':
            t = add(self, other, kind='refractive_index')
        elif self.type == 'Scalar_source_XYZ' or 'Scalar_field_XYZ':
            t = add(self, other, kind='source')
            
        return t

    def add(self, other, kind):
        """Adds two Scalar_field_xy. For example two light sources or two masks.

        Args:
            other (Scalar_field_XY): 2nd field to add
            kind (str): instruction how to add the fields: ['source', 'mask', 'phases', 'distances', 'no_overlap].
            - 'source': adds the fields as they are
            - 'mask': adds the fields as complex numbers and then normalizes so that the maximum amplitude is 1.
            - 'phases': adds the phases and then normalizes so that the maximum amplitude is 1.
            - 'np_overlap': adds the fields as they are. If the sum of the amplitudes is greater than 1, an error is produced
            - 'distances': adds the fields as they are. If the fields overlap, the field with the smallest distance is kept.


        Returns:
            Scalar_field_XY: `u3 = u1 + u2`
        """

        t = add(self, other, kind)
        return t
    
    
    @check_none('x','y','z','u',raise_exception=bool_raise_exception)
    def __sub__(self, other):
        """Substract two Scalar_field_XYZ For example two light sources or two masks.

        Args:
            other (Scalar_field_XYZ): field to substract

        Returns:
            Scalar_field_X: `u3 = u1 - u2`

        # TODO: It can be improved for maks (not having less than 1)
        """


        if self.type == 'Scalar_mask_XYZ':
            t = sub(self, other, kind='refractive_index')
        elif self.type == 'Scalar_source_XYZ' or 'Scalar_field_XYZ':
            t = sub(self, other, kind='source')
            
        return t



    @check_none('x','y','z','u',raise_exception=bool_raise_exception)
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

        if self.type == 'Scalar_mask_XYZ':
            t = rmul(self, number, kind='intensity')
        elif self.type == 'Scalar_field_XYZ':
            t = rmul(self, number, kind='amplitude')
            
        return t


    @check_none('x','y','z','u',raise_exception=bool_raise_exception)
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


    @check_none('X','Y','Z','u',raise_exception=bool_raise_exception)
    def __XYZ_rotate_point__(self, angles: tuple[float,float,float], point: tuple[float,float,float]):
        """Function to rotate around any of the 3 axis of rigid solid.

        Args:
            psi (float): Euler angle in radians
            phi (float): Euler angle in radians
            sigma (float): Euler angle in radians

        Returns:
            tuple[ NDarray, NDarray, NDarray]: Xrot, Yrot, Zrot
        
        References:
            http://estudiarfisica.wordpress.com/2011/03/17/ampliacion-del-solido-rigido-matrices-de-rotation-angles-y-transformaciones-de-euler-velocidad-angular-momento-angular-tensor-de-inercia-teorema-de-steiner-utilsizado/
        """
        
        psi, phi, sigma = angles
        x0, y0, z0 = point
                
        cp = cos(psi)
        sp = sin(psi)
        cf = cos(phi)
        sf = cos(phi)
        cs = cos(sigma)
        ss = sin(sigma)

        Xrot = (self.X-x0) * (cp*cf-sp*cs*sf)  + (self.Y-y0) * (cp*sf+sp*cs*cf)  + (self.Z-z0) * (sp*ss)
        Yrot = (self.X-x0) * (-sp*cf-cp*cs*sf) + (self.Y-y0) * (-sp*sf+cp*cs*cf) + (self.Z-z0) * (cp*ss)
        Zrot = (self.X-x0) * (ss*sf)           + (self.Y-y0) * (-ss*cf)          + (self.Z-z0) * (cs)
        
        return Xrot, Yrot, Zrot


    @check_none('X','Y','Z','u',raise_exception=bool_raise_exception)
    def __XYZ_rotate_axis__(self,  angle: float, point: tuple[float, float, float], axis: tuple[float, float, float]):
        """rotate around an axis.

        Args:
            axis (float, float, float): direction of the axis
            angle (float): angle of rotation in radians
            
        Returns:
            tuple[ NDarray, NDarray, NDarray]: Xrot, Yrot, Zrot
        """
        # normalized axis
        u, v, w = axis / np.sqrt(axis[0]**2 + axis[1]**2 + axis[2]**2)
        x0, y0, z0 = point


        ct = cos(angle)
        st = sin(angle)

        Xrot = (self.X-x0) * (u**2 + (v**2 + w**2) * ct) + (self.Y-y0) * (
            u * v * (1 - ct) - w * st) + (self.Z-z0) * (u * w * (1 - ct) + v * st)
        Yrot = (self.X-x0) * (u * v * (1 - ct) + w * st) + (self.Y-y0) * (
            v**2 + (u**2 + w**2) * ct) + (self.Z-z0) * (v * w * (1 - ct) - u * st)
        Zrot = (self.X-x0) * (u * w * (1 - ct) - v * st) + (self.Y-y0) * (
            v * w * (1 - ct) + u * st) + (self.Z-z0) * (w**2 + (u**2 + v**2) * ct)

        return Xrot, Yrot, Zrot



    @check_none('X','Y','Z','u',raise_exception=bool_raise_exception)
    def __XYZ_rotate__(self, rotation: dict):
        """Rotates the field around an axis or a point.
        
        Args:
            rotation (dict): kind: 'axis' or 'point'
                             if 'axis': angle (float) and axis (tuple[float,float,float])
                             if 'point': angle (tuple[float,float,float]) and point (tuple[float,float,float])
                             
        Returns:
            tuple[ NDarray, NDarray, NDarray]: Xrot, Yrot, Zrot
        """
        
        if rotation['kind'] == 'axis':
            axis = rotation['axis']
            angle = rotation['angle']
            point = rotation['point']
        elif rotation['kind'] == 'point':
            point = rotation['point']
            angles = rotation['angle']

        if rotation['kind'] == 'axis' and angle != 0:
            Xrot, Yrot, Zrot = self.__XYZ_rotate_axis__(angle, point, axis)
        elif rotation['kind'] == 'point':
            Xrot, Yrot, Zrot = self.__XYZ_rotate_point__(angles, point)
        else:
            Xrot=self.X
            Yrot=self.Y
            Zrot=self.Z
            
        return Xrot, Yrot, Zrot
            
    def size(self, verbose: bool = False):
        """returns the size of the instance in MB.

        Args:
            verbose (bool, optional): prints size in Mb. Defaults to False.

        Returns:
            float: size in MB
        """

        return get_instance_size_MB(self, verbose)

        
    @check_none('u',raise_exception=bool_raise_exception)
    def conjugate(self, new_field: bool = True):
        """Congugates the field.

        Args:
            new_field (bool, optional): Generates a new field. Defaults to True.

        Returns:
            _type_: _description_
        """

        if new_field is True:
            u_new = self.duplicate()
            u_new.u = np.conj(self.u)
            return u_new
        else:
            self.u = np.conj(self.u)


    def normalize(self, kind='amplitude', new_field: bool = False):
        """Normalizes the field so that intensity.max()=1.

        Args:
            kind (str): 'amplitude', or 'intensity'
            new_field (bool): If False the computation goes to self.u. If True a new instance is produced

        Returns
            u (numpy.array): normalized optical field
        """
        return normalize_field(self, kind, new_field)


    def duplicate(self, clear: bool = False):
        """Duplicates the instance

        Args:
            clear (bool, optional): If True, clears the field. Defaults to False.

        Returns:
            Scalar_field_XYZ: duplicated field
        """
        new_field = copy.deepcopy(self)
        if clear is True:
            new_field.clear_field()
        return new_field


    def reduce_to_1(self):
        """All the values greater than 1 pass to 1. This is used for Scalar_masks when we add two masks.
        """

        self = reduce_to_1(self)

    @check_none('u',raise_exception=bool_raise_exception)
    def clear_field(self):
        """clear field."""

        self.u = np.zeros(np.shape(self.u), dtype=complex)


    @check_none('X',raise_exception=bool_raise_exception)
    def clear_refractive_index(self):
        """clear refractive index n(x,z)=n_background."""

        self.n = self.n_background * np.ones_like(self.X, dtype=complex)


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
            final_filename = save_data_common(self, filename, add_name,
                                              description, verbose)
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

        if verbose:
            print(dict0.keys())

        if dict0 is not None:
            if isinstance(dict0, dict):
                self.__dict__ = dict0
            else:
                raise Exception('no dictionary in load_data')



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

    @check_none('u',raise_exception=bool_raise_exception)
    def intensity(self):
        """Returns intensity."""

        intensity = (np.abs(self.u)**2)
        return intensity


    @check_none('x','y','z','u')
    def oversampling(self, factor_rate: int | tuple):
        """Overfample function has been implemented in scalar X, XY, XZ, and XYZ frames reduce the pixel size of the masks and fields. 
        This is also performed with the cut_resample function. However, this function oversamples with integer factors.
        
        Args:
            factor_rate (int | tuple, optional): factor rate. Defaults to 2.
        """

        self = oversampling(self, factor_rate)

    @check_none('x','y','z','u',raise_exception=bool_raise_exception)
    def cut_resample(self,
                     x_limits: tuple[float, float] | str = '',
                     y_limits: tuple[float, float] | str = '',
                     z_limits: tuple[float, float] | str = '',
                     num_points: int | None = None,
                     new_field: bool = False,
                     interp_kind=(3, 1)):
        """it cut the field to the range (x0,x1). If one of this x0,x1 positions is out of the self.x range it do nothing. It is also valid for resampling the field, just write x0,x1 as the limits of self.x

        Args:
            x_limits (float,float): (x0,x1) starting and final points to cut, if '' - takes the current limit x[0] and x[-1]
            y_limits (float,float): (y0,y1) - starting and final points to cut, if '' - takes the current limit y[0] and z[-1]
            num_points (int): it resamples x, y and u, where [],'',0,None -> it leave the points as it is new_field (bool): it returns a new Scalar_field_XY
            interp_kind: numbers between 1 and 5
        """
        if x_limits == '':
            x0 = self.x[0]
            x1 = self.x[-1]
        else:
            x0, x1 = x_limits

        if y_limits == '':
            y0 = self.y[0]
            y1 = self.y[-1]
        else:
            y0, y1 = y_limits

        if z_limits == '':
            z0 = self.z[0]
            z1 = self.z[-1]
        else:
            z0, z1 = z_limits

        if x0 < self.x[0]:
            x0 = self.x[0]
        if x1 > self.x[-1]:
            x1 = self.x[-1]

        if y0 < self.y[0]:
            y0 = self.y[0]
        if y1 > self.y[-1]:
            y1 = self.y[-1]

        if z0 < self.z[0]:
            z0 = self.z[0]
        if z1 > self.z[-1]:
            z1 = self.z[-1]

        i_x0, _, _ = nearest(self.x, x0)
        i_x1, _, _ = nearest(self.x, x1)

        i_y0, _, _ = nearest(self.y, y0)
        i_y1, _, _ = nearest(self.y, y1)

        i_z0, _, _ = nearest(self.z, z0)
        i_z1, _, _ = nearest(self.z, z1)

        if num_points not in ([], '', 0, None):
            num_points_x, num_points_y, num_points_z = num_points
            x_new = np.linspace(x0, x1, num_points_x)
            y_new = np.linspace(y0, y1, num_points_y)
            z_new = np.linspace(z0, z1, num_points_z)
            field_n = Scalar_field_XYZ(x=x_new,
                                       y=y_new,
                                       z=z_new,
                                       wavelength=self.wavelength,
                                       n_background=self.n_background)

            X_new = field_n.X
            Y_new = field_n.Y
            Z_new = field_n.Z

            f_interp_amplitude = RegularGridInterpolator(
                (self.x, self.y, self.z), np.abs(self.u))
            f_interp_phase = RegularGridInterpolator((self.x, self.y, self.z),
                                                     np.angle(self.u))
            u_new_abs = f_interp_amplitude((X_new, Y_new, Z_new))
            u_new_phase = f_interp_phase((X_new, Y_new, Z_new))
            u_new = u_new_abs * np.exp(1j * u_new_phase)

            n_interp_real = RegularGridInterpolator((self.x, self.y, self.z),
                                                    np.real(self.n))
            n_interp_imag = RegularGridInterpolator((self.x, self.y, self.z),
                                                    np.imag(self.n))
            n_new_real = n_interp_real((X_new, Y_new, Z_new))
            n_new_imag = n_interp_imag((X_new, Y_new, Z_new))
            n_new = n_new_real + 1j * n_new_imag

        else:
            i_s = slice(i_x0, i_x1)
            j_s = slice(i_y0, i_y1)
            k_s = slice(i_z0, i_z1)
            x_new = self.x[i_s]
            y_new = self.y[j_s]
            z_new = self.z[k_s]
            X_new, Y_new, Z_new = np.meshgrid(x_new, y_new, z_new)
            u_new = self.u[i_s, j_s, k_s]
            n_new = self.n[i_s, j_s, k_s]

        if new_field is False:
            self.x = x_new
            self.y = y_new
            self.z = z_new

            self.X = X_new
            self.Y = Y_new
            self.Z = Z_new

            self.u = u_new
            self.n = n_new

        elif new_field is True:
            field_n = Scalar_field_XYZ(x=x_new,
                                       y=y_new,
                                       z=z_new,
                                       wavelength=self.wavelength,
                                       n_background=self.n_background)
            field_n.u = u_new
            field_n.n = n_new
            return field_n


    def incident_field(self, u0, z0: float | None = None):
        """Incident field for the experiment. It takes a Scalar_source_XYZ field.

        Args:
            u0 (Scalar_source_X): field produced by Scalar_source_XYZ (or a XYZ field)
            z0 (float): position of the incident field. if None, '', [], is at the beginning
        """

        self.u0 = u0

        if u0.x.shape == self.x.shape:
            if z0 in (None, '', []):
                self.u[:, :, 0] = self.u[:, :, 0] + u0.u

            else:
                iz, _, _ = nearest(self.z, z0)
                self.u[:, :, iz] = self.u[:, :, iz] + u0.u


    @check_none('x','y','u',raise_exception=bool_raise_exception)
    def final_field(self):
        """Returns the final field as a Scalar_field_XYZ."""

        u_final = Scalar_field_XY(x=self.x,
                                  y=self.y,
                                  wavelength=self.wavelength,
                                  n_background=1,
                                  info="from final_field at z0= {} um".format(
                                      self.z[-1]))
        u_final.u = self.u[:, :, -1]
        return u_final

    @check_none('z','u',raise_exception=bool_raise_exception)
    def __RS_multiprocessing__(self, i: int):
        """Internal for multiprocessing.

        Args:
            i (int): Number for for loop.
        """
        self.u[:, :, i] = self.u0.RS(amplification=(1, 1),
                                     z=self.z[i],
                                     n=self.n_background,
                                     new_field=False,
                                     matrix=True,
                                     kind='z',
                                     verbose=False)
        return self.u[:, :, i]

    def RS(self, verbose: bool = False, num_processors = 1):
        """Rayleigh Sommerfeld propagation algorithm

        Args:
            verbose (bool): shows the quality of algorithm (>1 good)
            num_processors (int): number of processors for multiprocessing

        Returns:
           time in the processing
        """

        time1 = time.time()
        if num_processors == 1:
            for iz in np.array(range(0, len(self.z))):
                self.u[:, :, iz] = self.u0.RS(amplification=(1, 1),
                                              z=self.z[iz],
                                              n=self.n_background,
                                              new_field=False,
                                              matrix=True,
                                              kind='z',
                                              verbose=verbose)
        else:
            pool = Pool(num_processors)
            t = pool.map(self.__RS_multiprocessing__, list(range(len(self.z))))
            pool.close()
            pool.join()
            for i in range(len(self.z)):
                self.u[:, :, i] = t[i]

        time2 = time.time()
        print(("time in RS= {}. num proc= {}".format(time2 - time1,
                                                     num_processors)))
        return time2 - time1

    @check_none('x','y','z','u',raise_exception=bool_raise_exception)
    def RS_amplification(self, amplification: tuple[int, int] = (1, 1)):
        """Rayleigh Sommerfeld propagation algorithm. it performs amplification

        Args:
            amplification (int,int): number of fields

        Returns:
           Scalar_field_XY:
        """

        x0 = self.x[0]
        y0 = self.y[0]
        x1 = self.x[-1]
        y1 = self.y[-1]
        nx = len(self.x)
        ny = len(self.y)

        Gx = np.linspace(x0 * amplification, x1 * amplification,
                         nx * amplification)
        Gy = np.linspace(y0 * amplification, y1 * amplification,
                         ny * amplification)

        field_output = Scalar_field_XY(Gx, Gy, self.wavelength)

        fieldij = Scalar_field_XYZ(self.x, self.y, self.wavelength)
        for ix in range(0, amplification):
            ixini = ix * nx
            ixfin = (ix + 1) * nx - 1
            xini = Gx[ixini]
            xfin = Gx[ixfin]
            for iy in range(0, amplification):
                iyini = iy * ny
                iyfin = (iy + 1) * ny - 1
                yini = Gy[iyini]
                yfin = Gy[iyfin]

                xij = np.linspace(xini, xfin, nx)
                yij = np.linspace(yini, yfin, ny)

                fieldij.propagacionRS(xij, yij)
                # quality=min(quality,parametros.quality);
                field_output[ixini:ixfin, iyini:iyfin] = fieldij.u

            # parametros.tiempo=tiempoTotal
            # parametros.quality=quality
        return field_output


    @check_none('x','y','z','u',raise_exception=bool_raise_exception)
    def BPM(self, has_edges: bool = True, pow_edge: int = 80, verbose: bool = False):
        """3D Beam propagation method (BPM).

        Args:
            has_edges (bool): If True absorbing edges are used.
            pow_edge (float): If has_edges, power of the supergaussian
            verbose (bool): shows data process by screen


        References:
            Algorithm in "Engineering optics with matlab" pag 119
        """

        k0 = 2 * np.pi / self.wavelength
        numx = len(self.x)
        numy = len(self.y)
        numz = len(self.z)

        deltax = self.x[-1] - self.x[0]
        deltay = self.y[-1] - self.y[0]
        deltaz = self.z[1] - self.z[0]
        # pixelx = np.linspace(-int(numx/2), int(numx/2), numx)
        # pixely = np.linspace(-numy/2, numy/2, numy)

        modo = self.u0.u

        kx1 = np.linspace(0, int(numx/2) + 1, int(numx/2))
        kx2 = np.linspace(-int(numx/2), -1, int(numx/2))
        kx = (2 * np.pi / deltax) * np.concatenate((kx1, kx2))

        ky1 = np.linspace(0, numy/2 + 1, int(numy/2))
        ky2 = np.linspace(-numy/2, -1, int(numy/2))
        ky = (2 * np.pi / deltay) * np.concatenate((ky1, ky2))

        KX, KY = np.meshgrid(kx, ky)

        phase1 = np.exp((1j * deltaz * (KX**2 + KY**2)) / (2 * k0))
        field = np.zeros(np.shape(self.n), dtype=complex)

        if has_edges is False:
            has_filter = np.zeros_like(self.z)
        elif isinstance(has_edges, int):
            has_filter = np.ones_like(self.z)
        else:
            has_filter = has_edges

        width_edge = 0.95*(self.x[-1]-self.x[0])/2
        x_center = (self.x[-1]+self.x[0])/2
        y_center = (self.y[-1]+self.y[0])/2

        filter_x = np.exp(-(np.abs(self.X[:, :, 0]-x_center) / width_edge)**pow_edge)
        filter_y = np.exp(-(np.abs(self.Y[:, :, 0]-y_center) / width_edge)**pow_edge)
        filter_function = filter_x*filter_y

        field[:, :, 0] = modo
        for k in range(0, numz):

            if has_filter[k] == 0:
                filter_edge = 1
            else:
                filter_edge = filter_function

            if verbose is True:
                print("BPM 3D: {}/{}".format(k, numz), end="\r")
            phase2 = np.exp(-1j * self.n[:, :, k] * k0 * deltaz)
            # Aplicamos la Transformada Inversa
            modo = ifft2((fft2(modo) * phase1)) * phase2
            field[:, :, k] = field[:, :, k] + modo * filter_edge
            self.u = field

    @check_none('x','y','z',raise_exception=bool_raise_exception)
    def PWD(self, n: float | None = None, matrix: bool = False, verbose: bool = False):
        """
        Plane wave decompostion algorithm.

        Args:
            n (np. array): refractive index, If None, it is n_background
            verbose (bool): If True prints state of algorithm

        Returns:
            numpy.array(): Field at at distance dz from the incident field

        References:
            1. Schmidt, S. et al. Wave-optical modeling beyond the thin-element-approximation. Opt. Express 24, 30188 (2016).


        Todo:
            include filter for edges
        """

        dx = self.x[1] - self.x[0]
        dy = self.y[1] - self.y[0]
        dz = self.z[1] - self.z[0]
        k0 = 2 * np.pi / self.wavelength

        if n is None:
            n = self.n_background

        kx = get_k(self.x, '+')
        ky = get_k(self.y, '+')
        Kx, Ky = np.meshgrid(kx, ky)
        K_perp2 = Kx**2 + Ky**2

        self.clear_field()
        num_steps = len(self.z)

        self.u[:, :, 0] = self.u0.u
        for i, zi in enumerate(self.z[0:-1]):
            result = self.u[:, :, i]
            result_next = PWD_kernel(result, n, k0, K_perp2, dz)
            self.u[:, :, i + 1] = result_next
            if verbose is True:
                print("{}/{}".format(i, num_steps), end='\r')

        if matrix is True:
            return self.u

    @check_none('x','y','z',raise_exception=bool_raise_exception)
    def WPM(self, has_edges: bool = True, pow_edge: int = 80, verbose: bool = False):
        """
        WPM Methods.
        'schmidt method is very fast, only needs discrete number of refractive indexes'


        Args:
            has_edges (bool): If True absorbing edges are used.
            pow_edge (float): If has_edges, power of the supergaussian
            verbose (bool): If True prints information

        References:

            1. M. W. Fertig and K.-H. Brenner, “Vector wave propagation method,” J. Opt. Soc. Am. A, vol. 27, no. 4, p. 709, 2010.

            2. S. Schmidt et al., “Wave-optical modeling beyond the thin-element-approximation,” Opt. Express, vol. 24, no. 26, p. 30188, 2016.

        """

        k0 = 2 * np.pi / self.wavelength
        x = self.x
        y = self.y
        z = self.z
        # dx = x[1] - x[0]
        # dy = y[1] - y[0]
        dz = z[1] - z[0]

        self.u[:, :, 0] = self.u0.u
        kx = get_k(x, flavour='+')
        ky = get_k(y, flavour='+')

        KX, KY = np.meshgrid(kx, ky)

        k_perp2 = KX**2 + KY**2
        # k_perp = np.sqrt(k_perp2)

        if has_edges is False:
            has_filter = np.zeros_like(self.z)
        elif isinstance(has_edges, int):
            has_filter = np.ones_like(self.z)
        else:
            has_filter = has_edges

        width_edge = 0.95*(self.x[-1]-self.x[0])/2
        x_center=(self.x[-1]+self.x[0])/2
        y_center=(self.y[-1]+self.y[0])/2
        
        px = (np.abs(self.X[:,:,0]-x_center) / width_edge)**pow_edge
        py = (np.abs(self.Y[:,:,0]-y_center) / width_edge)**pow_edge
        filter_function = np.exp(-px-py)

        t1 = time.time()

        num_steps = len(self.z)
        for j in range(1, num_steps):

            if has_filter[j] == 0:
                filter_edge = 1
            else:
                filter_edge = filter_function

            self.u[:, :, j] = self.u[:, :, j] + WPM_schmidt_kernel(
                self.u[:, :, j - 1], self.n[:, :, j - 1], k0, k_perp2,
                dz) * filter_edge

            if verbose is True:
                print("{}/{}".format(j, num_steps), sep='\r', end='\r')

        t2 = time.time()

        if verbose is True:
            print("Time = {:2.2f} s, time/loop = {:2.4} ms".format(
                t2 - t1, (t2 - t1) / len(self.z) * 1000))

        return (t2 - t1) / len(self.z) * 1000

    @check_none('x','y','z','u',raise_exception=bool_raise_exception)
    def to_Scalar_field_XY(self,
                           iz0: int | None = None,
                           z0: float | None = None,
                           is_class: bool = True,
                           matrix: bool = False):
        """pass results to Scalar_field_XY. Only one of the first two variables (iz0,z0) should be used

        Args:
            iz0 (int): position i of z data in array
            z0 (float): position z to extract
            is_class (bool): If True it returns a class
            matrix (bool): If True it returns a matrix

        """
        if is_class is True:
            field_output = Scalar_field_XY(x=self.x,
                                           y=self.y,
                                           wavelength=self.wavelength)
            if iz0 is None:
                iz, tmp1, tmp2 = nearest(self.z, z0)
            else:
                iz = iz0
                
            field_output.u = np.squeeze(self.u[:, :, iz])
            field_output.n = np.squeeze(self.n[:, :, iz])

            return field_output

        if matrix is True:
            if iz0 is None:
                iz, tmp1, tmp2 = nearest(self.z, z0)
            else:
                iz = iz0
            return np.squeeze(self.u[:, :, iz])


    @check_none('x','y','z','u',raise_exception=bool_raise_exception)
    def to_Scalar_field_XZ(self,
                           iy0: int | None = None,
                           y0: float | None = None,
                           is_class: bool = True,
                           matrix: bool = False):
        """pass results to Scalar_field_XZ. Only one of the first two variables (iy0,y0) should be used

        Args:
            iy0 (int): position i of y data in array
            y0 (float): position y to extract
            class (bool): If True it returns a class
            matrix (bool): If True it returns a matrix

        """
        if is_class is True:
            field_output = Scalar_field_XZ(x=self.x,
                                           z=self.z,
                                           wavelength=self.wavelength)
            if iy0 is None:
                iy, _, _ = nearest(self.y, y0)
            else:
                iy = iy0
            field_output.u = np.squeeze(self.u[iy, :, :].transpose())
            field_output.n = np.squeeze(self.n[iy, :, :].transpose())

                
            return field_output

        if matrix is True:
            if iy0 is None:
                iy, tmp1, tmp2 = nearest(self.y, y0)
            else:
                iy = iy0
            return np.squeeze(self.u[iy, :, :])


    @check_none('x','y','z','u',raise_exception=bool_raise_exception)
    def to_Scalar_field_YZ(self,
                           ix0: int | None = None,
                           x0: float | None = None,
                           is_class: bool = True,
                           matrix: bool = False):
        """pass results to Scalar_field_XZ. Only one of the first two variables (iy0,y0) should be used

        Args:
            ix0 (int): position i of x data in array
            x0 (float): position x to extract
            class (bool): If True it returns a class
            matrix (bool): If True it returns a matrix

        """
        if is_class is True:
            field_output = Scalar_field_XZ(x=self.y,
                                           z=self.z,
                                           wavelength=self.wavelength)
            if ix0 is None:
                ix, _, _ = nearest(self.x, x0)
            else:
                ix = ix0
            field_output.u = np.squeeze(self.u[:, ix, :]).transpose()
            field_output.n = np.squeeze(self.n[:, ix, :]).transpose()

            return field_output

        if matrix is True:
            if ix0 is None:
                ix, _, _ = nearest(self.x, x0)
            else:
                ix = ix0
            return np.squeeze(self.u[:, ix, :]).transpose()


    @check_none('x','y','z','u',raise_exception=bool_raise_exception)
    def to_Scalar_field_Z(self,
                          kind: str = 'amplitude',
                          x0: float | None = None,
                          y0: float | None = None,
                          has_draw: bool = True,
                          z_scale='mm'):
        """pass results to u(z). Only one of the first two variables (iy0,y0) and (ix0,x0) should be used.

        Args:
            kind (str): 'amplitude', 'intensity', 'phase'
            x0 (float): position x to extract
            y0 (float): position y to extract
            has_draw (bool): draw the field
            z_scale (str): 'mm', 'um'

        Returns:
            z (numpy.array): array with z
            field (numpy.array): amplitude, intensity or phase of the field
        """
        ix, _, _ = nearest(self.y, y0)
        iy, _, _ = nearest(self.x, x0)

        u = np.squeeze(self.u[iy, ix, :])

        if kind == 'amplitude':
            y = np.abs(u)
        elif kind == 'intensity':
            y = np.abs(u)**2
        elif kind == 'phase':
            y = np.angle(u)

        if has_draw is True:
            if z_scale == 'mm':
                plt.plot(self.z / mm, y, 'k', lw=2)
                plt.xlabel(r'$z\,(mm)$')
                plt.xlim(left=self.z[0] / mm, right=self.z[-1] / mm)

            elif z_scale == 'um':
                plt.plot(self.z, y, 'k', lw=2)
                plt.xlabel(r'$z\,(\mu m)$')
                plt.xlim(left=self.z[0], right=self.z[-1])

            plt.ylabel(kind)

        return self.z, y


    @check_none('u',raise_exception=bool_raise_exception)
    def average_intensity(self, has_draw: bool = False):
        """Returns average intensity as: (np.abs(self.u)**2).mean()

        Args:
            verbose(bool): If True prints data.

        Returns:
            intensity_mean (np.array): z array with average intensity at each plane z.

        """
        intensity_mean = (np.abs(self.u)**2).mean(axis=2)
        if has_draw is True:
            plt.figure()
            plt.imshow(intensity_mean)

        return intensity_mean


    @check_none('x','y','z','u',raise_exception=bool_raise_exception)
    def beam_widths(self,
                    kind: str = 'FWHM2D',
                    has_draw: tuple[bool, bool] = [True, False],
                    percentage: float = 0.5,
                    remove_background: bool = None,
                    verbose: bool = False):
        """Determines the widths of the beam

        Args:
            kind (str): kind of algorithm: 'sigma4', 'FWHM2D'
            has_draw (bool, bool): First for complete analysis, second for all FWHM2D computations
            percentage: TODO
            remove_background: TODO
            verbose (bool): prints info
        Returns:
            beam_width_x (np.array)
            beam_width_y (np.array)
            principal_axis_z (np.array)
        """

        # u_zi = Scalar_field_XY(self.x, self.y, self.wavelength)
        beam_width_x = np.zeros_like(self.z)
        beam_width_y = np.zeros_like(self.z)
        principal_axis_z = np.zeros_like(self.z)
        beam_width = np.zeros_like(self.z)

        for i, zi in enumerate(self.z):
            u_prop_mat = self.u[:, :, i].squeeze()

            if kind == 'sigma4':
                dx, dy, principal_axis, moments = beam_width_2D(
                    self.x, self.y,
                    np.abs(u_prop_mat)**2)

            elif kind == 'FWHM2D':
                intensity = np.abs(u_prop_mat)**2
                # intensity = correlate2d(
                #     intensity, np.ones((3, 3)) / 9, mode='same')
                dx, dy = FWHM2D(self.x,
                                self.y,
                                intensity,
                                percentage=percentage,
                                remove_background=remove_background,
                                has_draw=has_draw[1])
                principal_axis = 0.

            beam_width_x[i] = dx
            beam_width_y[i] = dy
            principal_axis_z[i] = principal_axis

            if verbose is True:
                print("{:2.0f} ".format(i))

        beam_width = np.sqrt(beam_width_x**2 + beam_width_y**2)
        if has_draw[0] is True:
            plt.figure()
            plt.plot(self.z, beam_width, 'r', label='axis')
            plt.xlabel(r"z ($\mu$m)")
            plt.ylabel(r"widths ($\mu$m)")
            plt.legend()

        return beam_width_x, beam_width_y, principal_axis_z


    @check_none('x','y','z','n',raise_exception=bool_raise_exception)
    def surface_detection(self, mode: int = 1, min_incr: float = 0.01, has_draw: bool = False):
        """detect edges of variation in refractive index

        Args:
            mode (int): TODO
            min_incr (float): minimum incremental variation to detect.
            has_draw (bool): if True, it draws the surface.        
        """

        if mode == 0:
            diff1 = gradient(np.abs(self.n), axis=0)
            diff2 = gradient(np.abs(self.n), axis=1)
            diff3 = gradient(np.abs(self.n), axis=2)
        elif mode == 1:
            diff1 = diff(np.abs(self.n), axis=0)
            diff2 = diff(np.abs(self.n), axis=1)
            diff3 = diff(np.abs(self.n), axis=2)
            diff1 = np.append(diff1, np.zeros((len(self.x), 1, 1)), axis=0)
            diff2 = np.append(diff2, np.zeros((1, len(self.y), 1)), axis=1)
            diff2 = np.append(diff2, np.zeros(1, 1, len(self.z)), axis=2)

        t = np.abs(diff1) + np.abs(diff2) + np.abs(diff3)

        ix, iy, iz = (t > min_incr).nonzero()

        if has_draw is True:
            plt.figure()
            extension = [self.z[0], self.z[-1], self.x[0], self.x[-1]]
            plt.imshow(t, extent=extension, alpha=0.5)

        return self.x[ix], self.x[iy], self.z[iz]


    @check_none('x','y','z','u',raise_exception=bool_raise_exception)
    def draw_proposal(self,
                      kind: str = 'intensity',
                      logarithm: float = 0,
                      normalize='maximum',
                      draw_borders: bool = False,
                      filename: str = '',
                      scale: str = '',
                      min_incr: float = 0.0005,
                      reduce_matrix: str = 'standard',
                      colorbar_kind: bool | str = False,
                      colormap_kind: str = "gist_heat"):
        """Draws  XYZ field.

        Args:
            kind (str): type of drawing: 'amplitude', 'intensity', 'phase', 'real'
            logarithm (float): If >0, intensity is scaled in logarithm
            normalize (bool): If True, max(intensity)=1
            draw_borders (bool): If True draw edges of objects
            filename (str): if not '' stores drawing in file,
            scale (str): '', 'scaled', 'equal', scales the XY drawing
            min_incr: incrimum increment in refractive index for detecting edges
            reduce_matrix (int, int), 'standard' or False: when matrix is enormous, we can reduce it only for drawing purposes. If True, reduction factor
        """

        if reduce_matrix is False:
            amplitude, intensity, phase = field_parameters(self.u)
        elif reduce_matrix == 'standard':
            num_x = len(self.x)
            num_y = len(self.y)
            num_z = len(self.z)
            reduction_x = int(num_x/2000)
            reduction_y = int(num_y/2000)
            reduction_z = int(num_z/2000)

            if reduction_x == 0:
                reduction_x = 1
            if reduction_y == 0:
                reduction_y = 1
            if reduction_z == 0:
                reduction_z = 1
            u_new = self.u[::reduction_x, ::reduction_y, ::reduction_z]
            amplitude, intensity, phase = field_parameters(u_new)
        else:
            u_new = self.u[::reduce_matrix[0], ::reduce_matrix[1], ::
                           reduce_matrix[2]]
            amplitude, intensity, phase = field_parameters(u_new)

        extension = [
            self.y[0], self.y[-1], self.x[0], self.x[-1], self.z[0], self.z[-1]
        ]

        plt.figure()

        if kind == 'intensity':
            I_drawing = intensity
            I_drawing = normalize_draw(I_drawing, logarithm, normalize)
        elif kind == 'amplitude':
            I_drawing = amplitude
            I_drawing = normalize_draw(I_drawing, logarithm, normalize)
        elif kind == 'phase':
            I_drawing = phase
        elif kind == 'real':
            I_drawing = np.real(self.u)
        else:
            print("bad kind parameter")
            return

        h1 = plt.imshow(I_drawing,
                        interpolation='bilinear',
                        aspect='auto',
                        origin='lower',
                        extent=extension)
        plt.xlabel(r'y ($\mu m$)')
        plt.ylabel(r'x ($\mu m$)')
        plt.zlabel(r'z ($\mu m$)')

        plt.axis(extension)
        if colorbar_kind not in (False, '', None):
            plt.colorbar(orientation=colorbar_kind)

        h1.set_cmap(colormap_kind)  # OrRd # Reds_r gist_heat
        plt.clim(I_drawing.min(), I_drawing.max())

        if scale != '':
            plt.axis(scale)

        if draw_borders is True:
            if self.borders is None:
                self.surface_detection(1, min_incr, reduce_matrix)
            plt.plot(self.borders[0], self.borders[1], 'w.', ms=1)

        if filename != '':
            plt.savefig(filename, dpi=100, bbox_inches='tight', pad_inches=0.1)

        return h1


    @check_none('x','y','z','u',raise_exception=bool_raise_exception)
    def draw_XY(self,
                z0: float = 5*mm,
                kind: Draw_XY_Options = 'intensity',
                logarithm: float = 0,
                normalize: str = 'maximum',
                title: str = '',
                filename: str = '',
                cut_value: float or None = None,
                has_colorbar='False',
                reduce_matrix=''):
        """ longitudinal profile XY at a given z value

        Args:
            z0 (float): value of z for interpolation
            kind (str): type of drawing: 'amplitude', 'intensity', 'phase', ' 'field', 'real_field', 'contour'
            logarithm (float): If >0, intensity is scaled in logarithm
            normalize (str):  False, 'maximum', 'area', 'intensity'
            title (str): title for the drawing
            filename (str): if not '' stores drawing in file,
            cut_value (float): if provided, maximum value to show
            has_colorbar (bool): if True draws the colorbar
            reduce_matrix ()
        """

        ufield = self.to_Scalar_field_XY(iz0=None,
                                         z0=z0,
                                         is_class=True,
                                         matrix=False)
        ufield.draw(kind=kind,
                    logarithm=logarithm,
                    normalize=normalize,
                    title=title,
                    filename=filename,
                    cut_value=cut_value,
                    has_colorbar=has_colorbar,
                    reduce_matrix=reduce_matrix)


    @check_none('x','y','z','u',raise_exception=bool_raise_exception)
    def draw_XZ(self,
                kind: Draw_XZ_Options = 'intensity',
                y0: float = 0*mm,
                logarithm: float = 0,
                normalize: str = '',
                draw_borders: bool = False,
                filename: str = '',
                **kwargs):
        """Longitudinal profile XZ at a given x0 value.

        Args:
            y0 (float): value of y for interpolation
            logarithm (float): If >0, intensity is scaled in logarithm
            normalize (str):  False, 'maximum', 'intensity', 'area'
            draw_borders (bool): check
            filename (str): filename to save
        """

        plt.figure()
        ufield = self.to_Scalar_field_XZ(y0=y0)
        # ufield.u = ufield.u.transpose()  # TODO: check if it is necessary
        h1 = ufield.draw(kind, logarithm, normalize, draw_borders, filename,
                         **kwargs)

        return h1


    @check_none('x','y','z','u',raise_exception=bool_raise_exception)
    def draw_YZ(self,
                kind: Draw_XZ_Options='intensity',
                x0=0*mm,
                logarithm=0,
                normalize='',
                draw_borders=False,
                filename=''):
        """Longitudinal profile YZ at a given x0 value.

        Args:
            x0 (float): value of x for interpolation
            logarithm (float): If >0, intensity is scaled in logarithm
            normalize (str):  False, 'maximum', 'intensity', 'area'
            draw_borders (bool): check
            filename (str): filename to save
        """

        percentage_intensity = CONF_DRAWING['percentage_intensity']

        plt.figure()
        ufield = self.to_Scalar_field_YZ(x0=x0)
        
        amplitude, I_drawing, phase = field_parameters(ufield.u, True)

        I_drawing = np.abs(ufield.u)**2


        if kind == 'intensity':
            I_drawing = I_drawing
            I_drawing = normalize_draw(I_drawing, logarithm, normalize)
            cmap = CONF_DRAWING['color_intensity']
        elif kind == 'amplitude':
            I_drawing = amplitude
            I_drawing = normalize_draw(I_drawing, logarithm, normalize)
            cmap = CONF_DRAWING['color_amplitude']
        elif kind == 'phase':
            phase = phase/degrees
            phase[I_drawing < percentage_intensity * (I_drawing.max())] = 0
            cmap = CONF_DRAWING['color_phase']

            I_drawing = phase
        elif kind == 'real':
            I_drawing = np.real(self.u)
        else:
            print("bad kind parameter")
            return


        if normalize == 'maximum':
            I_drawing = I_drawing / I_drawing.max()
        if normalize == 'area':
            area = (self.y[-1] - self.y[0]) * (self.z[-1] - self.z[0])
            I_drawing = I_drawing / area
        if normalize == 'intensity':
            I_drawing = I_drawing / (I_drawing.sum() / len(I_drawing))

        h1 = plt.imshow(I_drawing,
                        interpolation='bilinear',
                        aspect='auto',
                        origin='lower',
                        extent=[
                            self.z[0] , self.z[-1] , self.y[0],
                            self.y[-1]
                        ])
        plt.xlabel(r'z ($\mu$m)')
        plt.ylabel(r'y ($\mu$m)')
        h1.set_cmap(cmap)  # OrRd # Reds_r gist_heat
        plt.colorbar()
        

    @check_none('x','y','z','u',raise_exception=bool_raise_exception)
    def draw_XYZ(self, kind: Draw_XYZ_Options = 'intensity', drawing: Draw_pyvista_Options = 'volume', has_grid: bool=False,  filename: str = '', **kwargs):
        """Draws the intensity distribution or the refractive index. There are serveral procedures:

        Args:
            kind (str, optional): "intensity" or "refractive_index". Defaults to 'refractive_index'.
            drawing (str, optional): volume, clip, slices, projections. Defaults to 'volume'.
            has_grid (bool, optional): add grid. Defaults to False.
            filename (str, optional): saves images: html, png or svg. Defaults to ''.
            
            TODO: not drawing properly.
        """        
        pl = draw(self, kind, drawing, has_grid,  filename, **kwargs)
        return pl


    def video_isovalue(self,  filename: str, kind: video_isovalue_Options = 'refractive_index', **kwargs):
        """_summary_

        Args:
            filename (str): filename. Defaults to ''.
            kind (str, optional): "intensity" or "refractive_index". Defaults to 'refractive_index'.
        """
        video_isovalue(self,  filename, kind,  **kwargs)


    @check_none('x','y','z','u',raise_exception=bool_raise_exception)
    def video(self,
              filename: str = '',
              kind: str = 'intensity',
              fps: int = 15,
              frame: bool = True,
              axis: str = 'auto',
              verbose: bool = False,
              directory_name: str = 'new_video'):
        """Makes a video in the range given by self.z.

        Args:
            filename (str): filename (.avi, .mp4)
            kind (str): type of drawing:  'intensity', 'phase'.
            fps (int): frames per second
            frame (bool): figure with or without axis.
            verbose (bool): If True prints

        TODO: include logarithm and normalize
        """

        def f(x, kind):
            # return x
            amplitude, intensity, phase = field_parameters(
                x, has_amplitude_sign=True)
            if kind == 'intensity':
                # np.log(1 * intensity + 1)
                return intensity
            elif kind == 'phase':
                return phase

            elif kind == 'real':
                return np.real(x)

            else:
                return "no correct kind in video"

        if kind == 'intensity':
            cmap1 = self.CONF_DRAWING['color_intensity']
        elif kind == 'phase':
            cmap1 = self.CONF_DRAWING['color_phase']

        elif kind == 'real':
            cmap1 = self.CONF_DRAWING['color_real']

        else:
            return "no correct kind in video"

        file, extension = os.path.splitext(filename)
        if extension == '.avi':
            Writer = anim.writers['ffmpeg']
            writer = Writer(fps=fps)  # codec='ffv1'
        elif extension == '.mp4':
            Writer = anim.writers['ffmpeg']
            writer = Writer(fps=fps, bitrate=1e6,
                            codec='mpeg4')  # codec='mpeg4',
        else:
            print("file needs to be .avi or .mp4")
            print("No writer is available. is .png? Then correct.")
        xmin, xmax, ymin, ymax = self.x[0], self.x[-1], self.y[0], self.y[-1]

        if frame is True:
            plt.ion()
            fig, axes = plt.subplots(nrows=1)
            ax = plt.gca()
            plt.axis(axis)
        else:
            fig = plt.figure()
            ax = fig.add_axes([0, 0, 1, 1])

        frame = self.to_Scalar_field_XY(iz0=0,
                                        z0=None,
                                        is_class=True,
                                        matrix=False)

        intensity_global = f(self.u, kind).max()
        intensity = f(frame.u, kind)

        image = ax.imshow(intensity,
                          interpolation='bilinear',
                          aspect='equal',
                          origin='lower',
                          extent=[xmin, xmax, ymin, ymax])
        image.set_cmap(cmap1)  # seismic coolwarm gist_heat
        fig.canvas.draw()

        n_frames = len(self.z)

        if extension == '.png':
            current_directory = os.getcwd()
            try:
                os.makedirs(directory_name)
                print("new directory: {}".format(directory_name))
            except OSError:
                print("this directory is not new: overwrite.")

            os.chdir(directory_name)

            for i_prog in range(n_frames):
                frame = self.to_Scalar_field_XY(iz0=i_prog,
                                                z0=None,
                                                is_class=True,
                                                matrix=False) # type: ignore
                intensity = f(frame.u, kind)
                image.set_array(intensity)
                if kind == 'intensity':
                    image.set_clim(vmax=intensity_global)
                elif kind == 'phase':
                    image.set_clim(-np.pi, np.pi)
                texto = "z = {:2.3f} mm".format(self.z[i_prog] / mm)
                plt.xlabel(r"x (microns)")
                plt.ylabel(r"y (microns)")
                plt.title(texto)
                plt.draw()
                plt.savefig("{}_{:04.0f}.png".format(file, i_prog))
                if verbose:
                    print(
                        "{} de {}: z={}, max= {:2.2f} min={:2.2f}".format(
                            i_prog, n_frames, self.z[i_prog] / mm,
                            intensity.max(), intensity.min()),
                        end='\r')

            os.chdir(current_directory)

        elif extension == '.avi' or extension == '.mp4':
            with writer.saving(fig, filename, 300):
                for i_prog in range(n_frames):
                    frame = self.to_Scalar_field_XY(iz0=i_prog,
                                                    z0=None,
                                                    is_class=True,
                                                    matrix=False)
                    intensity = f(frame.u, kind)
                    image.set_array(intensity)
                    if kind == 'intensity':
                        image.set_clim(vmax=intensity_global)
                    elif kind == 'phase':
                        image.set_clim(-np.pi, np.pi)
                    texto = "z = {:2.3f} mm".format(self.z[i_prog] / mm)
                    plt.xlabel(r"x (microns)")
                    plt.ylabel(r"y (microns)")
                    plt.title(texto)
                    plt.draw()
                    writer.grab_frame(facecolor='k')
                    if verbose:
                        print("{} de {}: z={}, max= {:2.2f} min={:2.2f}".
                                format(i_prog, n_frames, self.z[i_prog] / mm,
                                        intensity.max(), intensity.min()),
                                end='\r')


        plt.close()
