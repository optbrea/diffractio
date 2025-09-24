# !/usr/bin/env python3

# ----------------------------------------------------------------------
# Name:        vector_fields_XZ.py
# Purpose:     Class for handling vector fields in the XZ plane
#
# Author:      Luis Miguel Sanchez Brea
#
# Created:     2024
# Licence:     GPLv3
# ----------------------------------------------------------------------


# flake8: noqa

"""
This module generates Vector_field_X class. It is required also for generating masks and fields.
The main atributes are:
    * self.x - x positions of the field
    * self.Ex - x component of electric field
    * self.Ey - y component of electric field
    * self.Ez - z component of electric field
    * self.wavelength - wavelength of the incident field. The field is monocromatic
    * self.info (str): description of data
    * self.type (str): Class of the field
    * self.date (str): date when performed


The magnitude is related to microns: `micron = 1.`

*Class for X vector fields*

*Definition of a scalar field*
    * add, substract fields
    * save, load data, clean, get, normalize
    * cut_resample

*Vector parameters*
    * polarization_states

*Drawing functions*
    * draw: intensity, intensities, phases, fields, stokes, param_ellipse, ellipses

"""

import copy
from matplotlib import cm, rcParams
import time
from numpy import gradient
from scipy.interpolate import RectBivariateSpline
import cmath



from .__init__ import degrees, eps, mm, np, plt
from .config import (Draw_refractive_index_Options, bool_raise_exception, CONF_DRAWING, 
                     get_vector_options, Draw_Vector_XZ_Options)
from .utils_typing import npt, Any, NDArray, NDArrayFloat, NDArrayComplex
from .utils_common import get_date, load_data_common, save_data_common, check_none, get_vector
from .utils_common import get_instance_size_MB
from .utils_drawing import normalize_draw, reduce_matrix_size, draw_edges, draw2D
from .utils_math import get_k, nearest
from .utils_optics import normalize_field, fresnel_equations_kx
from .scalar_fields_X import Scalar_field_X
from .scalar_fields_XZ import Scalar_field_XZ
from .scalar_masks_XZ import Scalar_mask_XZ
from .vector_fields_X import Vector_field_X

from .utils_common import get_vector_options
import diffractio

from py_pol.jones_vector import Jones_vector

from numpy.lib.scimath import sqrt as csqrt
from scipy.fftpack import fft, fftshift, ifft, ifftshift

percentage_intensity_config = CONF_DRAWING['percentage_intensity']


class Vector_field_XZ():
    """Class for vectorial fields.

    Args:
        x (numpy.array): linear array with equidistant positions. The number of data is preferibly 2**n.
        wavelength (float): wavelength of the incident field
        info (str): String with info about the simulation

    Attributes:
        self.x (numpy.array): linear array with equidistant positions. The number of data is preferibly 2**n.
        self.wavelength (float): wavelength of the incident field.
        self.Ex (numpy.array): Electric_x field
        self.Ey (numpy.array): Electric_y field
        self.Ez (numpy.array): Electric_z field
    """

    def __init__(self, x: NDArrayFloat | None = None, z: NDArrayFloat | None = None,
                 wavelength: float | None = None, n_background: float = 1., info: str = ""):
        self.x = x
        self.z = z
        self.wavelength = wavelength
        self.n_background = n_background

        self.X, self.Z = np.meshgrid(x, z)

        self.Ex = np.zeros_like(self.X, dtype=complex)
        self.Ey = np.zeros_like(self.X, dtype=complex)
        self.Ez = np.zeros_like(self.X, dtype=complex)

        self.Hx = None
        self.Hy = None
        self.Hz = None

        self.n = n_background * np.ones_like(self.X, dtype=complex)
        self.borders = None  

        self.Ex0 = np.zeros_like(self.x)
        self.Ey0 = np.zeros_like(self.x)

        self.reduce_matrix = "standard"  # 'None, 'standard', (5,5)
        self.type = "Vector_field_XZ"
        self.info = info
        self.date = get_date()
        self.CONF_DRAWING = CONF_DRAWING


    @check_none('x', 'z', 'Ex', 'Ey', raise_exception=bool_raise_exception)
    def __str__(self):
        """Represents data from class."""

        intensity = self.intensity()
        Imin = intensity.min()
        Imax = intensity.max()

        print("{}\n - x:  {},   z:  {},   Ex:  {}".format(
            self.type, self.x.shape, self.z.shape, self.Ex.shape))

        print(
            " - xmin:       {:2.2f} um,  xmax:      {:2.2f} um,  Dx:   {:2.2f} um".format(
                self.x[0], self.x[-1], self.x[1] - self.x[0]
            )
        )
        print(
            " - zmin:       {:2.2f} um,  zmax:      {:2.2f} um,  Dz:   {:2.2f} um".format(
                self.z[0], self.z[-1], self.x[1] - self.z[0]
            )
        )
        print(" - Imin:       {:2.2f},     Imax:      {:2.2f}".format(Imin, Imax))
        
        print(" - nmin:       {:2.2f},     nmax:      {:2.2f}".format(self.n.min(), self.n.max()))

        print(" - wavelength: {:2.2f} um".format(self.wavelength))
        print(" - date:       {}".format(self.date))
        print(" - info:       {}".format(self.info))

        return ""


    @check_none('x', 'z', 'Ex', 'Ey', 'Ez', raise_exception=bool_raise_exception)
    def __add__(self, other):
        """adds two Vector_field_X. For example two light sources or two masks

        Args:
            other (Vector_field_X): 2nd field to add
            kind (str): instruction how to add the fields:

        Returns:
            Vector_field_X: `E3 = E1 + E2`
        """

        EM = Vector_field_XZ(self.x, self.z, self.wavelength)

        EM.Ex = self.Ex + other.Ex
        EM.Ey = self.Ey + other.Ey
        EM.Ez = self.Ez + other.Ez

        return EM


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
            final_filename = save_data_common(
                self, filename, add_name, description, verbose
            )
            return final_filename
        except:
            return False


    def load_data(self, filename: str, verbose: bool = False):
        """Load data from a file to a Vector_field_X.
            The methods included are: npz, matlab

        Args:
            filename (str): filename
            verbose (bool): shows data process by screen
        """
        dict0 = load_data_common(self, filename)

        if dict0 is not None:
            if isinstance(dict0, dict):
                self.__dict__ = dict0
            else:
                raise Exception("no dictionary in load_data")

        if verbose is True:
            print(dict0.keys())


    @check_none('Ex', 'Ey', 'Ez', raise_exception=bool_raise_exception)
    def clear_field(self):
        """Removes the fields Ex, Ey, Ez"""

        self.Ex = np.zeros_like(self.Ex, dtype=complex)
        self.Ey = np.zeros_like(self.Ey, dtype=complex)
        self.Ez = np.zeros_like(self.Ez, dtype=complex)


    def duplicate(self, clear: bool = False):
        """Duplicates the instance"""
        new_field = copy.deepcopy(self)
        if clear is True:
            new_field.clear_field()
        return new_field


    def size(self, verbose: bool = False):
        """returns the size of the instance in MB.

        Args:
            verbose (bool, optional): prints size in Mb. Defaults to False.

        Returns:
            float: size in MB
        """

        return get_instance_size_MB(self, verbose)


    def normalize(self, kind='amplitude', new_field: bool = False):
        """Normalizes the field so that intensity.max()=1.

        Args:
            new_field (bool): If False the computation goes to self.u. If True a new instance is produced
            kind (str): 'amplitude', or 'intensity'

        Returns
            u (numpy.array): normalized optical field
        """

        return normalize_field(self, kind, new_field)


    @check_none('x', 'z', 'Ex', 'Ey', 'Ez', raise_exception=bool_raise_exception)
    def cut_resample(self,
                     x_limits: tuple[float, float] | None = None,
                     z_limits: tuple[float, float] | None = None,
                     num_points: int | None = None,
                     new_field: bool = False,
                     interp_kind: tuple[int, int] = (3, 1)):
        """Cuts the field to the range (x0,x1). (z0,z1). If one of this x0,x1 positions is out of the self.x range it do nothing. It is also valid for resampling the field, just write x0,x1 as the limits of self.x

        Args:
            x_limits (float,float): (x0,x1) starting and final points to cut. if '' - takes the current limit x[0] and x[-1]
            z_limits (float,float): (z0,z1) - starting and final points to cut. if '' - takes the current limit z[0] and z[-1]
            num_points (int): it resamples x, z and u. [], '',0,None -> it leave the points as it is
            new_field (bool): it returns a new Scalar_field_Xz
            interp_kind (int): numbers between 1 and 5
        """
        if x_limits is None:
            x0 = self.x[0]
            x1 = self.x[-1]
        else:
            x0, x1 = x_limits

        if z_limits is None:
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

            print(" size z_new, x_new", z_new.shape, x_new.shape)


            f_interp_abs_Ex = RectBivariateSpline(self.z,
                                                 self.x,
                                                 np.abs(self.Ex),
                                                 kx=kxu,
                                                 ky=kxu,
                                                 s=0)
            f_interp_phase_Ex = RectBivariateSpline(self.z,
                                                   self.x,
                                                   np.angle(self.Ex),
                                                   kx=kxu,
                                                   ky=kxu,
                                                   s=0)

            f_interp_abs_Ey = RectBivariateSpline(self.z,
                                                 self.x,
                                                 np.abs(self.Ey),
                                                 kx=kxu,
                                                 ky=kxu,
                                                 s=0)
            f_interp_phase_Ey = RectBivariateSpline(self.z,
                                                   self.x,
                                                   np.angle(self.Ey),
                                                   kx=kxu,
                                                   ky=kxu,
                                                   s=0)

            f_interp_abs_Ez = RectBivariateSpline(self.z,
                                                 self.x,
                                                 np.abs(self.Ez),
                                                 kx=kxu,
                                                 ky=kxu,
                                                 s=0)
            f_interp_phase_Ez = RectBivariateSpline(self.z,
                                                   self.x,
                                                   np.angle(self.Ez),
                                                   kx=kxu,
                                                   ky=kxu,
                                                   s=0)
            
            
            f_interp_abs_Hx = RectBivariateSpline(self.z,
                                                 self.x,
                                                 np.abs(self.Hx),
                                                 kx=kxu,
                                                 ky=kxu,
                                                 s=0)
            f_interp_phase_Hx = RectBivariateSpline(self.z,
                                                   self.x,
                                                   np.angle(self.Hx),
                                                   kx=kxu,
                                                   ky=kxu,
                                                   s=0)

            f_interp_abs_Hy = RectBivariateSpline(self.z,
                                                 self.x,
                                                 np.abs(self.Hy),
                                                 kx=kxu,
                                                 ky=kxu,
                                                 s=0)
            f_interp_phase_Hy = RectBivariateSpline(self.z,
                                                   self.x,
                                                   np.angle(self.Hy),
                                                   kx=kxu,
                                                   ky=kxu,
                                                   s=0)

            f_interp_abs_Hz = RectBivariateSpline(self.z,
                                                 self.x,
                                                 np.abs(self.Hz),
                                                 kx=kxu,
                                                 ky=kxu,
                                                 s=0)
            f_interp_phase_Hz = RectBivariateSpline(self.z,
                                                   self.x,
                                                   np.angle(self.Hz),
                                                   kx=kxu,
                                                   ky=kxu,
                                                   s=0)
            
            
            
            f_interp_abs_n = RectBivariateSpline(self.z,
                                                 self.x,
                                                 np.abs(self.n),
                                                 kx=kxu,
                                                 ky=kxu,
                                                 s=0)
            f_interp_phase_n = RectBivariateSpline(self.z,
                                                   self.x,
                                                   np.angle(self.n),
                                                   kx=kxu,
                                                   ky=kxu,
                                                   s=0)

            Ex_new_abs = f_interp_abs_Ex(z_new, x_new)
            Ex_new_phase = f_interp_phase_Ex(z_new, x_new)
            Ex_new = Ex_new_abs * np.exp(1j * Ex_new_phase)

            Ey_new_abs = f_interp_abs_Ey(z_new, x_new)
            Ey_new_phase = f_interp_phase_Ey(z_new, x_new)
            Ey_new = Ey_new_abs * np.exp(1j * Ey_new_phase)

            Ez_new_abs = f_interp_abs_Ez(z_new, x_new)
            Ez_new_phase = f_interp_phase_Ez(z_new, x_new)
            Ez_new = Ez_new_abs * np.exp(1j * Ez_new_phase)

            Hx_new_abs = f_interp_abs_Hx(z_new, x_new)
            Hx_new_phase = f_interp_phase_Hx(z_new, x_new)
            Hx_new = Hx_new_abs * np.exp(1j * Hx_new_phase)

            Hy_new_abs = f_interp_abs_Hy(z_new, x_new)
            Hy_new_phase = f_interp_phase_Hy(z_new, x_new)
            Hy_new = Hy_new_abs * np.exp(1j * Hy_new_phase)

            Hz_new_abs = f_interp_abs_Hz(z_new, x_new)
            Hz_new_phase = f_interp_phase_Hz(z_new, x_new)
            Hz_new = Hz_new_abs * np.exp(1j * Hz_new_phase)


            n_new_abs = f_interp_abs_n(z_new, x_new)
            n_new_phase = f_interp_phase_n(z_new, x_new)
            n_new = n_new_abs * np.exp(1j * n_new_phase)

        else:
            iz_s = slice(i_z0, i_z1)
            jx_s = slice(i_x0, i_x1)
            x_new = self.x[jx_s]
            z_new = self.z[iz_s]
            X_new, Z_new = np.meshgrid(x_new, z_new)
            Ex_new = self.Ex[iz_s, jx_s]
            Ey_new = self.Ey[iz_s, jx_s]
            Ez_new = self.Ez[iz_s, jx_s]
            Hx_new = self.Hx[iz_s, jx_s]
            Hy_new = self.Hy[iz_s, jx_s]
            Hz_new = self.Hz[iz_s, jx_s]            
            
            n_new = self.n[iz_s, jx_s]

        if new_field is False:
            self.x = x_new
            self.z = z_new
            self.Ex = Ex_new
            self.Ey = Ey_new
            self.Ez = Ez_new
            self.Hx = Hx_new
            self.Hy = Hy_new
            self.Hz = Hz_new
            self.X = X_new
            self.Z = Z_new
            self.n = n_new
        else:
            field = Vector_field_XZ(x=x_new,
                                    z=z_new,
                                    wavelength=self.wavelength)
            field.Ex = Ex_new
            field.Ey = Ey_new
            field.Ez = Ez_new
            field.Hx = Hx_new
            field.Hy = Hy_new
            field.Hz = Hz_new
            field.n = n_new
            return field

    @check_none('x', 'Ex', 'Ey', 'Ez', raise_exception=bool_raise_exception)
    def incident_field(self, E0: Vector_field_X  | None = None, u0: Scalar_field_X  | None = None, 
                       j0: Jones_vector  | None = None, z0: float | None = None):
        """Includes the incident field in Vector_field_XZ. 
        
        It can be performed using a Vector_field_X E0 or a Scalar_field_X u0 + Jones_vector j0.

        Args:
            E0 (Vector_field_X | None): Vector field of the incident field.
            u0 (Scalar_field_x | None): Scalar field of the incident field.
            j0 (py_pol.Jones_vector | None): Jones vector of the incident field.
            z0 (float | None): position of the incident field. if None, the field is at the beginning.
        """

        if np.logical_and.reduce((E0 is None, u0 is not None, j0 is not None)):
            E0 = Vector_field_X(self.x, self.wavelength, self.n_background)
            E0.Ex = u0.u * j0.M[0]
            E0.Ey = u0.u * j0.M[1]

        if z0 in (None, '', []):
            self.Ex0 = E0.Ex
            self.Ey0 = E0.Ey

            self.Ex[0, :] = self.Ex[0, :] + E0.Ex
            self.Ey[0, :] = self.Ey[0, :] + E0.Ey
        else:
            iz, _, _ = nearest(self.z, z0)
            self.Ex[iz, :] = self.Ex[iz, :] + E0.Ex
            self.Ey[iz, :] = self.Ey[iz, :] + E0.Ey


    @check_none('x', 'Ex', 'Ey', 'Ez', raise_exception=bool_raise_exception)
    def final_field(self):
        """Returns the final field as a Vector_field_X."""

        EH_final = Vector_field_X(x=self.x,
                                  wavelength=self.wavelength,
                                  n_background=self.n_background,
                                  info="from final_field at z0= {} um".format(
                                      self.z[-1]))
        EH_final.Ex = self.Ex[-1, :]
        EH_final.Ey = self.Ey[-1, :]
        EH_final.Ez = self.Ez[-1, :]
        EH_final.Hx = self.Hx[-1, :]
        EH_final.Hy = self.Hy[-1, :]
        EH_final.Hz = self.Hz[-1, :]
        return EH_final


    def refractive_index_from_scalarXZ(self, u_xz: Scalar_mask_XZ):
        """
        Refractive_index_from_scalarXZ. Gets the refractive index from a Scalar field and passes to a vector field.
        
        Obviously, the refractive index is isotropic.

        Args:
            self (Vector_field_XZ): Vector_field_XZ
            u_xz (Scalar_mask_XZ): Scalar_mask_XZ
        """
        self.n = u_xz.n
        
        edges = self.surface_detection( min_incr = 0.1,  has_draw = False)

        self.borders = edges           
        return edges
        

    @check_none('Ex', 'Ey', 'Ez', raise_exception=bool_raise_exception)
    def get(self, kind: get_vector_options, mode: str = 'modulus', is_matrix: bool = True, verbose: bool = False, **kwargs):
        """Takes the vector field and divide in Scalar_field_X.

        Args:
            kind (str): 'E', 'H', 'EH', 'fields', 'intensity', 'intensities', 'phases', 'poynting_vector', 
                        'poynting_vector_averaged',  'poynting_total',  'energy_density', 'irradiance', 
                        'stokes', 'params_ellipse'
            mode (str): 'modulus', 'real', 'imag', 'phase', 'dB'
            is_matrix (bool): If True returns a matrix or a list of matrices. If False returns a list of diffractio instances.
            verbose (bool): If True prints information
            **kwargs: additional arguments for get_vector

        Note:
            You can draw all the clases with: 
                for i, vector in enumerate(self):
                    vector.draw(kind='intensity', scale=scale, logarithm=logarithm)

                del vector

        Returns:
            Vector_field_X: (Ex, Ey, Ez),
        """

        if verbose is True:
            print("get_vector_options:", get_vector_options)

        if is_matrix is True:
            data = get_vector(self, kind, mode, **kwargs)
            return data
        else:

            matrices = get_vector(self, kind=kind, mode=mode, **kwargs)

            if isinstance(self, diffractio.vector_fields_X.Vector_field_X):
                class_scalar = diffractio.scalar_fields_X.Scalar_field_X
                type_scalar = 'Scalar_field_X'
            elif isinstance(self, diffractio.vector_fields_XY.Vector_field_XY):
                class_scalar = diffractio.scalar_fields_XY.Scalar_field_XY
                type_scalar = 'Scalar_field_XY'
            elif isinstance(self, diffractio.vector_fields_XYZ.Vector_field_XYZ):
                class_scalar = diffractio.scalar_fields_XYZ.Scalar_field_XYZ
                type_scalar = 'Scalar_field_XYZ'
            elif isinstance(self, diffractio.vector_fields_XZ.Vector_field_XZ):
                class_scalar = diffractio.scalar_fields_XZ.Scalar_field_XZ
                type_scalar = 'Scalar_field_XZ'
            elif isinstance(self, diffractio.vector_fields_Z.Vector_field_Z):
                class_scalar = diffractio.scalar_fields_Z.Scalar_field_Z
                type_scalar = 'Scalar_field_Z'
            elif isinstance(self, diffractio.vector_masks_XY.Vector_mask_XY):
                class_scalar = diffractio.scalar_masks_XY.Scalar_mask_XY
                type_scalar = 'Scalar_mask_XY'
            elif isinstance(self, diffractio.vector_sources_XY.Vector_source_XY):
                class_scalar = diffractio.scalar_sources_XY.Scalar_source_XY
                type_scalar = 'Scalar_source_XY'
            else:
                raise ValueError(f"Class {self.__class__} not recognized")   


            list_fields = []
            if isinstance(matrices, np.ndarray):
                field = self.duplicate(clear=True)
                field.type = type_scalar
                field.__class__ = class_scalar
                field.u = matrices.astype(np.complex128)
                list_fields.append(field)    
            else:
                if kind in ['EH', 'E2H2']:
                    matrices = [item for sublist in matrices for item in sublist]
            
                for i, matrix in enumerate(matrices):

                    field = self.duplicate(clear=True)
                    field.type = type_scalar
                    field.__class__ = class_scalar
                    field.u = matrix.astype(np.complex128)
                    del field.Ex, field.Ey, field.Ez, field.Hx, field.Hy, field.Hz
                    list_fields.append(field)
                return list_fields
    


    @check_none('x', 'z', 'Ex', 'Ey', 'Ez', raise_exception=bool_raise_exception)
    def apply_mask(self, u, new_field: bool = False):
        """Multiply field by binary scalar mask: self.Ex = self.Ex * u.u

        Args:
            u (Scalar_mask_X): mask
        """
        if new_field == False:
            self.Ex = self.Ex * u.u
            self.Ey = self.Ey * u.u
            self.Ez = self.Ez * u.u
        else:
            E_new = self.duplicate()
            E_new.Ex = self.Ex * u.u
            E_new.Ey = self.Ey * u.u
            E_new.Ez = self.Ez * u.u
            return E_new
        

    @check_none('x', 'z', raise_exception=bool_raise_exception)
    def FP_WPM(self, has_edges: bool = True, pow_edge: int = 80, matrix: bool = False, verbose: bool = False):
        """
        WPM Method. 'schmidt methodTrue is very fast, only needs discrete number of refractive indexes'

        Args:
            has_edges (bool): If True absorbing edges are used.
            pow_edge (float): If has_edges, power of the supergaussian
            matrix (bool): if True returns a matrix else
            verbose (bool): If True prints information

        References:
            - 1. M. W. Fertig and K.-H. Brenner, “Vector wave propagation method,” J. Opt. Soc. Am. A, vol. 27, no. 4, p. 709, 2010.
            - 2. S. Schmidt et al., “Wave-optical modeling beyond the thin-element-approximation,” Opt. Express, vol. 24, no. 26, p. 30188, 2016.
        """

        k0 = 2 * np.pi / self.wavelength

        x = self.x
        z = self.z

        dx = x[1] - x[0]
        dz = z[1] - z[0]

        self.Ex[0,:] = self.Ex0
        self.Ey[0,:] = self.Ey0

        self.Hx = np.zeros_like(self.Ex)
        self.Hy = np.zeros_like(self.Ex)
        self.Hz = np.zeros_like(self.Ex)

        kx = get_k(x, flavour="+")

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
        x_center = (self.x[-1] + self.x[0])/2

        filter_function = np.exp(-((np.abs(self.x - x_center) / width_edge) ** pow_edge))

        t1 = time.time_ns()

        num_steps = len(self.z)

        for j in range(1, num_steps):

            if has_filter[j] == 0:
                filter_edge = 1
            else:
                filter_edge = filter_function

            E_step, H_step = FP_WPM_schmidt_kernel(
                self.Ex[j - 1, :],
                self.Ey[j - 1, :],
                self.n[j - 1, :],
                self.n[j, :],
                k0,
                kx,
                self.wavelength,
                dz,
            ) * filter_edge

            self.Ex[j, :] = self.Ex[j, :] + E_step[0] * filter_edge
            self.Ey[j, :] = self.Ey[j, :] + E_step[1] * filter_edge
            self.Ez[j, :] = E_step[2] * filter_edge

            self.Hx[j, :] = H_step[0] * filter_edge
            self.Hy[j, :] = H_step[1] * filter_edge
            self.Hz[j, :] = H_step[2] * filter_edge

        # at the initial point the Ez field is not computed.
        self.Ex[0,:] = self.Ex[1,:]
        self.Ey[0,:] = self.Ey[1,:]
        self.Ez[0,:] = self.Ez[1,:]
        
        self.Hx[0,:] = self.Hx[1,:]
        self.Hy[0,:] = self.Hy[1,:]
        self.Hz[0,:] = self.Hz[1,:]

        t2 = time.time_ns()
        if verbose is True:
            print(
                "Time = {:2.2f} s, time/loop = {:2.4} ms".format(
                    (t2 - t1) / 1e9, (t2 - t1) / len(self.z) / 1e6
                )
            )

        if matrix is True:
            return (self.Ex, self.Ey, self.Ez), (self.Hx, self.Hy, self.Hz)





    def BWPM(self, has_edges: bool = True, pow_edge: int = 80, matrix: bool = False,  verbose: bool = False):
        """Algorithm for Backward Wave Propagation Method (BWPM)

        Args:
            has_edges (bool, optional): If True absorbing edges are used. Defaults to True.
            pow_edge (int, optional): Power of the supergaussian. Defaults to 80.
            matrix (bool, optional): If True returns a matrix else returns a list. Defaults to False.
            has_draw (bool, optional): If True draws the fields. Defaults to False.
            verbose (bool, optional): If True prints information. Defaults to False.
        """

        #self.clear_field()

        # Creo que propagación hacia adelante y obtención de campos reflejados
        field_back=self.bwpm_kernel(has_edges=has_edges, pow_edge=pow_edge, matrix=matrix, verbose=verbose)
        # estas son las fuentes reflejadas que se propagan hacia atraás, pero hay que darlas la vuelta

        # Recojo los campos reflejados y los paso a Ex, Ey Ez para luego propagar hacia atrás
        EH_xz_back = self.duplicate(clear=True)

        # Inicializo los campos de la fuente para que no se propaguen, los campos son internos.
        EH_xz_back.Ex0 = np.zeros_like(EH_xz_back.Ex0)
        EH_xz_back.Ey0 = np.zeros_like(EH_xz_back.Ey0)

        # Propagación de campos reflejados (pero se hace hacia adelante)

        # Aquí se invierte el sentido de la propagación, para propagar hacia adelante
        EH_xz_back.Ex = np.flipud(field_back.Ex_back)
        EH_xz_back.Ey = np.flipud(field_back.Ey_back)
        EH_xz_back.Ez = np.flipud(field_back.Ez_back)
        EH_xz_back.Hx = np.flipud(field_back.Hx_back)
        EH_xz_back.Hy = np.flipud(field_back.Hy_back)
        EH_xz_back.Hz = np.flipud(field_back.Hz_back)
        EH_xz_back.n  = np.flipud(field_back.n)

        EH_xz_back.FP_WPM(has_edges = has_edges)
    
        EH_xz_back.Ex = np.flipud(EH_xz_back.Ex)
        EH_xz_back.Ey = -np.flipud(EH_xz_back.Ey)
        EH_xz_back.Ez = -np.flipud(EH_xz_back.Ez)
        EH_xz_back.Hx = np.flipud(EH_xz_back.Hx)
        EH_xz_back.Hy = -np.flipud(EH_xz_back.Hy)
        EH_xz_back.Hz = -np.flipud(EH_xz_back.Hz)
        EH_xz_back.n  = np.flipud(EH_xz_back.n)      


        # Se suman los campos incidentes y reflejados, que se da la vuelta porque va hacia atrás
        self.Ex = self.Ex + EH_xz_back.Ex 
        self.Ey = self.Ey + EH_xz_back.Ey 
        self.Ez = self.Ez + EH_xz_back.Ez 
        self.Hx = self.Hx + EH_xz_back.Hx 
        self.Hy = self.Hy + EH_xz_back.Hy 
        self.Hz = self.Hz + EH_xz_back.Hz 
        self.n  = self.n


        return EH_xz_back


    def bwpm_kernel(self, has_edges: bool = True, pow_edge: int = 80, matrix: bool = False, verbose: bool = False):
        """
        WPM Method. 'schmidt method' is very fast, only needs discrete number of refractive indexes

        Args:
            has_edges (bool): If True absorbing edges are used.
            pow_edge (float): If has_edges, power of the supergaussian
            matrix (bool): if True returns a matrix else
            verbose (bool): If True prints information

        References:
            - 1. M. W. Fertig and K.-H. Brenner, “Vector wave propagation method,” J. Opt. Soc. Am. A, vol. 27, no. 4, p. 709, 2010.
            - 2. S. Schmidt et al., “Wave-optical modeling beyond the thin-element-approximation,” Opt. Express, vol. 24, no. 26, p. 30188, 2016.
        """

        k0 = 2 * np.pi / self.wavelength



        dx = self.x[1] - self.x[0]
        dz = self.z[1] - self.z[0]

        self.Ex[0,:] = self.Ex0
        self.Ey[0,:] = self.Ey0

        self.Hx = np.zeros_like(self.Ex)
        self.Hy = np.zeros_like(self.Ex)
        self.Hz = np.zeros_like(self.Ex)


        Ex_back = np.zeros_like(self.Ex)
        Ey_back = np.zeros_like(self.Ex)
        Ez_back = np.zeros_like(self.Ex)    
        Hx_back = np.zeros_like(self.Ex)
        Hy_back = np.zeros_like(self.Ex)
        Hz_back = np.zeros_like(self.Ex)


        kx = get_k(self.x, flavour="+")

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
        x_center = (self.x[-1] + self.x[0])/2

        filter_function = np.exp(-((np.abs(self.x - x_center) / width_edge) ** pow_edge))

        t1 = time.time_ns()

        num_steps = len(self.z)

        for j in range(1, num_steps):

            if has_filter[j] == 0:
                filter_edge = 1
            else:
                filter_edge = filter_function


            E_step, H_step = FP_WPM_schmidt_kernel(
                self.Ex[j - 1, :],
                self.Ey[j - 1, :],
                self.n[j - 1, :],
                self.n[j, :],
                k0,
                kx,
                self.wavelength,
                dz,
            ) * filter_edge

            E_step_back, H_step_back = BWPM_schmidt_kernel(
                self.Ex[j - 1, :],
                self.Ey[j - 1, :],
                self.n[j - 1, :],
                self.n[j, :],
                k0,
                kx,
                self.wavelength,
                dz,
            ) * filter_edge


            self.Ex[j, :] = self.Ex[j, :] + E_step[0] * filter_edge
            self.Ey[j, :] = self.Ey[j, :] + E_step[1] * filter_edge
            self.Ez[j, :] = E_step[2] * filter_edge

            self.Hx[j, :] = H_step[0] * filter_edge
            self.Hy[j, :] = H_step[1] * filter_edge
            self.Hz[j, :] = H_step[2] * filter_edge

            Ex_back[j, :] = E_step_back[0] * filter_edge
            Ey_back[j, :] = E_step_back[1] * filter_edge
            Ez_back[j, :] = E_step_back[2] * filter_edge

            Hx_back[j, :] = H_step_back[0] * filter_edge
            Hy_back[j, :] = H_step_back[1] * filter_edge
            Hz_back[j, :] = H_step_back[2] * filter_edge

        # at the initial point the Ez field is not computed.
    
        self.Ex[0,:] = self.Ex[1,:]
        self.Ey[0,:] = self.Ey[1,:]
        self.Ez[0,:] = self.Ez[1,:]

        self.Hx[0,:] = self.Hx[1,:]
        self.Hy[0,:] = self.Hy[1,:]
        self.Hz[0,:] = self.Hz[1,:]

        Ex_back[0,:] = Ex_back[1,:]
        Ey_back[0,:] = Ey_back[1,:]
        Ez_back[0,:] = Ez_back[1,:]

        Hx_back[0,:] = Hx_back[1,:]
        Hy_back[0,:] = Hy_back[1,:]
        Hz_back[0,:] = Hz_back[1,:]

        t2 = time.time_ns()
        if verbose is True:
            print(
                "Time = {:2.2f} s, time/loop = {:2.4} ms".format(
                    (t2 - t1) / 1e9, (t2 - t1) / len(self.z) / 1e6
                )
            )

        if matrix is True:
            E = (self.Ex, self.Ey, self.Ez)
            H = (self.Hx, self.Hy, self.Hz)
            E_back = (Ex_back, Ey_back, Ez_back)
            H_back = (Hx_back, Hy_back, Hz_back)
            return E, H, E_back, H_back
        else:

            self.Ex_back = Ex_back
            self.Ey_back = Ey_back
            self.Ez_back = Ez_back
            self.Hx_back = Hx_back
            self.Hy_back = Hy_back
            self.Hz_back = Hz_back
        
        return self


    @check_none('Ex', 'Ey', 'Ez', raise_exception=bool_raise_exception)
    def intensity(self):
        """ "Returns intensity."""
        intensity = np.abs(self.Ex) ** 2 + np.abs(self.Ey) ** 2 + np.abs(self.Ez) ** 2

        return intensity


    
    def check_energy(self, kind = 'all', has_draw : bool = True):
        """
        check_energy. Integrates the Sz field and checks the energy conservation.

        We have used the z component of the Poynting vector.

        Args:
            kind (str): 'all', 'Sz', 'Stot', 'Strans', 'U'
            has_draw (bool, optional): If True, it draws the energy at each plane z. Defaults to True.

        Returns:
            np.array: normalized (to the first data) energy at each plane z.
        """
        
        # permeability = 4 * np.pi * 1e-7
        # Z0 = 376.82

        Sx, Sy, Sz = self.get('poynting_vector_averaged')
        U = self.get('energy_density')
        S_tot = np.sqrt(Sx**2 + Sy**2 + Sz**2)
        S_trans = np.sqrt(Sx**2 + Sy**2)


        energy_z1 = (Sz).mean(axis=1)/(Sz[0, :]).mean()
        energy_z2 = S_tot.mean(axis=1)/(S_tot[0, :]).mean()
        energy_z3 = S_trans.mean(axis=1)/(S_trans.mean(axis=1)).max()
        energy_z4 = (U/self.n).mean(axis=1)/(U[0, :]/self.n[0,:]).mean()

        if has_draw:
            plt.figure()
            if kind == 'all' or kind == 'Sz':
                plt.plot(self.z, energy_z1, 'r', label='S$_{z}$')
            if kind == 'all' or kind == 'Strans':
                plt.plot(self.z, energy_z3, 'k', label='S$_{trans}$')
            if kind == 'all' or kind == 'Stot':
                plt.plot(self.z, energy_z2, 'g', label='S$_{tot}$')
            if kind == 'all' or kind == 'U':
                plt.plot(self.z, energy_z4, 'b', label='u/n')
                
            plt.xlim(self.z[0], self.z[-1])
            plt.grid('on')
            plt.xlabel(r"$z\,(mm)$")
            plt.ylabel(r"$Check$")
            plt.ylim(bottom=0)
            plt.legend()

        return energy_z1, energy_z2, energy_z3


    @check_none('x', 'z', 'n')
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

    def draw(
        self,
        kind: Draw_Vector_XZ_Options = "intensity",
        logarithm: float = 0,
        normalize: bool = False,
        cut_value: float | None = None,
        draw_borders: bool = True,
        filename="",
        scale: str = 'scaled',
        percentage_intensity: float | None = None,
        params_black: dict = None,
        params_white: dict = None,
        draw=True,
        **kwargs
    ):
        """Draws electromagnetic field

        Args:
            kind (str):  'intensity', 'intensities', intensities_rz, 'phases', fields', 'stokes', 'EH', 'E2H2', 'poynting_vector', 'poynting_vector_averaged',  'poynting_total',  'energy_density', 'irradiance', 'ellipses', 'param_ellipses', 'refractive_index', 'all'
            logarithm (float): If >0, intensity is scaled in logarithm
            normalize (bool): If True, max(intensity)=1
            cut_value (float): If not None, cuts the maximum intensity to this value
            draw_borders (bool): If True, draws the borders of the structure
            filename (str): if not '' stores drawing in file
            scale (str): 'scaled', 'physical'
            percentage_intensity (None or number): If None it takes from CONF_DRAWING['percentage_intensity'], else uses this value
            draw (bool): If True, it draws the field. Defaults to True.


        """

        if percentage_intensity is None:
            percentage_intensity = percentage_intensity_config

        if draw is True:
            if kind == "intensity":
                id_fig = self.__draw_intensity__(
                    logarithm, normalize, cut_value, draw_borders, scale, **kwargs
                )
            elif kind == "intensities":
                id_fig = self.__draw_intensities__(
                    logarithm, normalize, cut_value, draw_borders, scale, **kwargs
                )

            elif kind == "phases":
                id_fig = self.__draw_phases__(logarithm, normalize, cut_value, draw_borders, scale,  percentage_intensity, **kwargs)

            elif kind == "fields":
                id_fig = self.__draw_fields__(logarithm, normalize, cut_value, draw_borders, scale,  percentage_intensity, **kwargs)

            elif kind == "EH":
                id_fig = self.__draw_EH__(logarithm, normalize, cut_value, draw_borders, scale, **kwargs)

            elif kind == "E2H2":
                id_fig = self.__draw_E2H2__(logarithm, normalize, cut_value, draw_borders, scale, **kwargs)

            elif kind == "poynting_vector":
                id_fig = self.__draw_poynting_vector__(logarithm, normalize, cut_value, draw_borders, scale, **kwargs)

            elif kind == "poynting_vector_averaged":
                id_fig = self.__draw_poynting_vector_averaged__(logarithm, normalize, cut_value, draw_borders, scale, **kwargs)

            elif kind == "poynting_total":
                id_fig = self.__draw_poynting_total__(logarithm, normalize, cut_value, draw_borders, scale, **kwargs)

            elif kind == "energy_density":
                id_fig = self.__draw_energy_density__(logarithm, normalize, cut_value, draw_borders, scale, **kwargs)

            elif kind == "irradiance":
                id_fig = self.__draw_irradiance__(logarithm, normalize, cut_value, draw_borders, scale,  **kwargs)

            elif kind == "stokes":
                id_fig = self.__draw_stokes__(logarithm, normalize, cut_value, draw_borders, scale,  **kwargs)

            elif kind == "ellipses":
                id_fig = self.__draw_ellipses__(logarithm, normalize, cut_value, draw_borders, scale,  **kwargs)

            elif kind == "param_ellipses":
                id_fig = self.__draw_param_ellipse__(logarithm, normalize, cut_value, draw_borders, scale, **kwargs)

            elif kind == "refractive_index":
                id_fig = self.__draw_refractive_index__(logarithm, normalize, cut_value, draw_borders, scale, **kwargs)

            elif kind == "directions":
                id_fig = self.__draw_directions__(logarithm, normalize, cut_value, draw_borders, scale, **kwargs)

            elif kind == "arrows":
                id_fig = self.__draw_arrows__(logarithm, normalize, cut_value, draw_borders, scale, **kwargs)

            elif kind == "all":
                self.__draw_all__(params_black=params_black, params_white=params_white)
                id_fig = None

            elif kind.isdigit():
                pass


            else:
                print("not good kind parameter in vector_fields_XZ.draw()")
                id_fig = None

            plt.tight_layout()
            
            if filename != "":
                plt.savefig(filename, dpi=300, bbox_inches="tight", pad_inches=0.1)

            return id_fig


    def __draw_all__(self, params_black: dict = None, params_white: dict = None):

        if params_black is None:
            params_black = dict(
                scale='scaled',
                draw_borders=True,
                percentage_intensity=0.01,
                cut_value=None,
                normalize=False,
                logarithm=False,
                color='k.',
                ms=.75)

        if params_white is None:
            params_white = dict(
                scale='scaled',
                draw_borders=True,
                percentage_intensity=0.01,
                cut_value=None,
                normalize=False,
                logarithm=False,
                color='w.',
                ms=.75)
        
        self.draw('refractive_index'); plt.show()
        self.draw('intensities', **params_white); plt.show()
        self.draw('phases', **params_white); plt.show()
        self.draw('EH', **params_black); plt.show()
        self.draw('E2H2', **params_white); plt.show()
        self.draw('poynting_vector', **params_black); plt.show()
        self.draw('poynting_vector_averaged', **params_black); plt.show()
        self.draw('stokes', **params_black); plt.show()
        self.draw('intensity', **params_white); plt.show()
        self.draw('poynting_total', **params_white); plt.show()
        self.draw('energy_density', **params_white); plt.show()
        self.draw('irradiance', **params_white); plt.show()
        self.draw('ellipses', draw_arrow=False, **params_white); plt.show()
        self.draw('directions'); plt.show()
        self.draw('arrows', size_arrow=2, sep_x=4, sep_z=2); plt.show()
        #self.draw('param_ellipses', **params_black); plt.show()

        self.check_energy('U')


    def __draw_intensity__(self,  logarithm: float,  normalize: bool,  cut_value: float,
        draw_borders=False,
        scale = 'scaled',
        cmap=CONF_DRAWING["color_intensity"], 
        **kwargs
    ):
        """Draws the intensity

        Args:
            logarithm (float): If >0, intensity is scaled in logarithm
            normalize (bool): If True, max(intensity)=1
            cut_value (float): If not None, cuts the maximum intensity to this value
        """

        intensity = self.get("intensity")
        intensity = reduce_matrix_size(self.reduce_matrix, self.x, self.z, intensity)
        intensity = normalize_draw(intensity, logarithm, normalize, cut_value)


        fig, axs = plt.subplots(nrows=1, ncols=1)
        
        id_fig, ax, IDimage = draw2D_xz(
            intensity, self.z, self.x, ax=axs, xlabel=r"z $(\mu m)$", ylabel=r"x $(\mu m)$",
            cmap=CONF_DRAWING["color_intensity"], title=r'$I$')
        plt.axis(scale)
        draw_edges(self, plt, draw_borders, **kwargs)
        
        IDimage.set_clim(vmin=0)                
        cb_ax = fig.add_axes([0.2, 0, 0.6, 0.025])
        cbar = fig.colorbar(id_fig, cmap=cmap, cax=cb_ax, orientation='horizontal', shrink=0.5)
        plt.tight_layout() 


        return id_fig, ax, IDimage
    
    @check_none('x', 'z', 'Ex', 'Ey', 'Ez', raise_exception=bool_raise_exception)
    def __draw_intensities__(self,  logarithm: float,  normalize: bool,  cut_value: float,
        draw_borders=False,
        scale = 'scaled',
        cmap=CONF_DRAWING["color_intensity"],
        draw_z = True, **kwargs
    ):
        """internal funcion: draws phase

        Args:
            logarithm (float): If >0, intensity is scaled in logarithm
            normalize (bool): If True, max(intensity)=1
            cut_value (float): If not None, cuts the maximum intensity to this value
        """

        tx, ty = rcParams["figure.figsize"]

        intensity1 = np.abs(self.Ex) ** 2
        intensity1 = normalize_draw(intensity1, logarithm, normalize, cut_value)

        intensity2 = np.abs(self.Ey) ** 2
        intensity2 = normalize_draw(intensity2, logarithm, normalize, cut_value)

        intensity3 = np.abs(self.Ez) ** 2
        intensity3 = normalize_draw(intensity3, logarithm, normalize, cut_value)

        intensity_max = np.max((intensity1.max(), intensity2.max(), intensity3.max()))

        z0 = self.z
        x0 = self.x

        if draw_z is False:
            fig, axs = plt.subplots(nrows=1, ncols=2, sharex=True,  figsize=(1.5 * tx, 1 * ty))
            plt.subplot(1, 2, 1)
            id_fig, ax, IDimage = draw2D_xz(intensity1, z0, x0, axs[0], xlabel=r'z ($\mu$m)', ylabel=r'x ($\mu$m)', title=r'$I_x$', cmap=cmap)
            plt.axis(scale)
            draw_edges(self, plt,  draw_borders,  **kwargs)
            IDimage.set_clim(0, intensity_max)

            plt.subplot(1, 2, 2)
            id_fig, ax, IDimage = draw2D_xz(intensity2, z0, x0, axs[1], xlabel=r'z ($\mu$m)', ylabel='', title=r'$I_y$', cmap=cmap)
            plt.axis(scale)
            draw_edges(self, plt,  draw_borders, **kwargs)
            IDimage.set_clim(0, intensity_max)


            cb_ax = fig.add_axes([0.1, 0, 0.8, 0.05])
            cbar = fig.colorbar(id_fig, cmap=cmap, cax=cb_ax, orientation='horizontal', shrink=0.5)

        else:
            fig, axs = plt.subplots(nrows=1, ncols=3, sharex=True,  figsize=(2 * tx, 1 * ty))
            plt.subplot(1, 3, 1)
            id_fig, ax, IDimage = draw2D_xz(intensity1, z0, x0, axs[0], xlabel=r'z ($\mu$m)', ylabel=r'x ($\mu$m)', title=r'$I_x$', cmap=cmap)
            plt.axis(scale)
            draw_edges(self, plt,  draw_borders,  **kwargs)
            IDimage.set_clim(0, intensity_max)


            plt.subplot(1, 3, 2)
            id_fig, ax, IDimage = draw2D_xz(intensity2, z0, x0, axs[1], xlabel=r'z ($\mu$m)', ylabel='', title=r'$I_y$', cmap=cmap)
            plt.axis(scale)
            draw_edges(self, plt,  draw_borders, **kwargs)
            IDimage.set_clim(0, intensity_max)

            plt.subplot(1, 3, 3)
            id_fig, ax, IDimage = draw2D_xz(intensity3, z0, x0,axs[2], xlabel=r'z ($\mu$m)', ylabel='', title=r'$I_z$', cmap=cmap)
            plt.axis(scale)
            draw_edges(self, plt,  draw_borders, **kwargs)
            IDimage.set_clim(0, intensity_max)

            cb_ax = fig.add_axes([0.1, 0, 0.8, 0.05])
            cbar = fig.colorbar(id_fig, cmap=cmap, cax=cb_ax, orientation='horizontal', shrink=0.5)

        return fig, axs


    @check_none('x', 'z', 'Ex', 'Ey', 'Ez', raise_exception=bool_raise_exception)
    def __draw_phases__(self,  logarithm: float,  normalize: bool,  cut_value: float,
        draw_borders=False,
        scale = 'scaled',
        percentage_intensity=None,
        cmap=CONF_DRAWING["color_phase"],
        draw_z = True, **kwargs
    ):
        """internal funcion: draws phase

        Args:
            logarithm (float): If >0, intensity is scaled in logarithm
            normalize (bool): If True, max(intensity)=1
            cut_value (float): If not None, cuts the maximum intensity to this value
        """

        tx, ty = rcParams["figure.figsize"]

        phase_x = np.angle(self.Ex)/degrees
        phase_y = np.angle(self.Ey)/degrees
        phase_z = np.angle(self.Ez)/degrees

        intensity1 = np.abs(self.Ex)**2
        intensity2 = np.abs(self.Ex)**2
        intensity3 = np.abs(self.Ez)**2

        intensity1 = normalize_draw(intensity1, logarithm, normalize, cut_value)
        intensity2 = normalize_draw(intensity2, logarithm, normalize, cut_value)
        intensity3 = normalize_draw(intensity3, logarithm, normalize, cut_value)


        intensity_max = np.max((intensity1.max(), intensity2.max(), intensity3.max()))


        z0 = self.z
        x0 = self.x

        if percentage_intensity is None:
            percentage_intensity = percentage_intensity_config

        phase_x[intensity1 < percentage_intensity * (intensity1.max())] = 0
        phase_y[intensity2 < percentage_intensity * (intensity2.max())] = 0
        phase_z[intensity3 < percentage_intensity * (intensity3.max())] = 0


        if draw_z is False:
            fig, axs = plt.subplots(nrows=1, ncols=2, sharex=True,  figsize=(1.5 * tx, 1 * ty))
            plt.subplot(1, 2, 1)
            id_fig, ax, IDimage = draw2D_xz(phase_x, z0, x0, axs[0], xlabel=r'z ($\mu$m)', ylabel=r'x ($\mu$m)', title=r'$\phi_x$', cmap=cmap)
            plt.axis(scale)
            draw_edges(self, plt,  draw_borders,  **kwargs)
            IDimage.set_clim(-180,180)

            plt.subplot(1, 2, 2)
            id_fig, ax, IDimage = draw2D_xz(phase_y, z0, x0, axs[1], xlabel=r'z ($\mu$m)', ylabel='', title=r'$\phi_y$', cmap=cmap)
            plt.axis(scale)
            draw_edges(self, plt,  draw_borders, **kwargs)
            IDimage.set_clim(-180,180)


            cb_ax = fig.add_axes([0.1, 0, 0.8, 0.05])
            cbar = fig.colorbar(id_fig, cmap=cmap, cax=cb_ax, orientation='horizontal', shrink=0.5)

        else:
            fig, axs = plt.subplots(nrows=1, ncols=3, sharex=True,  figsize=(2 * tx, 1 * ty))
            plt.subplot(1, 3, 1)
            id_fig, ax, IDimage = draw2D_xz(phase_x, z0, x0, axs[0], xlabel=r'z ($\mu$m)', ylabel=r'x ($\mu$m)', title=r'$\phi_x$', cmap=cmap)
            plt.axis(scale)
            draw_edges(self, plt,  draw_borders,  **kwargs)
            IDimage.set_clim(-180,180)


            plt.subplot(1, 3, 2)
            id_fig, ax, IDimage = draw2D_xz(phase_y, z0, x0, axs[1], xlabel=r'z ($\mu$m)', ylabel='', title=r'$\phi_y$', cmap=cmap)
            plt.axis(scale)
            draw_edges(self, plt,  draw_borders, **kwargs)
            IDimage.set_clim(-180,180)

            plt.subplot(1, 3, 3)
            id_fig, ax, IDimage = draw2D_xz(phase_z, z0, x0,axs[2], xlabel=r'z ($\mu$m)', ylabel='', title=r'$\phi_z$', cmap=cmap)
            plt.axis(scale)
            draw_edges(self, plt,  draw_borders, **kwargs)
            IDimage.set_clim(-180,180)

            cb_ax = fig.add_axes([0.1, 0, 0.8, 0.05])
            cbar = fig.colorbar(id_fig, cmap=cmap, cax=cb_ax, orientation='horizontal', fraction=0.046, shrink=0.5)

        return fig, axs

    @check_none('x', 'z', 'Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz', raise_exception=bool_raise_exception)
    def __draw_fields__(self,  logarithm: float,  normalize: bool,  cut_value: float,
        draw_borders=False,
        scale = 'scaled',
        percentage_intensity: float | None = None,
        color_intensity=CONF_DRAWING["color_intensity"],
        color_phase=CONF_DRAWING["color_phase"],
        draw_z = True, **kwargs):
        """
        Args:
            logarithm (float): If >0, intensity is scaled in logarithm
            normalize (bool): If True, max(intensity)=1
            title (str): title of figure
            cut_value (float): If not None, cuts the maximum intensity to this value

        """

        if percentage_intensity is None:
            percentage_intensity = percentage_intensity_config

        intensity_x = np.abs(self.Ex) ** 2
        intensity_x = normalize_draw(intensity_x, logarithm, normalize, cut_value)

        intensity_y = np.abs(self.Ey) ** 2
        intensity_y = normalize_draw(intensity_y, logarithm, normalize, cut_value)

        intensity_max = np.max((intensity_x.max(), intensity_y.max()))
        tx, ty = rcParams["figure.figsize"]

        plt.figure(figsize=(2 * tx, 2 * ty))

        h1 = plt.subplot(2, 2, 1)

        self.__draw1__( intensity_x, color_intensity, r"$I_x$")
        plt.axis(scale)
        draw_edges(self, plt, draw_borders, **kwargs)
        plt.clim(0, intensity_max)

        h2 = plt.subplot(2, 2, 2)
        self.__draw1__(intensity_y, color_intensity,"$I_y$")
        plt.axis(scale)
        draw_edges(self, plt, draw_borders, **kwargs)
        plt.clim(0, intensity_max)

        h3 = plt.subplot(2, 2, 3)
        phase = np.angle(self.Ex)
        phase[intensity_x < percentage_intensity * (intensity_x.max())] = 0

        self.__draw1__(phase/degrees, color_phase, r"$\phi_x$")
        plt.axis(scale)
        draw_edges(self, plt, draw_borders, **kwargs)
        plt.clim(-180, 180)

        h4 = plt.subplot(2, 2, 4)
        phase = np.angle(self.Ey)
        phase[intensity_y < percentage_intensity * (intensity_y.max())] = 0

        self.__draw1__(phase/degrees, color_phase, r"$\phi_y$")
        plt.axis(scale)
        draw_edges(self, plt, draw_borders, **kwargs)
        plt.clim(-180, 180)
        
        h4 = plt.gca()
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.05, hspace=0)
        plt.tight_layout()
        return h1, h2, h3, h4


    @check_none('x', 'z', 'Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz', raise_exception=bool_raise_exception)
    def __draw_EH__(self,  logarithm: float,  normalize: bool,  cut_value: float,
        draw_borders=False,
        scale = 'scaled',
        cmap=CONF_DRAWING["color_amplitude_sign"],
        edge=None,
        draw_z = True,
        **kwargs
    ):
        """__internal__: draws amplitude and phase in 2x2 drawing

        Args:
            logarithm (float): If >0, intensity is scaled in logarithm
            normalize (bool): If True, max(intensity)=1
            title (str): title of figure
            cut_value (float): If not None, cuts the maximum intensity to this value

        """


        E_x = np.real(self.Ex)
        E_x = normalize_draw(E_x, logarithm, normalize, cut_value)

        E_y = np.real(self.Ey)
        E_y = normalize_draw(E_y, logarithm, normalize, cut_value)

        E_z = np.real(self.Ez)      
        E_z = normalize_draw(E_z, logarithm, normalize, cut_value)

        H_x = np.real(self.Hx)
        H_x = normalize_draw(H_x, logarithm, normalize, cut_value)

        H_y = np.real(self.Hy)
        H_y = normalize_draw(H_y, logarithm, normalize, cut_value)

        H_z = np.real(self.Hz)
        H_z = normalize_draw(H_z, logarithm, normalize, cut_value)

        tx, ty = rcParams["figure.figsize"]

        E_max = np.max((E_x.max(), E_y.max(), E_z.max()))
        H_max = np.max((H_x.max(), H_y.max(), H_z.max()))


        if draw_z is True:

            fig, axs = plt.subplots(
                nrows=2, ncols=3, sharex=True, sharey=True, figsize=(2 * tx, 2 * ty)
            )

            id_fig, ax, IDimage = draw2D_xz(
                E_x, self.z, self.x, ax=axs[0, 0], scale=scale, xlabel="", ylabel=r"x $(\mu m)$", cmap=cmap, title=r'E$_x$')
            draw_edges(self, axs[0, 0], draw_borders, **kwargs)
            IDimage.set_clim(-E_max,E_max)
            id_fig, ax, IDimage = draw2D_xz(
                E_y, self.z, self.x, ax=axs[0, 1], scale=scale, xlabel="", ylabel="", cmap=cmap, title=r'E$_y$')
            draw_edges(self, axs[0, 1], draw_borders, **kwargs)
            IDimage.set_clim(-E_max,E_max)
            id_fig, ax, IDimage = draw2D_xz(
                E_z, self.z, self.x, ax=axs[0, 2], scale=scale, xlabel="", ylabel="", cmap=cmap, title=r'E$_z$')
            draw_edges(self, axs[0, 2], draw_borders, **kwargs)
            IDimage.set_clim(-E_max,E_max)


            id_fig, ax, IDimage = draw2D_xz(
                H_x, self.z, self.x, ax=axs[1, 0], scale=scale, xlabel=r"z $(\mu m)$", ylabel=r"x $(\mu m)$", cmap=cmap, title=r'H$_x$')
            draw_edges(self, axs[1, 0], draw_borders, **kwargs)
            IDimage.set_clim(-H_max,H_max)
            id_fig, ax, IDimage = draw2D_xz(
                H_y, self.z, self.x, ax=axs[1, 1], scale=scale, xlabel=r"z $(\mu m)$", ylabel="", cmap=cmap, title=r'H$_y$')
            draw_edges(self, axs[1, 1], draw_borders, **kwargs)
            IDimage.set_clim(-H_max,H_max)
            id_fig, ax, IDimage = draw2D_xz(
                H_z, self.z, self.x, ax=axs[1, 2], scale=scale, xlabel=r"z $(\mu m)$", ylabel="", cmap=cmap, title=r'H$_z$')
            draw_edges(self, axs[1, 2], draw_borders, **kwargs)
            IDimage.set_clim(-H_max,H_max)

        else: 
            fig, axs = plt.subplots(
                nrows=2, ncols=2, sharex=True, sharey=True, figsize=(1.5 * tx, 2 * ty)
            )

            id_fig, ax, IDimage = draw2D_xz(
                E_x, self.z, self.x, ax=axs[0, 0], scale=scale, xlabel="", ylabel=r"x $(\mu m)$", cmap=cmap, title=r'E$_x$')
            draw_edges(self, axs[0, 0], draw_borders, **kwargs)
            IDimage.set_clim(-E_max,E_max)
            
            id_fig, ax, IDimage = draw2D_xz(
                E_y, self.z, self.x, ax=axs[0, 1], scale=scale, xlabel="", ylabel="", cmap=cmap, title=r'E$_y$')
            draw_edges(self, axs[0, 1], draw_borders, **kwargs)
            IDimage.set_clim(-E_max,E_max)

            id_fig, ax, IDimage = draw2D_xz(
                H_x, self.z, self.x, ax=axs[1, 0], scale=scale, xlabel=r"z $(\mu m)$", ylabel=r"x $(\mu m)$", cmap=cmap, title=r'H$_x$')
            draw_edges(self, axs[1, 0], draw_borders, **kwargs)
            IDimage.set_clim(-H_max,H_max)
            
            id_fig, ax, IDimage = draw2D_xz(
                H_y, self.z, self.x, ax=axs[1, 1], scale=scale, xlabel=r"z $(\mu m)$", ylabel="", cmap=cmap, title=r'H$_y$')
            draw_edges(self, axs[1, 1], draw_borders, **kwargs)
            IDimage.set_clim(-H_max,H_max)


        fig.subplots_adjust(right=1.25)
        cb_ax = fig.add_axes([0.2, 0, 0.6, 0.025])
        cbar = fig.colorbar(id_fig, cmap=cmap, cax=cb_ax, orientation='horizontal', shrink=0.5)

        plt.tight_layout()

        return self

    @check_none('x', 'z', 'Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz', raise_exception=bool_raise_exception)
    def __draw_E2H2__(self,  logarithm: float,  normalize: bool,  cut_value: float,
        draw_borders=False,
        scale = 'scaled',
        cmap=CONF_DRAWING["color_intensity"],
        edge=None,
        draw_z = True, 
        **kwargs
    ):
        """__internal__: draws amplitude and phase in 2x2 drawing

        Args:
            logarithm (float): If >0, intensity is scaled in logarithm
            normalize (bool): If True, max(intensity)=1
            title (str): title of figure
            cut_value (float): If not None, cuts the maximum intensity to this value

        """

        E_x = np.abs(self.Ex)**2
        E_x = normalize_draw(E_x, logarithm, normalize, cut_value)

        E_y = np.abs(self.Ey)**2
        E_y = normalize_draw(E_y, logarithm, normalize, cut_value)

        E_z = np.abs(self.Ez)**2
        E_z = normalize_draw(E_z, logarithm, normalize, cut_value)

        H_x = np.abs(self.Hx)**2
        H_x = normalize_draw(H_x, logarithm, normalize, cut_value)

        H_y = np.abs(self.Hy)**2
        H_y = normalize_draw(H_y, logarithm, normalize, cut_value)

        H_z = np.abs(self.Hz)**2
        H_z = normalize_draw(H_z, logarithm, normalize, cut_value)

        tx, ty = rcParams["figure.figsize"]

        E_max = np.max((E_x.max(), E_y.max(), E_z.max()))
        H_max = np.max((H_x.max(), H_y.max(), H_z.max()))

        if draw_z is True:

            fig, axs = plt.subplots(
                nrows=2, ncols=3, sharex=True, sharey=True, figsize=(2 * tx, 2 * ty)
            )

            id_fig, ax, IDimage = draw2D_xz(
                E_x, self.z, self.x, ax=axs[0, 0], scale=scale, xlabel="", ylabel=r"x $(\mu m)$", cmap=cmap, title=r'E$_x^2$')
            draw_edges(self, axs[0, 0], draw_borders, **kwargs)
            IDimage.set_clim(0,E_max)
            id_fig, ax, IDimage = draw2D_xz(
                E_y, self.z, self.x, ax=axs[0, 1], scale=scale, xlabel="", ylabel="", cmap=cmap, title=r'E$_y^2$')
            draw_edges(self, axs[0, 1], draw_borders, **kwargs)
            IDimage.set_clim(0,E_max)
            id_fig, ax, IDimage = draw2D_xz(
                E_z, self.z, self.x, ax=axs[0, 2], scale=scale, xlabel="", ylabel="", cmap=cmap, title=r'E$_z^2$')
            draw_edges(self, axs[0, 2], draw_borders, **kwargs)
            IDimage.set_clim(0,E_max)


            id_fig, ax, IDimage = draw2D_xz(
                H_x, self.z, self.x, ax=axs[1, 0], scale=scale, xlabel=r"z $(\mu m)$", ylabel=r"x $(\mu m)$", cmap=cmap, title=r'H$_x^2$')
            draw_edges(self, axs[1, 0], draw_borders, **kwargs)
            IDimage.set_clim(0,H_max)
            id_fig, ax, IDimage = draw2D_xz(
                H_y, self.z, self.x, ax=axs[1, 1], scale=scale, xlabel=r"z $(\mu m)$", ylabel="", cmap=cmap, title=r'H$_y^2$')
            draw_edges(self, axs[1, 1], draw_borders, **kwargs)
            IDimage.set_clim(0,H_max)
            id_fig, ax, IDimage = draw2D_xz(
                H_z, self.z, self.x, ax=axs[1, 2], scale=scale, xlabel=r"z $(\mu m)$", ylabel="", cmap=cmap, title=r'H$_z^2$')
            draw_edges(self, axs[1, 2], draw_borders, **kwargs)
            IDimage.set_clim(0,H_max)

        else: 
            fig, axs = plt.subplots(
                nrows=2, ncols=2, sharex=True, sharey=True, figsize=(1.5 * tx, 2 * ty)
            )

            id_fig, ax, IDimage = draw2D_xz(
                E_x, self.z, self.x, ax=axs[0, 0], scale=scale, xlabel="", ylabel=r"x $(\mu m)$", cmap=cmap, title=r'E$_x^2$')
            draw_edges(self, axs[0, 0], draw_borders, **kwargs)
            IDimage.set_clim(0,E_max)
            
            id_fig, ax, IDimage = draw2D_xz(
                E_y, self.z, self.x, ax=axs[0, 1], scale=scale, xlabel="", ylabel="", cmap=cmap, title=r'E$_y^2$')
            draw_edges(self, axs[0, 1], draw_borders, **kwargs)
            IDimage.set_clim(0,E_max)

            id_fig, ax, IDimage = draw2D_xz(
                H_x, self.z, self.x, ax=axs[1, 0], scale=scale, xlabel=r"z $(\mu m)$", ylabel=r"x $(\mu m)$", cmap=cmap, title=r'H$_x^2$')
            draw_edges(self, axs[1, 0], draw_borders, **kwargs)
            IDimage.set_clim(0,H_max)
            
            id_fig, ax, IDimage = draw2D_xz(
                H_y, self.z, self.x, ax=axs[1, 1], scale=scale, xlabel=r"z $(\mu m)$", ylabel="", cmap=cmap, title=r'H$_y^2$')
            draw_edges(self, axs[1, 1], draw_borders, **kwargs)
            IDimage.set_clim(0,H_max)


        fig.subplots_adjust(right=1.25)
        cb_ax = fig.add_axes([0.2, 0, 0.6, 0.025])
        cbar = fig.colorbar(id_fig, cmap=cmap, cax=cb_ax, orientation='horizontal', shrink=0.5)
        plt.tight_layout()

        return self

    def __draw_poynting_vector_averaged__(self,
        logarithm,
        normalize,
        cut_value,
        draw_borders=False,
        scale = 'scaled',
        cmap=CONF_DRAWING["color_amplitude_sign"],
        edge=None,
        **kwargs
        ):

        
        z0 = self.z
        x0 = self.x
        
        tx, ty = rcParams["figure.figsize"]

        
        Sx, Sy, Sz = self.get('poynting_vector_averaged', matrix=True)
        Sx = normalize_draw(Sx, logarithm, normalize, cut_value)
        Sy = normalize_draw(Sy, logarithm, normalize, cut_value)
        Sz = normalize_draw(Sz, logarithm, normalize, cut_value)


        S_max = np.max((Sx, Sy, Sz))
        S_min = np.min((Sx, Sy, Sz))
        S_lim = np.max((abs(S_max), np.abs(S_min)))

        fig, axs = plt.subplots(nrows=1, ncols=3, sharex=True,  figsize=(2 * tx, 1 * ty))
        plt.subplot(1, 3, 1)
        plt.title(r"$S_x$")
        id_fig, ax, IDimage = draw2D_xz(Sx, z0, x0, axs[0], xlabel=r'z ($\mu$m)', ylabel=r'x ($\mu$m)', title=r"$S_x$", cmap=cmap)
        plt.axis(scale)
        draw_edges(self, plt,  draw_borders,  **kwargs)
        IDimage.set_clim(-S_lim, S_lim)
        # axes[0].set_axis_off()

        plt.subplot(1, 3, 2)
        plt.title(r"$S_y$")
        id_fig, ax, IDimage = draw2D_xz(Sy, z0, x0, axs[1], xlabel=r'z ($\mu$m)', ylabel='', title=r"$S_y$", cmap=cmap)
        plt.axis(scale)
        draw_edges(self, plt,  draw_borders, **kwargs)
        IDimage.set_clim(-S_lim, S_lim)
        # axes[1].set_axis_off()

        plt.subplot(1, 3, 3)
        id_fig, ax, IDimage = draw2D_xz(Sz, z0, x0,axs[2], xlabel=r'z ($\mu$m)', ylabel='', title=r"$S_z$", cmap=cmap)
        plt.title(r"$S_z$")
        plt.axis(scale)
        draw_edges(self, plt,  draw_borders, **kwargs)
        IDimage.set_clim(-S_lim, S_lim)
        # axes[2].set_axis_off()

        cb_ax = fig.add_axes([0.1, 0, 0.8, 0.05])
        cbar = fig.colorbar(id_fig, cmap=cmap, cax=cb_ax, orientation='horizontal', shrink=0.5)

        plt.tight_layout()   

    def __draw_poynting_vector__(self, logarithm, normalize, cut_value, draw_borders=False,
        scale = 'scaled', cmap=CONF_DRAWING["color_amplitude_sign"], edge=None, **kwargs ):
        """Draws the poynting vector.

        Args:
            logarithm (_type_): _description_
            normalize (_type_): _description_
            cut_value (_type_): _description_
            draw_borders (bool, optional): _description_. Defaults to False.
            scale (str, optional): _description_. Defaults to 'scaled'.
            cmap (_type_, optional): _description_. Defaults to CONF_DRAWING["color_amplitude_sign"].
            edge (_type_, optional): _description_. Defaults to None.
        """
        
        z0 = self.z
        x0 = self.x
        
        tx, ty = rcParams["figure.figsize"]

        
        Sx, Sy, Sz = self.get('poynting_vector', matrix=True)
        Sx = normalize_draw(Sx, logarithm, normalize, cut_value)
        Sy = normalize_draw(Sy, logarithm, normalize, cut_value)
        Sz = normalize_draw(Sz, logarithm, normalize, cut_value)


        S_max = np.max((Sx, Sy, Sz))
        S_min = np.min((Sx, Sy, Sz))
        S_lim = np.max((abs(S_max), np.abs(S_min)))

        fig, axs = plt.subplots(nrows=1, ncols=3, sharex=True,  figsize=(2 * tx, 1 * ty))
        plt.subplot(1, 3, 1)
        plt.title(r"$S_z$")
        id_fig, ax, IDimage = draw2D_xz(Sx, z0, x0, axs[0], xlabel=r'z ($\mu$m)', ylabel=r'x ($\mu$m)', title=r"$S_x$", cmap=cmap)
        plt.axis(scale)
        draw_edges(self, plt,  draw_borders,  **kwargs)
        IDimage.set_clim(-S_lim, S_lim)
        # axes[0].set_axis_off()

        plt.subplot(1, 3, 2)
        plt.title(r"$S_y$")
        id_fig, ax, IDimage = draw2D_xz(Sy, z0, x0, axs[1], xlabel=r'z ($\mu$m)', ylabel='', title=r"$S_y$", cmap=cmap)
        plt.axis(scale)
        draw_edges(self, plt,  draw_borders, **kwargs)
        IDimage.set_clim(-S_lim, S_lim)
        # axes[1].set_axis_off()

        plt.subplot(1, 3, 3)
        id_fig, ax, IDimage = draw2D_xz(Sz, z0, x0,axs[2], xlabel=r'z ($\mu$m)', ylabel='', title=r"$S_z$", cmap=cmap)
        plt.title(r"$S_z$")
        plt.axis(scale)
        draw_edges(self, plt,  draw_borders, **kwargs)
        IDimage.set_clim(-S_lim, S_lim)
        # axes[2].set_axis_off()

        cb_ax = fig.add_axes([0.1, 0, 0.8, 0.05])
        cbar = fig.colorbar(id_fig, cmap=cmap, cax=cb_ax, orientation='horizontal', shrink=0.5)

        plt.tight_layout()   

    def __draw_poynting_total__(self,
        logarithm,
        normalize,
        cut_value,
        draw_borders=False,
        scale = 'scaled',
        cmap=CONF_DRAWING["color_intensity"],
        edge=None,
        **kwargs
        ):

        z0 = self.z
        x0 = self.x
        
        tx, ty = rcParams["figure.figsize"]
        
        S = self.get('poynting_total', matrix=True)
        S = normalize_draw(S, logarithm, normalize, cut_value)

        fig, axs = plt.subplots(nrows=1, ncols=1)
        
        id_fig, ax, IDimage = draw2D_xz(
            S, self.z, self.x, ax=axs, xlabel=r"z $(\mu m)$", ylabel=r"x $(\mu m)$",
            cmap=CONF_DRAWING["color_intensity"], title=r'$S_{total}$')
        plt.axis(scale)
        draw_edges(self, plt, draw_borders, **kwargs)
        
        IDimage.set_clim(vmin=0)                
        cb_ax = fig.add_axes([0.2, 0, 0.6, 0.025])
        cbar = fig.colorbar(id_fig, cmap=cmap, cax=cb_ax, orientation='horizontal', shrink=0.5)
        plt.tight_layout() 



    def __draw_energy_density__(self,
        logarithm,
        normalize,
        cut_value,
        draw_borders=False,
        scale = 'scaled',
        cmap=CONF_DRAWING["color_intensity"],
        edge=None,
        **kwargs
        ):

        z0 = self.z
        x0 = self.x
        
        tx, ty = rcParams["figure.figsize"]

        S = self.get('energy_density', matrix=True)
        S = np.real(S)
        S = normalize_draw(S, logarithm, normalize, cut_value)
        
        fig, axs = plt.subplots(nrows=1, ncols=1)
        
        id_fig, ax, IDimage = draw2D_xz(
            S, self.z, self.x, ax=axs, xlabel=r"z $(\mu m)$", ylabel=r"x $(\mu m)$", 
            cmap=CONF_DRAWING["color_intensity"], title=r'energy density')
        plt.axis(scale)
        draw_edges(self, plt, draw_borders, **kwargs)
        
        IDimage.set_clim(vmin=0)                
        cb_ax = fig.add_axes([0.2, 0, 0.6, 0.025])
        cbar = fig.colorbar(id_fig, cmap=cmap, cax=cb_ax, orientation='horizontal', shrink=0.5)
        plt.tight_layout() 



    def __draw_irradiance__(self,
        logarithm,
        normalize,
        cut_value,
        draw_borders=False,
        scale = 'scaled',
        cmap=CONF_DRAWING["color_intensity"],
        edge=None,
        mode='modulus',
        **kwargs
        ):

        z0 = self.z
        x0 = self.x
        
        tx, ty = rcParams["figure.figsize"]

        
        S = self.get('irradiance', mode=mode, matrix=True)
        S = np.real(S)
        S = normalize_draw(S, logarithm, normalize, cut_value)


        fig, axs = plt.subplots(nrows=1, ncols=1)
        
        id_fig, ax, IDimage = draw2D_xz(
            S, self.z, self.x, ax=axs, xlabel=r"z $(\mu m)$", ylabel=r"x $(\mu m)$",
            cmap=CONF_DRAWING["color_intensity"], title=r'irradiance')
        plt.axis(scale)
        draw_edges(self, plt, draw_borders, **kwargs)
        
        IDimage.set_clim(vmin=0)                
        cb_ax = fig.add_axes([0.2, 0, 0.6, 0.025])
        cbar = fig.colorbar(id_fig, cmap=cmap, cax=cb_ax, orientation='horizontal', shrink=0.5)
        plt.tight_layout() 


    def __draw_stokes__(self,  logarithm: float,  normalize: bool,  cut_value: float,
        draw_borders=False,
        scale = 'scaled',
        color_intensity=CONF_DRAWING["color_intensity"],
        color_stokes=CONF_DRAWING["color_stokes"], 
        orientation = 'horizontal', **kwargs
    ):
        """__internal__: computes and draws CI, CQ, CU, CV parameters
        
        The polarization state is obtained with Ex and Ey. If there is Ez field it is not considered. 

        TODO: Include Ez field in the Stokes parameters.
        """


        tx, ty = rcParams["figure.figsize"]

        S0, S1, S2, S3 = self.get('stokes')

        S0 = normalize_draw(S0, logarithm, normalize, cut_value)
        S1 = normalize_draw(S1, logarithm, normalize, cut_value)
        S2 = normalize_draw(S2, logarithm, normalize, cut_value)
        S3 = normalize_draw(S3, logarithm, normalize, cut_value)

        intensity_max = S0.max()

        if orientation=='horizontal':
            plt.figure(figsize=(3 * tx, 1 * ty))
            h1 = plt.subplot(1,4,1)
        elif orientation == 'vertical':
            plt.figure(figsize=(1 * tx, 3 * ty))
            h1 = plt.subplot(4,1,1)
        else:
            plt.figure(figsize=(1.5 * tx, 1.5 * ty))
            h1 = plt.subplot(2,2,1)

        self.__draw1__(S0, color_intensity, r"$S_0$")
        plt.axis(scale)
        draw_edges(self, plt,  draw_borders, color='w.')
        plt.clim(0, intensity_max)

        if orientation=='horizontal':
            h2 = plt.subplot(1,4,2)
        elif orientation == 'vertical':
            h2 = plt.subplot(4,1,2)
        else:
            h2 = plt.subplot(2,2,2)
        self.__draw1__(S1, color_stokes, r"$S_1$")
        plt.axis(scale)
        draw_edges(self, plt,  draw_borders, color='k.')
        plt.clim(-intensity_max, intensity_max)

        if orientation=='horizontal':
            h3 = plt.subplot(1,4,3)
        elif orientation == 'vertical':
            h3 = plt.subplot(4,1,3)
        else:
            h3 = plt.subplot(2,2,3)
        self.__draw1__(S2, color_stokes, r"$S_2$")
        plt.axis(scale)
        draw_edges(self, plt,  draw_borders, color='k.')
        plt.clim(-intensity_max, intensity_max)

        if orientation=='horizontal':
            h4 = plt.subplot(1,4,4)
        elif orientation == 'vertical':
            h4 = plt.subplot(4,1,4)
        else:
            h4 = plt.subplot(2,2,4)
        self.__draw1__(S3, color_stokes, r"$S_3$")
        plt.axis(scale)
        draw_edges(self, plt,  draw_borders, color='k.')
        plt.clim(-intensity_max, intensity_max)

        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.05, hspace=0)
        plt.tight_layout()
        return (h1, h2, h3, h4)
    
    

    def __draw_param_ellipse__(
        self, logarithm: float,  normalize: bool,  cut_value: float,
        draw_borders=False,
        scale = 'scaled',
        color_intensity=CONF_DRAWING["color_intensity"],
        color_phase=CONF_DRAWING["color_phase"], **kwargs
    ):
        """__internal__: computes and draws polariations ellipses.
        Args:
            color_intensity (_type_, optional): _description_. Defaults to CONF_DRAWING["color_intensity"].
            color_phase (_type_, optional): _description_. Defaults to CONF_DRAWING["color_phase"].

        Returns:
            _type_: _description_
        """

        A, B, theta, h = self.get('params_ellipse')

        A = reduce_matrix_size(self.reduce_matrix, self.x, self.z, A)
        B = reduce_matrix_size(self.reduce_matrix, self.x, self.z, B)
        theta = reduce_matrix_size(self.reduce_matrix, self.x, self.z, theta)
        h = reduce_matrix_size(self.reduce_matrix, self.x, self.z, h)

        A = normalize_draw(A, logarithm, normalize, cut_value)
        B = normalize_draw(B, logarithm, normalize, cut_value)


        tx, ty = rcParams["figure.figsize"]

        fig, axs = plt.subplots(
                nrows=2, ncols=2, sharex=True, sharey=True, figsize=(1.5 * tx, 2 * ty)
            )

        max_intensity = max(A.max(), B.max())

        cmap_intensity = CONF_DRAWING["color_intensity"]
        cmap_phase = CONF_DRAWING["color_phase"]    

        kwargs2 = dict(kwargs)
        kwargs2['color'] = 'w.'

        id_fig, ax, IDimage = draw2D_xz(
            A, self.z, self.x, ax=axs[0, 0], scale=scale, xlabel="", ylabel=r"x $(\mu m)$", cmap=cmap_intensity, title=r'A')
        draw_edges(self, axs[0, 0], draw_borders, **kwargs2)
        IDimage.set_clim(0,max_intensity)
        
        id_fig, ax, IDimage = draw2D_xz(
            B, self.z, self.x, ax=axs[0, 1], scale=scale, xlabel="", ylabel="", cmap=cmap_intensity, title=r'B')
        draw_edges(self, axs[0, 1], draw_borders, **kwargs2)
        IDimage.set_clim(0,max_intensity)

        id_fig, ax, IDimage = draw2D_xz(
            theta/degrees, self.z, self.x, ax=axs[1, 0], scale=scale, xlabel=r"z $(\mu m)$", ylabel=r"x $(\mu m)$", cmap=cmap_phase, title=r'$\theta$')
        draw_edges(self, axs[1, 0], draw_borders, **kwargs)
        IDimage.set_clim(-180, 180)
        
        id_fig, ax, IDimage = draw2D_xz(
            h, self.z, self.x, ax=axs[1, 1], scale=scale, xlabel=r"z $(\mu m)$", ylabel="", cmap=cmap_phase, title=r'h')
        draw_edges(self, axs[1, 1], draw_borders, **kwargs)
        IDimage.set_clim(-1,1)


        fig.subplots_adjust(right=1.25)
        cb_ax = fig.add_axes([0.2, 0, 0.6, 0.025])
        cbar = fig.colorbar(id_fig, cmap=cmap_intensity, cax=cb_ax, orientation='horizontal', shrink=0.5)
        plt.tight_layout()


        # plt.figure(figsize=(2 * tx, 2 * ty))

        # max_intensity = max(A.max(), B.max())

        # h1 = plt.subplot(2, 2, 1)
        # self.__draw1__(A, color_intensity, "$A$")
        # plt.clim(0, max_intensity)
        # h2 = plt.subplot(2, 2, 2)
        # self.__draw1__(B, color_intensity, "$B$")
        # plt.clim(0, max_intensity)

        # h3 = plt.subplot(2, 2, 3)
        # self.__draw1__(theta/degrees, color_phase, r"$\phi$")
        # plt.clim(-180, 180)
        # h4 = plt.subplot(2, 2, 4)
        # self.__draw1__(h, "gist_heat", "$h$")
        # plt.tight_layout()
        # return (h1, h2, h3, h4)

    def __draw_ellipses__(
        self,
        logarithm: float = 0.,
        normalize: bool = False,
        cut_value="",
        draw_borders=False,
        scale='scaled',
        num_ellipses=(31, 31),
        amplification=0.75,
        color_line="w",
        line_width=.75,
        draw_arrow=True,
        head_width=.5,
        ax=False,
        color_intensity=CONF_DRAWING["color_intensity"], 
        **kwargs
    ):
    
        """

        Args:
            logarithm (float, optional): _description_. Defaults to 0..
            normalize (bool, optional): _description_. Defaults to False.
            cut_value (str, optional): _description_. Defaults to "".
            num_ellipses (tuple, optional): _description_. Defaults to (21, 21).
            amplification (float, optional): _description_. Defaults to 0.75.
            color_line (str, optional): _description_. Defaults to "w".
            line_width (int, optional): _description_. Defaults to 1.
            draw_arrow (bool, optional): _description_. Defaults to True.
            head_width (int, optional): _description_. Defaults to 2.
            ax (bool, optional): _description_. Defaults to False.
            color_intensity (_type_, optional): _description_. Defaults to CONF_DRAWING["color_intensity"].
        """

        percentage_intensity = CONF_DRAWING["percentage_intensity"]
        intensity_max = (np.abs(self.Ex) ** 2 + np.abs(self.Ey) ** 2).max()

        Dx = self.x[-1] - self.x[0]
        Dz = self.z[-1] - self.z[0]
        size_x = Dx / (num_ellipses[0])
        size_z = Dz / (num_ellipses[1])
        x_centers = size_x/2 + size_x * np.array(range(0, num_ellipses[0]))
        z_centers = size_z/2 + size_z * np.array(range(0, num_ellipses[1]))

        num_x, num_z = len(self.x), len(self.z)
        ix_centers = num_x / (num_ellipses[0])
        iz_centers = num_z / (num_ellipses[1])

        ix_centers = (
            np.round(ix_centers/2 + ix_centers * np.array(range(0, num_ellipses[0])))
        ).astype("int")
        iz_centers = (
            np.round(iz_centers/2 + iz_centers * np.array(range(0, num_ellipses[1])))
        ).astype("int")

        Ix_centers, Iz_centers = np.meshgrid(
            ix_centers.astype("int"), iz_centers.astype("int")
        )

        verbose = False
        if verbose is True:
            print(num_x, num_z, ix_centers, iz_centers)
            print(Dx, Dz, size_x, size_z)
            print(x_centers, z_centers)
            print(Ix_centers, Iz_centers)

        E0x = self.Ex[Iz_centers, Ix_centers]
        E0y = self.Ey[Iz_centers, Ix_centers]

        angles = np.linspace(0, 360*degrees, 64)

        if ax is False:
            id_fig, ax, IDimage=self.__draw_intensity__( logarithm=logarithm, normalize=normalize, 
                            cut_value=cut_value, draw_borders=draw_borders, scale = scale)

        for i, xi in enumerate(ix_centers):
            for j, yj in enumerate(iz_centers):
                Ex = np.real(E0x[j, i] * np.exp(1j * angles))
                Ey = np.real(E0y[j, i] * np.exp(1j * angles))

                max_r = np.sqrt(np.abs(Ex) ** 2 + np.abs(Ey) ** 2).max()
                size_dim = min(size_x, size_z)

                if max_r > 0 and max_r**2 > percentage_intensity * intensity_max:
                    Ex = Ex / max_r * size_dim * amplification/2 + self.x[int(xi)]
                    Ey = Ey / max_r * size_dim * amplification/2 + self.z[int(yj)]

                    ax.plot(Ey, Ex, color_line, lw=line_width)
                    if draw_arrow:
                        ax.arrow( Ey[0], Ex[0], Ey[0] - Ey[1], Ex[0] - Ex[1],
                            width=0, head_width=head_width, fc=color_line,
                            ec=color_line, length_includes_head=False,
                        )


    def __draw_directions__( self,  logarithm: bool = False,
                                  normalize: bool = False,
                                  cut_value: float = 0,
                                  draw_borders: bool = True,
                                  scale: str = 'scaled',
                                  cmap=CONF_DRAWING["color_directions"],
                                  colorbar_kind= 'vertical',
                                  percentage_intensity: float = 0.01,
                                  **kwargs
                              ):
        """Draws directions of the fields.
        Args:
            logarithm (float): If >0, intensity is scaled in logarithm
            normalize (bool): If True, max(intensity)=1
            cut_value (float): If not None, cuts the maximum intensity to this value
            draw_borders (bool): If True draw edges of objects
            scale (str): '', 'scaled', 'equal', scales the XY drawing
            percentage_intensity (float): minimum intensity to draw directions, relative to maximum intensity
        """

        Sx, Sy, Sz = self.get('poynting_vector_averaged')

        direction = np.arctan2(Sx, Sz)

        irradiance = self.get('irradiance')


        drawing = direction/degrees
        drawing[irradiance <percentage_intensity * (irradiance.max())] = 0


        z0 = self.z
        x0 = self.x
        
        tx, ty = rcParams["figure.figsize"]

        
        S = self.get('irradiance', matrix=True)
        S = np.real(S)
        S = normalize_draw(S, logarithm, normalize, cut_value)


        id_fig, IDax, IDimage = draw2D(drawing.transpose(), x = self.z, y = self.x, color=CONF_DRAWING["color_directions"], scale=scale)
        plt.title('Direction of Poynting vector')
        ax_bar =plt.colorbar(orientation='vertical')
        plt.clim(-180,180)
        ax_bar.set_ticks(np.arange(-180, 181, 45))
        draw_edges(self, plt, draw_borders, **kwargs)

        plt.tight_layout() 


    def __draw_arrows__( self,  logarithm: bool = False,
                                  normalize: bool = False,
                                  cut_value: float = 0,
                                  draw_borders: bool = True,
                                  scale: str = 'scaled',
                                  cmap=CONF_DRAWING["color_arrows"],
                                  colorbar_kind= 'vertical',
                                  sep_x: int = 1,
                                  sep_z: int = 1,
                                  size_arrow: float = 10,
                                  **kwargs
                              ):
        """Draws directions of the fields.
        Args:
            logarithm (float): If >0, intensity is scaled in logarithm
            normalize (bool): If True, max(intensity)=1
            cut_value (float): If not None, cuts the maximum intensity to this value
            draw_borders (bool): If True draw edges of objects
            scale (str): '', 'scaled', 'equal', scales the XY drawing
            percentage_intensity (float): minimum intensity to draw directions, relative to maximum intensity
        """

        Sx, Sy, Sz = self.get('poynting_vector_averaged')

        direction = np.arctan2(Sx, Sz)



        fig1, ax1 = plt.subplots()

        SZ_final = Sz * size_arrow
        SX_final = Sx * size_arrow

        ax1.set_title('Arrows')
        Q = ax1.quiver(self.Z[::sep_z, ::sep_x], self.X[::sep_z, ::sep_x],
                        SZ_final[::sep_z, ::sep_x], SX_final[::sep_z, ::sep_x], 
                        direction[::sep_z, ::sep_x], cmap=cmap, scale=.5)

        factor = 180 / np.pi  # Example: convert radians to degrees
        cbar = plt.colorbar(Q, ax=ax1, orientation=colorbar_kind)
        tick_vals = np.array([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])  # in radians
        cbar.set_ticks(tick_vals)
        cbar.set_ticklabels([f"{tick * factor:.1f}" for tick in tick_vals])

        if scale not in ("", None, []):
            plt.axis(scale)

        draw_edges(self, plt, draw_borders, **kwargs)
        plt.xlabel(r'z ($\mu m$)')
        plt.ylabel(r'x ($\mu m$)')



    @check_none('x', 'z', 'n')
    def __draw_refractive_index__(self,
                                  logarithm: bool = False,
                                  normalize: bool = False,
                                  cut_value: float = 0,
                                  draw_borders: bool = True,
                                  scale: str = 'scaled',
                                  colorbar_kind= 'vertical'
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

        colormap_kind= cm.Blues
        edge_matrix = None
        min_incr = 0.01
        reduce_matrix = 'standard'

        title: str = ''
        filename: str = ''

        kind = 'all'


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
        if n_draw.min() < 1:
            if n_draw.max() > 100:
                plt.clim(0, 100)
            else:
                plt.clim(0, n_draw.max())
        elif n_draw.min() > 1:
            if n_draw.max() > 100:
                plt.clim(n_draw.min(), 100)
            else:
                plt.clim(n_draw.min(), n_draw.max())


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
    
    
    def __draw1__(self, image, colormap, title: str = "", has_max=False):
        """_summary_

        Args:
            image (_type_): _description_
            colormap (_type_): _description_
            title (str, optional): _description_. Defaults to "".
            has_max (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        extension = [self.z[0], self.z[-1], self.x[0], self.x[-1]]

        h = plt.imshow(
            image.transpose(),
            interpolation="bilinear",
            aspect="auto",
            origin="lower",
            extent=extension,
        )
        h.set_cmap(colormap)
        plt.axis(extension)

        plt.title(title, fontsize=16)

        if has_max is True:
            text_up = "{}".format(image.max())
            plt.text(
                self.x.max(),
                self.z.max(),
                text_up,
                fontsize=14,
                bbox=dict(edgecolor="white", facecolor="white", alpha=0.75),
            )

            text_down = "{}".format(image.min())
            plt.text(
                self.x.max(),
                self.z.min(),
                text_down,
                fontsize=14,
                bbox=dict(edgecolor="white", facecolor="white", alpha=0.75),
            )

        plt.xlabel(r"z $(\mu m)$")
        plt.ylabel(r"x $(\mu m)$")
        if colormap is not None:
            plt.colorbar(orientation="horizontal", fraction=0.046, shrink=0.5)
            h.set_clim(0, image.max())

        return h


def FP_PWD_kernel_simple(Ex, Ey, n1, n2, k0, kx, wavelength, dz):
    """Step for Plane wave decomposition (PWD) algorithm.

    Args:
        Ex (np.array): field Ex
        Ey (np.array): field Ey
        n1 (np.array): refractive index at the first layer
        n2 (np.array): refractive index at the second layer
        k0 (float): wavenumber
        kx (np.array): transversal wavenumber
        wavelength (float): wavelength
        dz (float): increment in distances: z[1]-z[0]

    Returns:
        E  list(Ex, Ey, Ez): Field E(z+dz) at at distance dz from the incident field.
        H  list(Ex, Ey, Ez): Field H(z+dz) at at distance dz from the incident field.
        
    """

    # amplitude of waveplanes
    Exk = fftshift(fft(Ex))
    Eyk = fftshift(fft(Ey))


    kr = n1 * k0 # first layer
    ks = n2 * k0 # second layer
            
    ky = np.zeros_like(kx) # we are in XZ frame
    k_perp2 = kx**2 + ky**2

    kz_r = np.sqrt(kr**2 - k_perp2) # first layer
    kz_s = np.sqrt(ks**2 - k_perp2) # second layer

    kr = kr.astype(np.complex128)
    ks = ks.astype(np.complex128)

    P = np.exp(1j * kz_s * dz)
    P_factor = P / (k_perp2*kr*ks)
    Gamma = kz_r*kz_s + kz_s * k_perp2 / kz_r
    

    # Fresnel coefficients
    t_TM, t_TE, _, _ = fresnel_equations_kx(kx, wavelength, n1, n2, [1, 1, 0, 0], has_draw=False)

    t_TM = t_TM.astype(np.complex128)
    t_TE = t_TE.astype(np.complex128)
        
    T00 = P_factor * (t_TM*kx**2*Gamma + t_TE*ky**2*kr*ks)
    T01 = P_factor * (t_TM*kx*ky*Gamma - t_TE*kx*ky*kr*ks)  
    T10 = P_factor * (t_TM*kx*ky*Gamma - t_TE*kx*ky*kr*ks)  
    T11 = P_factor * (t_TM*ky**2*Gamma + t_TE*kx**2*kr*ks)  
    
    # Simpler since ky = 0, but keep to translate to 3D 
    
    # T00 = P * (t_TM*kx**2*Gamma) / (k_perp2*kr*ks) 
    # T01 = np.zeros_like(kx) 
    # T10 = np.zeros_like(kx)  
    # T11 = P * (t_TE*kx**2*kr*ks) / (k_perp2*kr*ks) 
    
    nan_indices = np.where(np.isnan(T00)) 
    
    option = 1 # TODO: fix better
    
    if option == 1:

        T00[nan_indices]=T00[nan_indices[0]-1]
        T01[nan_indices]=T01[nan_indices[0]-1]
        T10[nan_indices]=T10[nan_indices[0]-1]
        T11[nan_indices]=T11[nan_indices[0]-1] 
        
    elif option == 2:
    
        if len(nan_indices)>0:
            T00_b = P * (t_TM*kx**2*Gamma + t_TE*ky**2*kr*ks) / (k_perp2*kr*ks+1e-10) 
            T01_b = P * (t_TM*kx*ky*Gamma - t_TE*kx*ky*kr*ks) / (k_perp2*kr*ks+1e-10) 
            T10_b = P * (t_TM*kx*ky*Gamma - t_TE*kx*ky*kr*ks) / (k_perp2*kr*ks+1e-10) 
            T11_b = P * (t_TM*ky**2*Gamma + t_TE*kx**2*kr*ks) / (k_perp2*kr*ks+1e-10) 
        
            T00[nan_indices]=T00_b[nan_indices]
            T01[nan_indices]=T01_b[nan_indices]
            T10[nan_indices]=T10_b[nan_indices]
            T11[nan_indices]=T11_b[nan_indices] 
    
    ex0 = T00 * Exk + T01 * Eyk
    ey0 = T10 * Exk + T11 * Eyk 
    ez0 = - (kx*ex0+ky*ey0) / (kz_s)
    
    # ex0 = T00 * Exk 
    # ey0 = T11 * Eyk 
    # ez0 = - (kx*ex0+ky*ey0) / (kz_r)
    
    
    TM00 = -kx*ky*Gamma 
    TM01 = -(ky*ky*Gamma + kz_s**2)
    TM10 = +(kx*kx*Gamma + kz_s**2)
    TM11 = +kx*ky*Gamma
    TM20 = -ky*kz_s
    TM21 = +kx*kz_s
    
    Z0 = 376.82  # ohms (impedance of free space)
    H_factor = n2 / (ks * kz_s * Z0)
    
    hx0 = (TM00*ex0+TM01*ey0) * H_factor
    hy0 = (TM10*ex0+TM11*ey0) * H_factor
    hz0 = (TM20*ex0+TM21*ey0) * H_factor
        

    Ex_final = ifft(ifftshift(ex0))
    Ey_final = ifft(ifftshift(ey0))
    Ez_final = ifft(ifftshift(ez0))


    Hx_final = ifft(ifftshift(hx0))
    Hy_final = ifft(ifftshift(hy0))
    Hz_final = ifft(ifftshift(hz0))

    return (Ex_final, Ey_final, Ez_final), (Hx_final, Hy_final, Hz_final)





def FP_PWD_kernel_simple_proposal(Ex, Ey, n1, n2, k0, kx, wavelength, dz):
    """Step for Plane wave decomposition (PWD) algorithm.

    Args:
        Ex (np.array): field Ex
        Ey (np.array): field Ey
        n1 (np.array): refractive index at the first layer
        n2 (np.array): refractive index at the second layer
        k0 (float): wavenumber
        kx (np.array): transversal wavenumber
        wavelength (float): wavelength
        dz (float): increment in distances: z[1]-z[0]

    Returns:
        E  list(Ex, Ey, Ez): Field E(z+dz) at at distance dz from the incident field.
        H  list(Ex, Ey, Ez): Field H(z+dz) at at distance dz from the incident field.
        
    """

    # amplitude of waveplanes
    Exk = fftshift(fft(Ex))
    Eyk = fftshift(fft(Ey))


    kr = n1 * k0 # first layer
    # print(n1, n2)
    ks = n2 * k0 # second layer

    ky = np.zeros_like(kx, dtype=np.complex128) # we are in XZ frame
    k_perp2 = (kx**2 + ky**2).astype(np.complex128)
    kr = kr.astype(np.complex128)
    ks = ks.astype(np.complex128)

    kz_r = np.sqrt(kr**2 - k_perp2) # first layer
    kz_s = np.sqrt(ks**2 - k_perp2) # second layer
    

    P = np.exp(1j * kz_s * dz)
    Pprima = P / (k_perp2*kr*ks+1e-40)
    Gamma = kz_r*kz_s + kz_s * k_perp2 / kz_r
    

    # Fresnel coefficients
    t_TM, t_TE, _, _ = fresnel_equations_kx(kx, wavelength, n1, n2, [1, 1, 0, 0], has_draw=False)

    t_TM = t_TM.astype(np.complex128)
    t_TE = t_TE.astype(np.complex128)
 
    
    T00 = Pprima * (t_TM*kx**2*Gamma + t_TE*ky**2*kr*ks) 
    T01 = Pprima * (t_TM*kx*ky*Gamma - t_TE*kx*ky*kr*ks) 
    T10 = Pprima * (t_TM*kx*ky*Gamma - t_TE*kx*ky*kr*ks)
    T11 = Pprima * (t_TM*ky**2*Gamma + t_TE*kx**2*kr*ks) 

    # Simpler since ky = 0, but keep to translate to 3D
    
    # T00 = Pprima * (t_TM*kx**2*Gamma) / (k_perp2*kr*ks) 
    # T01 = np.zeros_like(kx) 
    # T10 = np.zeros_like(kx)  
    # T11 = Pprima * (t_TE*kx**2*kr*ks) / (k_perp2*kr*ks) 
    
   
    
    ex0 = T00 * Exk + T01 * Eyk
    ey0 = T10 * Exk + T11 * Eyk 
    ez0 = - (kx*ex0+ky*ey0) / kz_s
    
    # ex0 = T00 * Exk 
    # ey0 = T11 * Eyk 
    # ez0 = - (kx*ex0+ky*ey0) / kz_s
    

    # thesis Fertig 2011 (3.40) pág 66 I do not feel confident yet
    TM00 = -kx*ky*Gamma 
    TM01 = -(ky*ky*Gamma + kz_s**2)
    TM10 = +(kx*kx*Gamma + kz_s**2)
    TM11 = +kx*ky*Gamma
    TM20 = -ky*kz_s
    TM21 = +kx*kz_s
    
    Z0 = 376.82  # ohms (impedance of free space)
    H_factor = n2 / (ks * kz_s * Z0)
    
    hx0 = (TM00*ex0+TM01*ey0) * H_factor
    hy0 = (TM10*ex0+TM11*ey0) * H_factor
    hz0 = (TM20*ex0+TM21*ey0) * H_factor
        

    Ex_final = ifft(ifftshift(ex0))
    Ey_final = ifft(ifftshift(ey0))
    Ez_final = ifft(ifftshift(ez0))


    Hx_final = ifft(ifftshift(hx0))
    Hy_final = ifft(ifftshift(hy0))
    Hz_final = ifft(ifftshift(hz0))

    return (Ex_final, Ey_final, Ez_final), (Hx_final, Hy_final, Hz_final)



def FP_WPM_schmidt_kernel(Ex, Ey, n1, n2, k0, kx, wavelength, dz):
    """
    Kernel for fast propagation of WPM method

    Args:
        Ex (np.array): field Ex
        Ey (np.array): field Ey
        n1 (np.array): refractive index at the first layer
        n2 (np.array): refractive index at the second layer
        k0 (float): wavenumber
        kx (np.array): transversal wavenumber
        wavelength (float): wavelength
        dz (float): increment in distances: z[1]-z[0]

    Returns:
        E  list(Ex, Ey, Ez): Field E(z+dz) at at distance dz from the incident field.
        H  list(Hx, Hy, Hz): Field H(z+dz) at at distance dz from the incident field.

    References:

        1. M. W. Fertig and K.-H. Brenner, “Vector wave propagation method,” J. Opt. Soc. Am. A, vol. 27, no. 4, p. 709, 2010.

        2. S. Schmidt et al., “Wave-optical modeling beyond the thin-element-approximation,” Opt. Express, vol. 24, no. 26, p. 30188, 2016.
    """
    Nr = np.unique(n1)
    Ns = np.unique(n2)

    Ex_final = np.zeros_like(Ex, dtype=complex)
    Ey_final = np.zeros_like(Ex, dtype=complex)
    Ez_final = np.zeros_like(Ex, dtype=complex)

    Hx_final = np.zeros_like(Ex, dtype=complex)
    Hy_final = np.zeros_like(Ex, dtype=complex)
    Hz_final = np.zeros_like(Ex, dtype=complex)
  

    for r, n_r in enumerate(Nr):
        for s, n_s in enumerate(Ns):
            Imz = np.array(np.logical_and(n1 == n_r, n2 == n_s))
            E, H = FP_PWD_kernel_simple(Ex, Ey, n_r, n_s, k0, kx, wavelength, dz)

            Ex_final = Ex_final + Imz * E[0]
            Ey_final = Ey_final + Imz * E[1]
            Ez_final = Ez_final + Imz * E[2]
            Hx_final = Hx_final + Imz * H[0]
            Hy_final = Hy_final + Imz * H[1]
            Hz_final = Hz_final + Imz * H[2]
            
    return (Ex_final, Ey_final, Ez_final), (Hx_final, Hy_final, Hz_final)



def draw2D_xz(
        image,
        x,
        y,
        ax=None,
        xlabel=r"x $(\mu m)$",
        ylabel=r"$y  (\mu m)$",
        title="",
        cmap="YlGnBu",  # YlGnBu  seismic
        interpolation='bilinear',  # 'bilinear', 'nearest'
        scale='scaled',
        reduce_matrix='standard',
        range_scale='um',
        verbose=False):
    """makes a drawing of XY

    Args:
        image (numpy.array): image to draw
        x (numpy.array): positions x
        y (numpy.array): positions y
        ax (): axis
        xlabel (str): label for x
        ylabel (str): label for y
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
    if reduce_matrix in (None, '', []):
        pass
    elif reduce_matrix == 'standard':
        num_x = len(x)
        num_y = len(y)
        reduction_x = int(num_x / 500)
        reduction_y = int(num_x / 500)

        if reduction_x == 0:
            reduction_x = 1
        if reduction_y == 0:
            reduction_y = 1

        image = image[::reduction_x, ::reduction_y]
    else:
        image = image[::reduce_matrix[0], ::reduce_matrix[1]]

    if verbose is True:
        print(("image size {}".format(image.shape)))

    if ax is None:
        id_fig = plt.figure()
        ax = id_fig.add_subplot(111)
    else:
        id_fig = None

    if range_scale == 'um':
        extension = (x[0], x[-1], y[0], y[-1])
    else:
        extension = (x[0] / mm, x[-1] / mm, y[0] / mm, y[-1] / mm)
        xlabel = "x (mm)"
        ylabel = "y (mm)"

    IDimage = ax.imshow(image.transpose(),
                        interpolation=interpolation,
                        aspect='auto',
                        origin='lower',
                        extent=extension,
                        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if scale != '':
        ax.axis(scale)

    IDimage.set_cmap(cmap)

    return id_fig, ax, IDimage





def BWPM_kernel_simple(Ex, Ey, n1, n2, k0, kx, wavelength, dz):
    """Step for Plane wave decomposition (PWD) algorithm.

    Args:
        Ex (np.array): field Ex
        Ey (np.array): field Ey
        n1 (np.array): refractive index at the first layer
        n2 (np.array): refractive index at the second layer
        k0 (float): wavenumber
        kx (np.array): transversal wavenumber
        wavelength (float): wavelength
        dz (float): increment in distances: z[1]-z[0]

    Returns:
        E  list(Ex, Ey, Ez): Field E(z+dz) at at distance dz from the incident field.
        H  list(Ex, Ey, Ez): Field H(z+dz) at at distance dz from the incident field.
        
    """

    # amplitude of waveplanes
    Exk = fftshift(fft(Ex))
    Eyk = fftshift(fft(Ey))


    kr = n1 * k0 # first layer
    ks = n2 * k0 # second layer

    kr = kr.astype(np.complex128)
    ks = ks.astype(np.complex128)
            
    ky = np.zeros_like(kx) # we are in XZ frame
    k_perp2 = kx**2 + ky**2

    kz_r = np.sqrt(kr**2 - k_perp2) # first layer
    kz_s = np.sqrt(ks**2 - k_perp2) # second layer

    P = np.exp(1j * kz_s * dz)
    P_factor = P/(k_perp2*kr**2) 
    Gamma = kz_r*kz_s + kz_s * k_perp2 / kz_r
    

    # Fresnel coefficients
    _,_, r_TM, r_TE = fresnel_equations_kx(kx, wavelength, n1, n2, [0,0,1,1], has_draw=False)
    

    R00 = P_factor * (-r_TM*kx**2*Gamma + r_TE*ky**2*kr**2) 
    R01 = P_factor * (-r_TM*kx*ky*Gamma - r_TE*kx*ky*kr**2)
    R10 = P_factor * (-r_TM*kx*ky*Gamma - r_TE*kx*ky*kr**2)
    R11 = P_factor * (-r_TM*ky**2*Gamma + r_TE*kx**2*kr**2)

    # Simpler since ky = 0, but keep to translate to 3D
    
    # R00 = P * (r_TM*kx**2*Gamma) / (k_perp2*kr**2) 
    # R01 = np.zeros_like(kx) 
    # R10 = np.zeros_like(kx)  
    # R11 = P * (r_TE*kx**2*kr**2) / (k_perp2*kr**2) 
    
    nan_indices = np.where(np.isnan(R00)) 
    
        
    if len(nan_indices)>0:
        R00_b = P * (-r_TM*kx**2*Gamma + r_TE*ky**2*kr**2) / (k_perp2*kr**2+1e-10) 
        R01_b = P * (-r_TM*kx*ky*Gamma - r_TE*kx*ky*kr**2) / (k_perp2*kr**2+1e-10) 
        R10_b = P * (-r_TM*kx*ky*Gamma - r_TE*kx*ky*kr**2) / (k_perp2*kr**2+1e-10) 
        R11_b = P * (-r_TM*ky**2*Gamma + r_TE*kx**2*kr**2) / (k_perp2*kr**2+1e-10) 
    
        R00[nan_indices]=R00_b[nan_indices]
        R01[nan_indices]=R01_b[nan_indices]
        R10[nan_indices]=R10_b[nan_indices]
        R11[nan_indices]=R11_b[nan_indices] 
    
    ex0 = R00 * Exk + R01 * Eyk
    ey0 = R10 * Exk + R11 * Eyk 
    ez0 = - (kx*ex0+ky*ey0) / (kz_s)
        
    
    TM00 = -kx*ky*Gamma 
    TM01 = -(ky*ky*Gamma + kz_s**2)
    TM10 = +(kx*kx*Gamma + kz_s**2)
    TM11 = +kx*ky*Gamma
    TM20 = -ky*kz_s
    TM21 = +kx*kz_s
    
    Z0 = 376.82  # ohms (impedance of free space)
    H_factor = n2 / (ks * kz_s * Z0)
    
    hx0 = (TM00*ex0+TM01*ey0) * H_factor
    hy0 = (TM10*ex0+TM11*ey0) * H_factor
    hz0 = (TM20*ex0+TM21*ey0) * H_factor
        
    Ex_final = ifft(ifftshift(ex0))
    Ey_final = ifft(ifftshift(ey0))
    Ez_final = ifft(ifftshift(ez0))


    Hx_final = ifft(ifftshift(hx0))
    Hy_final = ifft(ifftshift(hy0))
    Hz_final = ifft(ifftshift(hz0))

    return (Ex_final, Ey_final, Ez_final), (Hx_final, Hy_final, Hz_final)




def BWPM_schmidt_kernel(Ex, Ey, n1, n2, k0, kx, wavelength, dz):
    """
    Kernel for fast propagation of WPM method

    Args:
        Ex (np.array): field Ex
        Ey (np.array): field Ey
        n1 (np.array): refractive index at the first layer
        n2 (np.array): refractive index at the second layer
        k0 (float): wavenumber
        kx (np.array): transversal wavenumber
        wavelength (float): wavelength
        dz (float): increment in distances: z[1]-z[0]

    Returns:
        E  list(Ex, Ey, Ez): Field E(z+dz) at at distance dz from the incident field.
        H  list(Hx, Hy, Hz): Field H(z+dz) at at distance dz from the incident field.

    References:

        1. M. W. Fertig and K.-H. Brenner, “Vector wave propagation method,” J. Opt. Soc. Am. A, vol. 27, no. 4, p. 709, 2010.

        2. S. Schmidt et al., “Wave-optical modeling beyond the thin-element-approximation,” Opt. Express, vol. 24, no. 26, p. 30188, 2016.
    """
    Nr = np.unique(n1)
    Ns = np.unique(n2)

    Ex_final = np.zeros_like(Ex, dtype=complex)
    Ey_final = np.zeros_like(Ex, dtype=complex)
    Ez_final = np.zeros_like(Ex, dtype=complex)

    Hx_final = np.zeros_like(Ex, dtype=complex)
    Hy_final = np.zeros_like(Ex, dtype=complex)
    Hz_final = np.zeros_like(Ex, dtype=complex)


    for r, n_r in enumerate(Nr):
        for s, n_s in enumerate(Ns):
            Imz = np.array(np.logical_and(n1 == n_r, n2 == n_s))
            E, H = BWPM_kernel_simple(Ex, Ey, n_r, n_s, k0, kx, wavelength, dz)

            Ex_final = Ex_final + Imz * E[0]
            Ey_final = Ey_final + Imz * E[1]
            Ez_final = Ez_final + Imz * E[2]
            Hx_final = Hx_final + Imz * H[0]
            Hy_final = Hy_final + Imz * H[1]
            Hz_final = Hz_final + Imz * H[2]
            
    return (Ex_final, Ey_final, Ez_final), (Hx_final, Hy_final, Hz_final)

