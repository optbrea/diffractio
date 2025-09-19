# !/usr/bin/env python3


# ----------------------------------------------------------------------
# Name:        vector_fields_XYZ.py
# Purpose:     Class and methods for handling 3D vector fields
#
# Author:      Luis Miguel Sanchez Brea
#
# Created:     2024
# Licence:     GPLv3
# ----------------------------------------------------------------------

# flake8: noqa

"""
This module generates Vector_field_XYZ class. It is required also for generating masks and fields.
The main atributes are:
    * self.Ex - x component of electric field
    * self.Ey - y component of electric field
    * self.Ez - z component of electric field
    * self.wavelength - wavelength of the incident field. The field is monocromatic
    * self.x - x positions of the field
    * self.y - y positions of the field
    * self.z - z positions of the field
    * self.X (numpy.array): equal size to x * y * z. complex field
    * self.Y (numpy.array): equal size to x * y * z. complex field
    * self.Z (numpy.array): equal size to x * y * z. complex field
    * self.quality (float): quality of RS algorithm
    * self.info (str): description of data
    * self.type (str): Class of the field
    * self.date (str): date when performed


The magnitude is related to microns: `micron = 1.`

*Class for XY vector fields*

*Definition of a scalar field*
    * add, substract fields
    * save, load data, clean, get, normalize
    * cut_resample
    * appy_mask

*Vector parameters*
    * polarization_states
    * polarization_ellipse

*Propagation*
    * RS - Rayleigh Sommerfeld

*Drawing functions*
    * draw: intensity, intensities, phases, fields, stokes, param_ellipse, ellipses

"""
import copy
import time

from .__init__ import degrees, eps, mm, np, plt
from .config import bool_raise_exception, CONF_DRAWING, Draw_Vector_XY_Options, Draw_Vector_XZ_Options, get_vector_options
from .utils_typing import npt, Any, NDArray,  NDArrayFloat, NDArrayComplex
from .utils_common import get_date, load_data_common, save_data_common, check_none, get_vector
from .utils_common import get_instance_size_MB

from .utils_drawing import normalize_draw, reduce_matrix_size
from .utils_math import get_k, nearest
from .utils_optics import normalize_field, fresnel_equations_kx

from .scalar_fields_X import Scalar_field_X
from .scalar_fields_XY import Scalar_field_XY
from .scalar_fields_XZ import Scalar_field_XZ
from .scalar_fields_XYZ import Scalar_field_XYZ
from .scalar_masks_XY import Scalar_mask_XY
from .scalar_masks_XYZ import Scalar_mask_XYZ
from .vector_fields_XY import Vector_field_XY
from .vector_masks_XY import Vector_mask_XY

from py_pol.jones_vector import Jones_vector

from py_pol.jones_vector import Jones_vector

from numpy.lib.scimath import sqrt as csqrt
from scipy.fftpack import fft, fftshift, ifft, ifftshift, fft2, ifft2

from py_pol.jones_vector import Jones_vector

from numpy.lib.scimath import sqrt as csqrt
from scipy.fftpack import fft, fftshift, ifft, ifftshift, fft2, ifft2


percentage_intensity = CONF_DRAWING['percentage_intensity']


class Vector_field_XYZ():
    """Class for vectorial fields.

    Args:
        x (numpy.array): linear array with equidistant positions. The number of data is preferibly 2**n.
        y (numpy.array): linear array with equidistant positions. The number of data is preferibly 2**n.
        z (numpy.array): linear array with equidistant positions. The number of data is preferibly 2**n.
        wavelength (float): wavelength of the incident field
        info (str): String with info about the simulation

    Attributes:
        self.x (numpy.array): linear array with equidistant positions. The number of data is preferibly 2**n.
        self.y (numpy.array): linear array with equidistant positions. The number of data is preferibly 2**n.
        self.z (numpy.array): linear array with equidistant positions. The number of data is preferibly 2**n.
        self.wavelength (float): wavelength of the incident field.
        self.Ex (numpy.array): Electric_x field
        self.Ey (numpy.array): Electric_y field
        self.Ez (numpy.array): Electric_z field
    """

    def __init__(self, x: NDArrayFloat | None = None, y: NDArrayFloat | None = None,
                 z: NDArrayFloat | None = None, wavelength: float | None = None,
                 n_background: float = 1., info: str = ""):
        self.x = x
        self.y = y
        self.z = z
        self.wavelength = wavelength
        self.X, self.Y, self.Z = np.meshgrid(self.x, self.y, self.z)

        self.Ex = np.zeros_like(self.X, dtype=complex)
        self.Ey = np.zeros_like(self.X, dtype=complex)
        self.Ez = np.zeros_like(self.X, dtype=complex)
        self.Hx = None
        self.Hy = None
        self.Hz = None

        self.n = n_background*np.ones_like(self.X, dtype=complex)

        self.Ex0 = None
        self.Ey0 = None
        self.Ez0 = None

        self.reduce_matrix = 'standard'  # 'None, 'standard', (5,5)
        self.n_background = n_background
        self.type = 'Vector_field_XYZ'
        self.info = info
        self.date = get_date()
        self.CONF_DRAWING = CONF_DRAWING


    @check_none('x', 'y', 'z', 'Ex', 'Ey', 'Ez', raise_exception=bool_raise_exception)
    def __str__(self):
        """Represents data from class."""

        intensity = self.intensity()
        Imin = intensity.min()
        Imax = intensity.max()

        print("{}\n - x:  {},   y:  {},  z:  {},   Ex:  {}".format(
            self.type, self.x.shape, self.y.shape, self.z.shape, self.Ex.shape))

        print(
            " - xmin:       {:2.2f} um,  xmax:      {:2.2f} um,  Dx:   {:2.2f} um"
            .format(self.x[0], self.x[-1], self.x[1] - self.x[0]))
        print(
            " - ymin:       {:2.2f} um,  ymay:      {:2.2f} um,  Dy:   {:2.2f} um"
            .format(self.y[0], self.y[-1], self.y[1] - self.y[0]))
        print(
            " - zmin:       {:2.2f} um,  zmaz:      {:2.2f} um,  Dz:   {:2.2f} um"
            .format(self.z[0], self.z[-1], self.z[1] - self.z[0]))
        print(" - Imin:       {:2.2f},     Imax:      {:2.2f}".format(
            Imin, Imax))

        print(" - wavelength: {:2.2f} um".format(self.wavelength))
        print(" - date:       {}".format(self.date))
        if self.info != "":
            print(" - info:       {}".format(self.info))
        return ""


    @check_none('x', 'y', 'z', 'Ex', 'Ey', 'Ez', raise_exception=bool_raise_exception)
    def __add__(self, other):
        """adds two Vector_field_XY. For example two light sources or two masks

        Args:
            other (Vector_field_XY): 2nd field to add
            kind (str): instruction how to add the fields:

        Returns:
            Vector_field_XY: `E3 = E1 + E2`
        """

        EM = Vector_field_XYZ(self.x, self.y, self.z, self.wavelength)

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
            final_filename = save_data_common(self, filename, add_name,
                                              description, verbose)
            return final_filename
        except:
            return False


    def load_data(self, filename: str, verbose: bool = False):
        """Load data from a file to a Vector_field_XY.
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
                raise Exception('no dictionary in load_data')

        if verbose is True:
            print(dict0.keys())

    @check_none('Ex', 'Ey', 'Ez', raise_exception=bool_raise_exception)
    def clear_field(self):
        """simple - removes the field: self.E=0 """

        self.Ex = np.zeros_like(self.Ex, dtype=complex)
        self.Ey = np.zeros_like(self.Ex, dtype=complex)
        self.Ez = np.zeros_like(self.Ex, dtype=complex)



    def size(self, verbose: bool = False):
        """returns the size of the instance in MB.

        Args:
            verbose (bool, optional): prints size in Mb. Defaults to False.

        Returns:
            float: size in MB
        """

        return get_instance_size_MB(self, verbose)


    def duplicate(self, clear: bool = False):
        """Duplicates the instance"""
        new_field = copy.deepcopy(self)
        if clear is True:
            new_field.clear_field()
        return new_field



    @check_none('Ex', 'Ey', 'Ez', raise_exception=bool_raise_exception)
    def get(self, kind: get_vector_options, mode: str = 'modulus', **kwargs):
        """Takes the vector field and divide in Scalar_field_X.

        Args:
            kind (str): 'fields', 'intensity', 'intensities', 'phases', 'stokes', 'params_ellipse'

        Returns:
            Vector_field_X: (Ex, Ey, Ez),
        """

        data = get_vector(self, kind, mode, **kwargs)
        return data


    @check_none('x', 'Ex', 'Ey', 'Ez', raise_exception=bool_raise_exception)
    def incident_field(self, E0: Vector_field_XY  | None = None, u0: Scalar_field_XY  | None = None, 
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
            E0 = Vector_field_XY(self.x, self.y, self.wavelength, self.n_background)
            E0.Ex = u0.u * j0.M[0]
            E0.Ey = u0.u * j0.M[1]

        if z0 in (None, '', []):
            self.Ex0 = E0.Ex
            self.Ey0 = E0.Ey

            self.Ex[:,:,0] = self.Ex[:,:,0] + E0.Ex
            self.Ey[:,:,0] = self.Ey[:,:,0] + E0.Ey
        else:
            self.Ex0 = None
            self.Ey0 = None
            iz, _, _ = nearest(self.z, z0)
            self.Ex[:,:,iz] = self.Ex[:,:,iz] + E0.Ex
            self.Ey[:,:,iz] = self.Ey[:,:,iz] + E0.Ey


    def refractive_index_from_scalarXYZ(self, u_xyz: Scalar_mask_XYZ):
        """
        refractive_index_from_scalarXZ. Gets the refractive index from a Scalar field and passes to a vector field.
        
        Obviously, the refractive index is isotropic.

        Args:
            self (Vector_field_XZ): Vector_field_XZ
            u_xz (Scalar_mask_XZ): Scalar_mask_XZ
        """
        self.n = u_xyz.n
        
        # edges = self.surface_detection( min_incr = 0.1, reduce_matrix = 'standard', has_draw = False)
               
        # self.borders = edges           
        # return edges


    def FP_WPM(self, has_edges: bool = True, pow_edge: int = 80, matrix: bool = False, verbose: bool = False):
        """
        WPM Method. 'schmidt methodTrue is very fast, only needs discrete number of refractive indexes'


        Args:
            has_edges (bool): If True absorbing edges are used.
            pow_edge (float): If has_edges, power of the supergaussian
            matrix (bool): if True returns a matrix else
            verbose (bool): If True prints information

        References:

            1. M. W. Fertig and K.-H. Brenner, “Vector wave propagation method,” J. Opt. Soc. Am. A, vol. 27, no. 4, p. 709, 2010.

            2. S. Schmidt et al., “Wave-optical modeling beyond the thin-element-approximation,” Opt. Express, vol. 24, no. 26, p. 30188, 2016.

        """

        k0 = 2 * np.pi / self.wavelength

        x = self.x
        y = self.y
        z = self.z

        dx = x[1] - x[0]
        dy = y[1] - y[0]
        dz = z[1] - z[0]

        self.Ex[:,:,0] = self.Ex0
        self.Ey[:,:,0] = self.Ey0

        self.Hx = np.zeros_like(self.Ex)
        self.Hy = np.zeros_like(self.Ex)
        self.Hz = np.zeros_like(self.Ex)

        kx = get_k(x, flavour="+")
        ky = get_k(y, flavour="+")

        KX, KY = np.meshgrid(kx, ky)
        K_perp2 = KX**2 + KY**2

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
        y_center = (self.y[-1]+self.y[0])/2

        filter_x = np.exp(-(np.abs(self.X[:, :, 0]-x_center) / width_edge)**pow_edge)
        filter_y = np.exp(-(np.abs(self.Y[:, :, 0]-y_center) / width_edge)**pow_edge)
        filter_function = filter_x*filter_y

        radius = np.sqrt((self.X[:, :, 0]-x_center)**2+(self.Y[:, :, 0]-y_center)**2)

        filter_function = np.exp(-(radius / width_edge)**pow_edge)


        t1 = time.time_ns()

        num_steps = len(self.z)
        for j in range(1, num_steps):

            if has_filter[j] == 0:
                filter_edge = 1
            else:
                filter_edge = filter_function

            E_step, H_step = FP_WPM_schmidt_kernel(
                self.Ex[:,:,j-1],
                self.Ey[:,:,j-1],
                self.n[:,:,j-1],
                self.n[:,:,j],
                k0,
                kx,
                ky,
                self.wavelength,
                dz,
            ) * filter_edge

            self.Ex[:,:,j] = self.Ex[:,:,j] + E_step[0] * filter_edge
            self.Ey[:,:,j] = self.Ey[:,:,j] + E_step[1] * filter_edge
            self.Ez[:,:,j] = E_step[2] * filter_edge

            self.Hx[:,:,j] = H_step[0] * filter_edge
            self.Hy[:,:,j] = H_step[1] * filter_edge
            self.Hz[:,:,j] = H_step[2] * filter_edge

        # at the initial point the Ez field is not computed.
        self.Ez[:,:,0] = self.Ez[:,:,1]
        
        self.Hx[:,:,0] = self.Hx[:,:,1]
        self.Hy[:,:,0] = self.Hy[:,:,1]
        self.Hz[:,:,0] = self.Hz[:,:,1]

        t2 = time.time_ns()
        if verbose is True:
            print(
                "Time = {:2.2f} s, time/loop = {:2.4} ms".format(
                    (t2 - t1) / 1e9, (t2 - t1) / len(self.z) / 1e6
                )
            )

        if matrix is True:
            return (self.Ex, self.Ey, self.Ez), (self.Hx, self.Hy, self.Hz)

        return self
        

    @check_none('Ex', 'Ey', 'Ez', raise_exception=bool_raise_exception)
    def intensity(self):
        """"Returns intensity.
        """
        intensity = np.abs(self.Ex)**2 + np.abs(self.Ey)**2 + np.abs(
            self.Ez)**2

        return intensity

    
    @check_none('x', 'y', 'z', 'Ex', 'Ey', 'Ez', raise_exception=bool_raise_exception)
    def normalize(self, kind='amplitude', new_field: bool = False):
        """Normalizes the field so that intensity.max()=1.

        Args:
            new_field (bool): If False the computation goes to self.u. If True a new instance is produced
            kind (str): 'amplitude', or 'intensity'

        Returns
            u (numpy.array): normalized optical field
        """
        return normalize_field(self, kind, new_field)


    @check_none('x', 'y', 'z', 'Ex', 'Ey', 'Ez', raise_exception=bool_raise_exception)
    def to_Vector_field_XY(self,
                           iz0: int | None = None,
                           z0: float | None = None):
        """pass results to Scalar_field_XY. Only one of the first two variables (iz0,z0) should be used

        Args:
            iz0 (int): position i of z data in array
            z0 (float): position z to extract
            class (bool): If True it returns a class
            matrix (bool): If True it returns a matrix
        """
        field_output = Vector_field_XY(x=self.x,
                                       y=self.y,
                                       wavelength=self.wavelength)
        if iz0 is None:
            iz, _, _ = nearest(self.z, z0)
        else:
            iz = iz0
        field_output.Ex = np.squeeze(self.Ex[:, :, iz])
        field_output.Ey = np.squeeze(self.Ey[:, :, iz])
        field_output.Ez = np.squeeze(self.Ez[:, :, iz])
        field_output.Hx = np.squeeze(self.Hx[:, :, iz])
        field_output.Hy = np.squeeze(self.Hy[:, :, iz])
        field_output.Hz = np.squeeze(self.Hz[:, :, iz])
        field_output.n = np.squeeze(self.n[:, :, iz])

        return field_output


    @check_none('x', 'y', 'z', 'Ex', 'Ey', 'Ez', raise_exception=bool_raise_exception)
    def to_Vector_field_XZ(self,
                           iy0: int | None = None,
                           y0: float | None = None):
        """pass results to Vector_field_XZ. Only one of the first two variables (iy0,y0) should be used

        Args:
            iy0 (int): position i of y data in array
            y0 (float): position y to extract
            class (bool): If True it returns a class
            matrix (bool): If True it returns a matrix

        """
        from .vector_fields_XZ import Vector_field_XZ
        field_output = Vector_field_XZ(x=self.x,
                                       z=self.z,
                                       wavelength=self.wavelength)
        if iy0 is None:
            iy, _, _ = nearest(self.y, y0)
        else:
            iy = iy0
        field_output.Ex = np.squeeze(self.Ex[iy, :, :]).transpose()
        field_output.Ey = np.squeeze(self.Ey[iy, :, :]).transpose()
        field_output.Ez = np.squeeze(self.Ez[iy, :, :]).transpose()
        try:
            field_output.Hx = np.squeeze(self.Hx[iy, :, :]).transpose()
            field_output.Hy = np.squeeze(self.Hy[iy, :, :]).transpose()
            field_output.Hz = np.squeeze(self.Hz[iy, :, :]).transpose()
        except:
            pass
        
        try:
            field_output.n = np.squeeze(self.n[iy, :, :]).transpose()
        except:
            pass
        
        
        return field_output


    @check_none('x', 'y', 'z', 'Ex', 'Ey', 'Ez', raise_exception=bool_raise_exception)
    def to_Vector_field_YZ(self,
                           ix0: int | None = None,
                           x0: float | None = None):
        """pass results to Vector_field_XZ. Only one of the first two variables (iy0,y0) should be used

        Args:
            ix0 (int): position i of x data in array
            x0 (float): position x to extract
            class (bool): If True it returns a class
            matrix (bool): If True it returns a matrix

        """
        from .vector_fields_XZ import Vector_field_XZ
        field_output = Vector_field_XZ(x=self.y,
                                       z=self.z,
                                       wavelength=self.wavelength)
        if ix0 is None:
            ix, _, _ = nearest(self.x, x0)
        else:
            ix = ix0
        field_output.Ex = np.squeeze(self.Ex[:, ix, :]).transpose()
        field_output.Ey = np.squeeze(self.Ey[:, ix, :]).transpose()
        field_output.Ez = np.squeeze(self.Ez[:, ix, :]).transpose()
        try:
            field_output.Hx = np.squeeze(self.Hx[:, ix, :]).transpose()
            field_output.Hy = np.squeeze(self.Hy[:, ix, :]).transpose()
            field_output.Hz = np.squeeze(self.Hz[:, ix, :]).transpose()
        except:
            pass
        
        try:
            field_output.n = np.squeeze(self.n[:, ix, :]).transpose()
        except:
            pass

        return field_output

    @check_none('x', 'y', 'z', 'Ex', 'Ey', 'Ez', raise_exception=bool_raise_exception)
    def to_Vector_field_Z(self, kind: str = 'amplitude', x0: int | None = None,
                          y0: int | None = None, has_draw: bool = True,
                          z_scale: str = 'um'):
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
        ix, _, _ = nearest(self.x, x0)
        iy, _, _ = nearest(self.y, y0)

        Ex = np.squeeze(self.Ex[iy, ix, :])
        Ey = np.squeeze(self.Ey[iy, ix, :])
        Ez = np.squeeze(self.Ez[iy, ix, :])

        if kind == 'amplitude':
            field_x = np.abs(Ex)
            field_y = np.abs(Ey)
            field_z = np.abs(Ez)
        elif kind == 'intensity':
            field_x = np.abs(Ex)**2
            field_y = np.abs(Ey)**2
            field_z = np.abs(Ez)**2
        elif kind == 'phase':
            field_x = np.angle(Ex)
            field_y = np.angle(Ey)
            field_z = np.angle(Ez)

        if has_draw is True:
            if z_scale == 'mm':
                plt.plot(self.z / mm, field_x, 'k', lw=2)
                plt.xlabel(r'$z\,(mm)$')
                plt.xlim(left=self.z[0] / mm, right=self.z[-1] / mm)

            elif z_scale == 'um':
                plt.plot(self.z, field_x, 'k', lw=2)
                plt.xlabel(r'$z\,(\mu m)$')
                plt.xlim(left=self.z[0], right=self.z[-1])

            plt.ylabel(kind)

        return (field_x, field_y, field_z)


    @check_none('x', 'y', 'z', 'Ex', 'Ey', 'Ez', raise_exception=bool_raise_exception)
    def draw_XY(self,
                z0: float,
                kind: Draw_Vector_XY_Options = 'intensity',
                logarithm: float = 0,
                normalize: str = 'maximum',
                title: str = '',
                filename: str = '',
                cut_value: float | None = None,
                has_colorbar: bool = 'False',
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

        ufield = self.to_Vector_field_XY(z0=z0)
        ufield.draw(kind=kind,
                    logarithm=logarithm,
                    normalize=normalize,
                    title=title,
                    filename=filename,
                    cut_value=cut_value,
                    has_colorbar=has_colorbar,
                    reduce_matrix=reduce_matrix)


    @check_none('x', 'y', 'z', 'Ex', 'Ey', 'Ez', raise_exception=bool_raise_exception)
    def draw_XZ(self,
                kind: Draw_Vector_XZ_Options = 'intensity',
                y0: float = 0*mm,
                logarithm: float = 0,
                normalize: bool = False,
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
        ufield = self.to_Vector_field_XZ(y0=y0)
        h1 = ufield.draw(kind, logarithm, normalize, draw_borders, filename,
                         **kwargs)

        return h1


    @check_none('x', 'y', 'z', 'Ex', 'Ey', 'Ez', raise_exception=bool_raise_exception)
    def draw_YZ(self,
                kind: Draw_Vector_XZ_Options = 'intensity',
                x0: float = 0*mm,
                logarithm: float = 0,
                normalize: bool = False,
                draw_borders: bool = False,
                filename: str = '',
                **kwargs):
        """Longitudinal profile XZ at a given x0 value.

        Args:
            x0 (float): value of x0 for interpolation
            logarithm (float): If >0, intensity is scaled in logarithm
            normalize (str):  False, 'maximum', 'intensity', 'area'
            draw_borders (bool): check
            filename (str): filename to save
        """

        plt.figure()
        ufield = self.to_Vector_field_YZ(x0=x0)
        h1 = ufield.draw(kind, logarithm, normalize, draw_borders, filename,
                         **kwargs)

        return h1




def FP_WPM_schmidt_kernel(Ex, Ey, n1, n2, k0, kx, ky, wavelength, dz):
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
            E, H = FP_PWD_kernel_simple(Ex, Ey, n_r, n_s, k0, kx, ky, wavelength, dz)

            Ex_final = Ex_final + Imz * E[0]
            Ey_final = Ey_final + Imz * E[1]
            Ez_final = Ez_final + Imz * E[2]
            Hx_final = Hx_final + Imz * H[0]
            Hy_final = Hy_final + Imz * H[1]
            Hz_final = Hz_final + Imz * H[2]
    return (Ex_final, Ey_final, Ez_final), (Hx_final, Hy_final, Hz_final)


def FP_PWD_kernel_simple(Ex, Ey, n1, n2, k0, kx, ky, wavelength, dz):
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
    Exk = fftshift(fft2(Ex))
    Eyk = fftshift(fft2(Ey))

    kr = n1 * k0 # first layer
    ks = n2 * k0 # second layer
            
    KX, KY = np.meshgrid(kx, ky)
    Kperp2 = KX**2 + KY**2
    Kperp = np.sqrt(Kperp2)

    kz_r = np.sqrt(kr**2 - Kperp2) # first layer
    kz_s = np.sqrt(ks**2 - Kperp2) # second layer

    P = np.exp(1j * kz_s * dz)
    Gamma = kz_r*kz_s + kz_s * Kperp2 / kz_r
    

    # Fresnel coefficients
    t_TM, t_TE, _, _ = fresnel_equations_kx(KX, wavelength, n1, n2, [1, 1, 0, 0], has_draw=False)

    T00 = P * (t_TM*KX**2*Gamma + t_TE*KY**2*kr*ks) / (Kperp2*kr*ks) 
    T01 = P * (t_TM*KX*KY*Gamma - t_TE*KX*KY*kr*ks) / (Kperp2*kr*ks) 
    T10 = P * (t_TM*KX*KY*Gamma - t_TE*KX*KY*kr*ks) / (Kperp2*kr*ks) 
    T11 = P * (t_TM*KY**2*Gamma + t_TE*KX**2*kr*ks) / (Kperp2*kr*ks) 
        
    nan_indices = np.where(np.isnan(T00)) 

    if nan_indices is not None: 
        T00_b = P * (t_TM*KX**2*Gamma + t_TE*KY**2*kr*ks) / (Kperp2*kr*ks+1e-10) 
        T01_b = P * (t_TM*KX*KY*Gamma - t_TE*KX*KY*kr*ks) / (Kperp2*kr*ks+1e-10) 
        T10_b = P * (t_TM*KX*KY*Gamma - t_TE*KX*KY*kr*ks) / (Kperp2*kr*ks+1e-10) 
        T11_b = P * (t_TM*KY**2*Gamma + t_TE*KX**2*kr*ks) / (Kperp2*kr*ks+1e-10) 
    
        T00[nan_indices]=T00_b[nan_indices]
        T01[nan_indices]=T01_b[nan_indices]
        T10[nan_indices]=T10_b[nan_indices]
        T11[nan_indices]=T11_b[nan_indices] 
    

    ex0 = T00 * Exk + T01 * Eyk
    ey0 = T10 * Exk + T11 * Eyk 
    ez0 = - (KX*ex0+KY*ey0) / (kz_s)
    
    # thesis Fertig 2011 (3.40) pág 66 I do not feel confident yet
    TM00 = -KX*KY*Gamma 
    TM01 = -(KY*KY*Gamma + kz_s**2)
    TM10 = +(KX*KX*Gamma + kz_s**2)
    TM11 = +KX*KY*Gamma
    TM20 = -KY*kz_s
    TM21 = +KX*kz_s
        
    Z0 = 376.82  # ohms (impedance of free space)
    H_factor = n2 / (ks * kz_s * Z0)
    
    hx0 = (TM00*ex0+TM01*ey0) * H_factor
    hy0 = (TM10*ex0+TM11*ey0) * H_factor
    hz0 = (TM20*ex0+TM21*ey0) * H_factor
        

    Ex_final = ifft2(ifftshift(ex0))
    Ey_final = ifft2(ifftshift(ey0))
    Ez_final = ifft2(ifftshift(ez0))

    Hx_final = ifft2(ifftshift(hx0))
    Hy_final = ifft2(ifftshift(hy0))
    Hz_final = ifft2(ifftshift(hz0))

    return (Ex_final, Ey_final, Ez_final), (Hx_final, Hy_final, Hz_final)



def _compute1Elipse__(x0: float, y0: float, A: float, B: float, theta: float,
                      amplification: float = 1):
    """computes polarization ellipse for drawing.

    Args:
        x0 (float): position x of ellipse
        y0 (float): position y of ellipse
        A (float): axis 1 of ellipse
        B (float): axis 2 of ellipse
        theta (float): angle of ellipse
        amplification (float): increase of size of ellipse
    """
    # esto es para verlo más grande
    A = A * amplification
    B = B * amplification

    fi = np.linspace(0, 2 * np.pi, 64)
    cf = np.cos(fi - theta)
    sf = np.sin(fi - theta)

    r = 1 / np.sqrt(np.abs(cf / (A + eps)**2 + sf**2 / (B + eps)**2))

    x = r * np.cos(fi) + x0
    y = r * np.sin(fi) + y0

    return x, y
