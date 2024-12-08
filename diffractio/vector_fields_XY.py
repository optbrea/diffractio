# !/usr/bin/env python3


# ----------------------------------------------------------------------
# Name:        vector_fields_XY.py
# Purpose:     Class for handling vector fields in the XY plane
#
# Author:      Luis Miguel Sanchez Brea
#
# Created:     2024
# Licence:     GPLv3
# ----------------------------------------------------------------------


# flake8: noqa

"""
This module generates Vector_field_XY class. It is required also for generating masks and fields.
The main atributes are:
    * self.Ex - x component of electric field
    * self.Ey - y component of electric field
    * self.Ez - z component of electric field
    * self.wavelength - wavelength of the incident field. The field is monocromatic
    * self.x - x positions of the field
    * self.y - y positions of the field
    * self.X (numpy.array): equal size to x * y. complex field
    * self.Y (numpy.array): equal size to x * y. complex field
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


*Propagation*
    * RS - Rayleigh Sommerfeld

*Drawing functions*
    * draw: intensity, intensities, phases, fields, stokes, param_ellipse, ellipses

"""

from matplotlib import rcParams
import copy

from numpy import gradient
from scipy.fftpack import fft2, fftshift, ifft2
from scipy.interpolate import RectBivariateSpline
from py_pol.jones_matrix import Jones_matrix
from py_pol.jones_vector import Jones_vector

from .__init__ import degrees, eps, mm, np, plt, um
from .config import bool_raise_exception, CONF_DRAWING, Draw_Vector_XY_Options, get_vector_options
from .utils_typing import NDArrayFloat
from .utils_common import load_data_common, save_data_common, get_date, check_none, get_vector, check_none
from .utils_common import get_instance_size_MB

from .utils_drawing import normalize_draw, reduce_matrix_size, draw_edges
from .utils_math import nearest
from .scalar_fields_XY import Scalar_field_XY
from .scalar_masks_XY import Scalar_mask_XY


percentage_intensity = CONF_DRAWING['percentage_intensity']
percentage_Ez = CONF_DRAWING['percentage_Ez']


class Vector_field_XY():
    """Class for vectorial fields.

    Args:
        x (numpy.array): linear array with equidistant positions. The number of data is preferibly 2**n.
        y (numpy.array): linear array with equidistant positions. The number of data is preferibly 2**n.
        wavelength (float): wavelength of the incident field
        info (str): String with info about the simulation

    Attributes:
        self.x (numpy.array): linear array with equidistant positions. The number of data is preferibly 2**n.
        self.y (numpy.array): linear array with equidistant positions. The number of data is preferibly 2**n.
        self.wavelength (float): wavelength of the incident field.
        self.Ex (numpy.array): Electric_x field
        self.Ey (numpy.array): Electric_y field
    """

    def __init__(self, x: NDArrayFloat | None = None, y: NDArrayFloat | None = None,
                 wavelength: float | None = None, n_background: float = 1.0, info: str = ""):
        self.x = x
        self.y = y
        self.wavelength = wavelength
        self.X, self.Y = np.meshgrid(x, y)
        self.n_background = n_background


        self.Ex = np.zeros_like(self.X, dtype=complex)
        self.Ey = np.zeros_like(self.X, dtype=complex)
        self.Ez = np.zeros_like(self.X, dtype=complex)

        self.n = None
        self.borders = None

        self.reduce_matrix = 'standard'  # 'None, 'standard', (5,5)
        self.type = 'Vector_field_XY'
        self.info = info
        self.date = get_date()
        self.CONF_DRAWING = CONF_DRAWING


    def __str__(self):
        """Represents data from class."""

        intensity = self.intensity()
        Imin = intensity.min()
        Imax = intensity.max()

        print("{}\n - x:  {},   y:  {},   Ex:  {}".format(
            self.type, self.x.shape, self.y.shape, self.Ex.shape))
        print(
            " - xmin:       {:2.2f} um,  xmax:      {:2.2f} um,  Dx:   {:2.2f} um"
            .format(self.x[0], self.x[-1], self.x[1] - self.x[0]))
        print(
            " - ymin:       {:2.2f} um,  ymay:      {:2.2f} um,  Dy:   {:2.2f} um"
            .format(self.y[0], self.y[-1], self.y[1] - self.y[0]))
        print(" - Imin:       {:2.2f},     Imax:      {:2.2f}".format(
            Imin, Imax))

        print(" - wavelength: {:2.2f} um".format(self.wavelength))
        print(" - date:       {}".format(self.date))
        print(" - info:       {}".format(self.info))

        return ""

    @check_none('x','y','Ex','Ey',raise_exception=bool_raise_exception)
    def __add__(self, other, kind: str = 'standard'):
        """adds two Vector_field_XY. For example two light sources or two masks

        Args:
            other (Vector_field_XY): 2nd field to add
            kind (str): instruction how to add the fields:

        Returns:
            Vector_field_XY: `E3 = E1 + E2`
        """

        EM = Vector_field_XY(self.x, self.y, self.wavelength)

        EM.Ex = self.Ex + other.Ex
        EM.Ey = self.Ey + other.Ey

        return EM

    @check_none('x','y','X','Y',raise_exception=bool_raise_exception)
    def __rotate__(self, angle, position=None):
        """Rotation of X,Y with respect to position

        Args:
            angle (float): angle to rotate, in radians
            position (float, float): position of center of rotation
        """

        if position is None:
            x0 = (self.x[-1] + self.x[0])/2
            y0 = (self.y[-1] + self.y[0])/2
        else:
            x0, y0 = position

        Xrot = (self.X - x0) * np.cos(angle) + (self.Y - y0) * np.sin(angle)
        Yrot = -(self.X - x0) * np.sin(angle) + (self.Y - y0) * np.cos(angle)
        return Xrot, Yrot


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


    @check_none('x','y','Ex','Ey',raise_exception=bool_raise_exception)
    def clear(self):
        """simple - removes the field:"""

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


    def to_py_pol(self):
        """Pass diffractio vector field or mask to py_pol package for software analysis.
        Returns:
            py_pol.jones_matrix or py_pol.jones_vector

        """
        
        if self.type == 'Vector_mask_XY':
            m0 = Jones_matrix(name="from Diffractio")
            m0.from_components((self.M00, self.M01, self.M10, self.M11))
            m0.shape = self.M00.shape
            return m0

        elif self.type in ('Vector_field_XY', 'Vector_source_XY'):
            j0 = Jones_vector(name="from Diffractio")
            j0.from_components(Ex=self.Ex, Ey=self.Ey)
            return j0

    def duplicate(self, clear: bool = False):
        """Duplicates the instance,

        Args:
            clear (bool, optional): Clear the data. Defaults to False.

        Returns:
            Vector_field_XY: New instance
        """

        new_field = copy.deepcopy(self)
        if clear is True:
            new_field.clear_field()
        return new_field


    @check_none('Ex','Ey','Ez',raise_exception=bool_raise_exception)
    def get(self, kind: get_vector_options, mode: str = 'modulus', **kwargs):
        """Takes the vector field and divide in Scalar_field_X.

        Args:
            kind (str): 'fields', 'intensity', 'intensities', 'phases', 'stokes', 'params_ellipse'

        Returns:
            Vector_field_X: (Ex, Ey, Ez),
        """

        data = get_vector(self, kind, mode, **kwargs)
        return data


    @check_none('x','y','Ex','Ey','Ez',raise_exception=bool_raise_exception)
    def pupil(self, r0: tuple[float, float] | None = None, radius: tuple[float, float] | None = None,
              angle: float = 0*degrees):
        """place a pupil in the field. If r0 or radius are None, they are computed using the x,y parameters.

        Args:
            r0 (float, float): center of circle/ellipse
            radius (float, float) or (float): radius of circle/ellipse
            angle (float): angle of rotation in radians

        Example:

            pupil(r0=(0*um, 0*um), radius=(250*um, 125*um), angle=0*degrees)
        """

        if r0 in (0, None, '', []):
            x0 = (self.x[-1] + self.x[0])/2
            y0 = (self.y[-1] + self.y[0])/2
            r0 = (x0, y0)

        if radius is None:
            radiusx = (self.x[-1] - self.x[0])/2
            radiusy = (self.y[-1] - self.y[0])/2
            radius = (radiusx, radiusy)

        if isinstance(radius, (float, int, complex)):
            radiusx, radiusy = (radius, radius)
        else:
            radiusx, radiusy = radius
        radius = (radiusx, radiusy)

        x0, y0 = r0

        Xrot, Yrot = self.__rotate__(angle, (x0, y0))

        pupil0 = np.zeros(np.shape(self.X))
        ipasa = (Xrot)**2 / (radiusx + 1e-15)**2 + (Yrot)**2 / (radiusy**2 +
                                                                1e-15) < 1
        pupil0[ipasa] = 1
        self.Ex = self.Ex * pupil0
        self.Ey = self.Ey * pupil0
        self.Ez = self.Ez * pupil0


    @check_none('Ex','Ey','Ez',raise_exception=bool_raise_exception)
    def apply_mask(self, u, new_field: bool = False):
        """Multiply field by binary scalar mask: self.Ex = self.Ex * u.u

        Args:
           u (Scalar_mask_XY): mask

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


    def refractive_index_from_scalarXY(self, u_xy: Scalar_mask_XY):
        """
        refractive_index_from_scalarXY. Gets the refractive index from a Scalar field and passes to a vector field.
        
        Obviously, the refractive index is isotropic.

        Args:
            self (Vector_field_XY): Vector_field_XY
            u_xy (Scalar_mask_XY): Scalar_mask_XY
        """
        self.n = u_xy.n
        
        edges = self.surface_detection( min_incr = 0.1, reduce_matrix = 'standard', has_draw = False)

        self.borders = edges           
        return edges
    
    
    @check_none('Ex','Ey','Ez',raise_exception=bool_raise_exception)
    def intensity(self):
        """"Returns intensity.
        """
        intensity = np.abs(self.Ex)**2 + np.abs(self.Ey)**2 + np.abs(
            self.Ez)**2

        return intensity


    # @check_none('x','y',raise_exception=bool_raise_exception)
    # def RS(self,
    #        z=10*mm,
    #        n: float = 1.,
    #        new_field: bool = True,
    #        amplification=(1, 1),
    #        verbose: bool = False):
    #     """Fast-Fourier-Transform  method for numerical integration of diffraction Rayleigh-Sommerfeld formula. `Thin Element Approximation` is considered for determining the field just after the mask: :math:`\mathbf{E}_{0}(\zeta,\eta)=t(\zeta,\eta)\mathbf{E}_{inc}(\zeta,\eta)` Is we have a field of size N*M, the result of propagation is also a field N*M. Nevertheless, there is a parameter `amplification` which allows us to determine the field in greater observation planes (jN)x(jM).

    #     Args:
    #         z (float): distance to observation plane.
    #             if z<0 inverse propagation is executed
    #         n (float): refractive index
    #         new_field (bool): if False the computation goes to self.u
    #                         if True a new instance is produced
    #         amplification (int, int): number of frames in x and y direction
    #         verbose (bool): if True it writes to shell. Not implemented yet

    #     Returns:
    #         if New_field is True: Scalar_field_X, else None

    #     Note:
    #         One adventage of this approach is that it returns a quality parameter: if self.quality>1, propagation is right.

    #     References:
    #         From Applied Optics vol 45 num 6 pp. 1102-1110 (2006)
    #     """

    #     e0x, e0y, _ = self.get()

    #     # estas son las components justo en la posicion pedida
    #     Ex = e0x.RS(z=z,
    #                 n=n,
    #                 new_field=True,
    #                 kind='z',
    #                 amplification=amplification)
    #     Ey = e0y.RS(z=z,
    #                 n=n,
    #                 new_field=True,
    #                 kind='z',
    #                 amplification=amplification,
    #                 verbose=verbose)

    #     if new_field is True:
    #         EM = Vector_field_XY(Ex.x, Ex.y, self.wavelength)
    #         EM.Ex = Ex.u
    #         EM.Ey = Ey.u
    #         EM.Ez = np.zeros_like(EM.X)
    #         EM.x = Ex.x
    #         EM.y = Ex.y
    #         return EM

    #     else:
    #         self.Ex = Ex.u
    #         self.Ey = Ey.u
    #         self.Ez = np.zeros_like(EM.X)
    #         self.x = Ex.x
    #         self.y = Ex.y


    @check_none('x','y','Ex','Ey','Ez',raise_exception=False)
    def VFFT(self,
             radius: float,
             focal: float,
             n: float = 1.,
             new_field: bool = False,
             shift: bool = True,
             remove0: bool = True,
             matrix: bool = False,
             has_draw: bool = False):
        """Vector Fast Fourier Transform (FFT) of the field.

        Ei = (Eix, Eiy, Eiz) is the local electric field vector.

        Args:
            radius (float): radius of lens
            focal (float): focal
            n (float): refractive index
            matrix (bool):  if True only matrix is returned. if False, returns Scalar_field_X.
            new_field (bool): if True returns Vector_field_XY, else it puts in self.
            shift (bool):  if True, fftshift is performed.
            remove0 (bool): if True, central point is removed.

        Returns:
            (np.array or vector_fields_XY or None): FFT of the input field.

        Reference:
            Jahn, Kornél, and Nándor Bokor. 2010. “Intensity Control of the Focal Spot by Vectorial Beam Shaping.” Optics Communications 283 (24): 4859–65. https://doi.org/10.1016/j.optcom.2010.07.030.

        TODO: Some inconsistency in the radius of the circle lower than the size of the field.
        """
        from .vector_sources_XY import Vector_source_XY

        num_x, num_y = self.X.shape

        # numerical aperture
        sin_theta_max = radius / np.sqrt(radius**2 + focal**2)
        # NA = n * sin_theta_max

        r = np.sqrt(self.X**2 + self.Y**2)
        phi = np.arctan2(self.Y, self.X)
        theta = r / focal

        u = self.X / radius
        v = self.Y / radius

        circle_mask = Scalar_mask_XY(self.x, self.y, self.wavelength)
        circle_mask.circle(r0=(0*um, 0*um), radius=radius)

        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)

        apodization_factor = np.sqrt(np.abs(np.cos(theta)))

        G = 1 / np.sqrt(np.abs(1 - sin_theta_max**2 * (u**2 + v**2)))
        G = G * circle_mask.u
        G = np.real(G)

        M00 = cos_phi**2 * cos_theta + sin_phi**2
        M01 = sin_phi * cos_phi * cos_theta - sin_phi * cos_phi
        M02 = -sin_theta * cos_phi

        M10 = sin_phi * cos_phi * cos_theta - sin_phi * cos_phi
        M11 = sin_phi**2 * cos_theta + cos_phi**2
        M12 = -sin_theta * sin_phi

        M20 = sin_theta * cos_phi
        M21 = sin_theta * sin_phi
        M22 = cos_theta

        Eix = self.Ex
        Eiy = self.Ey
        Eiz = self.Ez

        E0x = M00 * Eix + M01 * Eiy + M02 * Eiz
        E0y = M10 * Eix + M11 * Eiy + M12 * Eiz
        E0z = M20 * Eix + M21 * Eiy + M22 * Eiz

        factor = -(1j * sin_theta_max**2 / (focal * self.wavelength))
        Ep_x = fft2(apodization_factor * G * E0x)
        Ep_y = fft2(apodization_factor * G * E0y)
        Ep_z = fft2(apodization_factor * G * E0z)

        if remove0:
            Ep_x[0, 0] = 0
            Ep_y[0, 0] = 0
            # Ep_z[0,0] = 0

        if shift:
            Ep_x = fftshift(Ep_x)
            Ep_y = fftshift(Ep_y)
            Ep_z = fftshift(Ep_z)

        Esx = factor * Ep_x
        Esy = factor * Ep_y
        Esz = factor * Ep_z

        if matrix is True:
            return np.stack((Esx, Esy, Esz), axis=2)

        num_x = self.x.size
        delta_x = self.x[1] - self.x[0]
        freq_nyquist_x = 1 / (2 * delta_x)

        kx = np.linspace(-freq_nyquist_x, freq_nyquist_x, num_x) * focal
        num_y = self.y.size
        delta_y = self.y[1] - self.y[0]
        freq_nyquist_y = 1 / (2 * delta_y)
        ky = np.linspace(-freq_nyquist_y, freq_nyquist_y, num_y) * focal

        if new_field is True:
            field_output = Vector_source_XY(self.x, self.y, self.wavelength)
            field_output.x = kx
            field_output.y = ky

            field_output.X, field_output.Y = np.meshgrid(field_output.x, field_output.y)
            field_output.Ex = Esx
            field_output.Ey = Esy
            field_output.Ez = Esz

            return field_output

        else:
            self.Ex = Esx
            self.Ey = Esy
            self.Ez = Esz
            self.x = kx
            self.y = ky
            self.X, self.Y = np.meshgrid(self.x, self.y)



    @check_none('x','y','Ex','Ey','Ez',raise_exception=False)
    def IVFFT(self,
              radius: float,
              focal: float,
              n: float = 1.,
              new_field: bool = False,
              matrix: bool = False,
              has_draw: bool = False):
        """Inverse Vector Fast Fourier Transform (FFT) of the field.

        Ei = (Eix, Eiy, Eiz) is the local electric field vector.

        Args:
            radius (float): radius of lens
            focal (float): focal
            n (float): refractive index
            matrix (bool):  if True only matrix is returned. if False, returns Scalar_field_X.
            new_field (bool): if True returns Vector_field_XY, else it puts in self.
            has_draw (bool): if True draw the field.

        Returns:
            (np.array or vector_fields_XY or None): FFT of the input field.

        Reference:
            Jahn, Kornél, and Nándor Bokor. 2010. “Intensity Control of the Focal Spot by Vectorial Beam Shaping.” Optics Communications 283 (24): 4859–65. https://doi.org/10.1016/j.optcom.2010.07.030.


        TODO: Radius of the circle lower than the size of the field.
        """

        from diffractio.vector_sources_XY import Vector_source_XY

        # numerical aperture
        sin_theta_max = radius / np.sqrt(radius**2 + focal**2)
        # NA = n * sin_theta_max

        # dx = self.x[1] - self.x[0]
        # dy = self.y[1] - self.y[0]
        num_x, num_y = self.X.shape

        r = np.sqrt(self.X**2 + self.Y**2)
        phi = np.arctan2(self.Y, self.X)
        theta = r / focal

        u = self.X / radius
        v = self.Y / radius

        # X_obs = sin_theta_max * self.X / self.wavelength
        # Y_obs = sin_theta_max * self.Y / self.wavelength

        circle_mask = Scalar_mask_XY(self.x, self.y, self.wavelength)
        circle_mask.circle(r0=(0*um, 0*um), radius=radius)

        self.pupil(r0=(0., 0.), radius=radius)

        cos_theta = np.cos(theta)
        sin_theta = -np.sin(theta)
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)

        #  apodization_factor = np.sqrt(np.abs(np.cos(theta)))

        G = np.sqrt(np.abs(1 - sin_theta_max**2 * (u**2 + v**2)))
        G = G * circle_mask.u
        G = np.real(G)

        M00 = cos_phi**2 * cos_theta + sin_phi**2
        M01 = sin_phi * cos_phi * cos_theta - sin_phi * cos_phi
        M02 = -sin_theta * cos_phi

        M10 = sin_phi * cos_phi * cos_theta - sin_phi * cos_phi
        M11 = sin_phi**2 * cos_theta + cos_phi**2
        M12 = -sin_theta * sin_phi

        M20 = sin_theta * cos_phi
        M21 = sin_theta * sin_phi
        M22 = cos_theta

        Eix = self.Ex
        Eiy = self.Ey
        Eiz = self.Ez

        factor = -(1j * sin_theta_max**2 / (focal * self.wavelength))**(-1)

        Esx = factor * ifft2(Eix * G)
        Esy = factor * ifft2(Eiy * G)
        Esz = factor * ifft2(Eiz * G)

        Esx = factor * ifft2(Eix * G)
        Esy = factor * ifft2(Eiy * G)
        Esz = factor * ifft2(Eiz * G)

        E0x = M00 * Esx + M01 * Esy + M02 * Esz
        E0y = M10 * Esx + M11 * Esy + M12 * Esz
        E0z = M20 * Esx + M21 * Esy + M22 * Esz

        Esx = E0x
        Esy = E0y
        Esz = E0z

        if matrix is True:
            return np.stack((Esx, Esy, Esz), axis=2)

        num_x = self.x.size
        delta_x = self.x[1] - self.x[0]
        freq_nyquist_x = 1 / (2 * delta_x)

        kx = np.linspace(-freq_nyquist_x, freq_nyquist_x, num_x) * focal
        num_y = self.y.size
        delta_y = self.y[1] - self.y[0]
        freq_nyquist_y = 1 / (2 * delta_y)
        ky = np.linspace(-freq_nyquist_y, freq_nyquist_y, num_y) * focal

        if new_field is True:
            field_output = Vector_source_XY(self.x, self.y, self.wavelength)
            field_output.x = kx
            field_output.y = ky

            field_output.X, field_output.Y = np.meshgrid(field_output.x, field_output.y)
            field_output.Ex = Esx
            field_output.Ey = Esy
            field_output.Ez = Esz

            return field_output

        else:
            self.Ex = Esx
            self.Ey = Esy
            self.Ez = Esz
            self.x = kx
            self.y = ky
            self.X, self.Y = np.meshgrid(self.x, self.y)



    @check_none('x','y',raise_exception=False)
    def VRS(self, z: float, n: float = 1., new_field: bool = True, verbose: bool = False,
            amplification: tuple[int, int] = (1, 1)):
        """Fast-Fourier-Transform  method for numerical integration of diffraction Vector Rayleigh-Sommerfeld formula.

        Args:
            z (float): distance to observation plane.
                if z<0 inverse propagation is executed
            n (float): refractive index
            new_field (bool): if False the computation goes to self.u. If True a new instance is produced

            verbose (bool): if True it writes to shell. Not implemented yet

        Returns:
            if New_field is True: Scalar_field_X, else None

        References:

            H. Ye, C.-W. Qiu, K. Huang, J. Teng, B. Luk’yanchuk, y S. P. Yeo, «Creation of a longitudinally polarized subwavelength hotspot with an ultra-thin planar lens: vectorial Rayleigh–Sommerfeld method», Laser Phys. Lett., vol. 10, n.º 6, p. 065004, jun. 2013.
            DOI: 10.1088/1612-2011/10/6/065004
            http://stacks.iop.org/1612-202X/10/i=6/a=065004?key=crossref.890761f053b56d7a9eeb8fc6da4d9b4e
        """

        e0xm, e0ym, _ = self.get('E')


        e0x = Scalar_field_XY(self.x, self.y, self.wavelength)
        e0x.u = e0xm
        e0y = Scalar_field_XY(self.x, self.y, self.wavelength)
        e0y.u = e0ym
        e0z = Scalar_field_XY(self.x, self.y, self.wavelength)
        e0z.u = np.zeros_like(e0x.u)

        # estas son las components justo en la posicion pedida
        Ex = e0x.RS(z=z,
                    n=n,
                    new_field=True,
                    kind='z',
                    verbose=verbose,
                    amplification=amplification)
        Ey = e0y.RS(z=z,
                    n=n,
                    new_field=True,
                    kind='z',
                    verbose=verbose,
                    amplification=amplification)

        r = np.sqrt(self.X**2 + self.Y**2 + z**2)
        e0z.u = e0x.u * self.X / r + e0y.u * self.Y / r
        Ez = e0z.RS(z=z,
                    n=n,
                    new_field=True,
                    kind='0',
                    verbose=verbose,
                    amplification=amplification)
        
        self.x = Ex.x
        self.y = Ey.y

        if new_field is True:
            EM = Vector_field_XY(self.x, self.y, self.wavelength)
            EM.Ex = Ex.u
            EM.Ey = Ey.u
            EM.Ez = Ez.u
            return EM

        else:
            self.Ex = Ex.u
            self.Ey = Ey.u
            self.Ez = Ez.u


    @check_none('x','y',raise_exception=bool_raise_exception)
    def VCZT(self, z, xout=None, yout=None, verbose: bool = False):
        """Vector Z Chirped Transform algorithm (VCZT)

        The code for this algoritm is based on "Hu, Yanlei, et al. "Efficient full-path optical calculation of scalar and 
        vector diffraction using the Bluestein method." Light: Science & Applications 9.1 (2020): 119."
        
        However, the convolution Kernel has been changed to Rayleigh-Sommerfeld.

        Args:
            z (float): diffraction distance
            xout (np.array): x array with positions of the output plane
            yout (np.array): y array with positions of the output plane
            verbose (bool): If True prints some information

        Returns:
            E_out (variable): Output field. It depends on the size of xout, yout, and z.

        References:
            - [Light: Science and Applications, 9(1), (2020)] 
        """
        if xout is None:
            xout = self.x

        if yout is None:
            yout = self.y

        k = 2 * np.pi / self.wavelength

        if isinstance(z, (float, int)):
            num_z = 1
            # print("z = 0 dim")
        else:
            num_z = len(z)
            # print("z = 1 dim")

        if isinstance(xout, (float, int)):
            num_x = 1
            # print("x = 0 dim")
            xstart = xout
            xend = xout
        else:
            num_x = len(xout)
            # print("x = 1 dim")

            xstart = xout[0]
            xend = xout[-1]

        if isinstance(yout, (float, int)):
            num_y = 1
            # print("y = 0 dim")
            ystart = yout
            yend = yout
        else:
            num_y = len(yout)
            # print("y = 1 dim")

            ystart = yout[0]
            yend = yout[-1]

        e0xm, e0ym, _ = self.get('E')
        e0x = Scalar_field_XY(self.x, self.y, self.wavelength)
        e0x.u = e0xm
        e0y = Scalar_field_XY(self.x, self.y, self.wavelength)
        e0y.u = e0ym
        e0z = Scalar_field_XY(self.x, self.y, self.wavelength)
        e0z.u = np.zeros_like(e0x.u)

        if num_z == 1:
            r = np.sqrt(self.X**2 + self.Y**2 + z**2)
            e0z_u = e0x.u * self.X / r + e0y.u * self.Y / r
            e0z_u = e0z_u * z / r
            e0z.u = e0z_u

            e0x = e0x.CZT(z, xout, yout)
            e0y = e0y.CZT(z, xout, yout)
            e0z = e0z.CZT(z, xout, yout)

            if num_x == 1 and num_y == 1:
                return e0x, e0y, e0z

            elif num_x > 1 and num_y == 1:
                from diffractio.vector_fields_X import Vector_field_X
                E_out = Vector_field_X(xout, self.wavelength)
                E_out.Ex = e0x.u
                E_out.Ey = e0y.u
                E_out.Ez = e0z.u
                return E_out
            elif num_x == 1 and num_y > 1:
                from diffractio.vector_fields_X import Vector_field_X
                E_out = Vector_field_X(yout, self.wavelength)
                E_out.Ex = e0x.u
                E_out.Ey = e0y.u
                E_out.Ez = e0z.u
                return E_out
            elif num_x > 1 and num_y > 1:
                from diffractio.vector_fields_XY import Vector_field_XY
                E_out = Vector_field_XY(xout, yout, self.wavelength)
                E_out.Ex = e0x.u
                E_out.Ey = e0y.u
                E_out.Ez = e0z.u
                return E_out

        elif num_z > 1:
            if verbose is True:
                print("1/3", end='\r')
            e0x_zs = e0x.CZT(z, xout, yout, verbose=verbose)
            if verbose is True:
                print("2/3", end='\r')
            e0y_zs = e0y.CZT(z, xout, yout, verbose=verbose)
            if verbose is True:
                print("3/3", end='\r')
            e0z_zs = e0x_zs.duplicate()

            u_zs = np.zeros_like(e0x_zs.u)

            for i, z_now in enumerate(z):
                if verbose:
                    print("{}/{}".format(i, num_z), end='\r')

                r = np.sqrt(self.X**2 + self.Y**2 + z_now**2)
                e0z_u = e0x.u * self.X / r + e0y.u * self.Y / r
                e0z_u = e0z_u * z_now / r
                e0z.u = e0z_u

                e0z_u = e0z.CZT(z_now, xout, yout)

                if num_x == 1 and num_y == 1:
                    e0z_zs.u[i] = e0z_u

                elif num_x > 1 and num_y == 1:
                    e0z_zs.u[i, :] = e0z_u.u
                elif num_x > 1 and num_y > 1:
                    e0z_zs.u[:, :, i] = e0z_u.u

            if num_x == 1 and num_y == 1:
                from diffractio.vector_fields_Z import Vector_field_Z
                E_out = Vector_field_Z(z, self.wavelength)
                E_out.Ex = e0x_zs.u
                E_out.Ey = e0y_zs.u
                E_out.Ez = e0z_zs.u
                return E_out

            elif num_x > 1 and num_y == 1:
                from diffractio.vector_fields_XZ import Vector_field_XZ
                E_out = Vector_field_XZ(xout, z, self.wavelength)
                E_out.Ex = e0x_zs.u
                E_out.Ey = e0y_zs.u
                E_out.Ez = e0z_zs.u
                return E_out

            elif num_x == 1 and num_y > 1:
                from diffractio.vector_fields_XZ import Vector_field_XZ
                E_out = Vector_field_XZ(yout, z, self.wavelength)
                E_out.Ex = e0x_zs.u
                E_out.Ey = e0y_zs.u
                E_out.Ez = e0z_zs.u
                return E_out

            elif num_x > 1 and num_y > 1:
                from diffractio.vector_fields_XYZ import Vector_field_XYZ
                E_out = Vector_field_XYZ(xout, yout, z, self.wavelength)
                E_out.Ex = e0x_zs.u
                E_out.Ey = e0y_zs.u
                E_out.Ez = e0z_zs.u
                return E_out


    @check_none('x','y','Ex','Ey','Ez',raise_exception=bool_raise_exception)
    def normalize(self, kind:str = 'amplitude'):
        """Normalizes the field, to the maximum intensity.
        
        Args:
            kind (str): 'amplitude' or 'intensity'.
        """

        if kind =='amplitude':
            maximum = np.sqrt(np.abs(self.Ex)**2 + np.abs(self.Ey)**2 +
                np.abs(self.Ez)**2).max()
        elif kind == 'intensity':
            maximum = (np.abs(self.Ex)**2 + np.abs(self.Ey)**2 +
                np.abs(self.Ez)**2).max()

        self.Ex = self.Ex / maximum
        self.Ey = self.Ey / maximum
        self.Ez = self.Ez / maximum


    @check_none('x','y','Ex','Ey','Ez',raise_exception=bool_raise_exception)
    def cut_resample(self,
                     x_limits: tuple[float, float] | None = None,
                     y_limits: tuple[float, float] | None = None,
                     num_points: int | None = None,
                     new_field: bool = False,
                     interp_kind: tuple[int, int] = (3, 1)):
        """Cuts the field to the range (x0,x1). (y0,y1). If one of this x0,x1 positions is out of the self.x range it do nothing. It is also valid for resampling the field, just write x0,x1 as the limits of self.x

        Args:
            x_limits (float,float): (x0,x1) starting and final points to cut. if '' - takes the current limit x[0] and x[-1]
            y_limits (float,float): (y0,y1) - starting and final points to cut. if '' - takes the current limit y[0] and y[-1]
            num_points (int): it resamples x, y and u. [],'',0,None -> it leave the points as it is
            new_field (bool): it returns a new Scalar_field_XY
            interp_kind (int): numbers between 1 and 5
        """
        if x_limits is None:
            x0 = self.x[0]
            x1 = self.x[-1]
        else:
            x0, x1 = x_limits

        if y_limits is None:
            y0 = self.y[0]
            y1 = self.y[-1]
        else:
            y0, y1 = y_limits

        if x0 < self.x[0]:
            x0 = self.x[0]
        if x1 > self.x[-1]:
            x1 = self.x[-1]

        if y0 < self.y[0]:
            y0 = self.y[0]
        if y1 > self.y[-1]:
            y1 = self.y[-1]

        i_x0, _, _ = nearest(self.x, x0)
        i_x1, _, _ = nearest(self.x, x1)
        i_y0, _, _ = nearest(self.y, y0)
        i_y1, _, _ = nearest(self.y, y1)

        kxu, kxn = interp_kind

        if num_points not in ([], '', 0, None):
            num_points_x, num_points_y = num_points
            x_new = np.linspace(x0, x1, num_points_x)
            y_new = np.linspace(y0, y1, num_points_y)
            X_new, Y_new = np.meshgrid(x_new, y_new)

            f_interp_abs_x = RectBivariateSpline(self.x,
                                                 self.y,
                                                 np.abs(self.Ex),
                                                 kx=kxu,
                                                 ky=kxu,
                                                 s=0)
            f_interp_phase_x = RectBivariateSpline(self.x,
                                                   self.y,
                                                   np.angle(self.Ex),
                                                   kx=kxu,
                                                   ky=kxu,
                                                   s=0)

            f_interp_abs_y = RectBivariateSpline(self.x,
                                                 self.y,
                                                 np.abs(self.Ey),
                                                 kx=kxu,
                                                 ky=kxu,
                                                 s=0)
            f_interp_phase_y = RectBivariateSpline(self.x,
                                                   self.y,
                                                   np.angle(self.Ey),
                                                   kx=kxu,
                                                   ky=kxu,
                                                   s=0)

            f_interp_abs_z = RectBivariateSpline(self.x,
                                                 self.y,
                                                 np.abs(self.Ez),
                                                 kx=kxu,
                                                 ky=kxu,
                                                 s=0)
            f_interp_phase_z = RectBivariateSpline(self.x,
                                                   self.y,
                                                   np.angle(self.Ez),
                                                   kx=kxu,
                                                   ky=kxu,
                                                   s=0)

            Ex_new_abs = f_interp_abs_x(x_new, y_new)
            Ex_new_phase = f_interp_phase_x(x_new, y_new)
            Ex_new = Ex_new_abs * np.exp(1j * Ex_new_phase)

            Ey_new_abs = f_interp_abs_y(x_new, y_new)
            Ey_new_phase = f_interp_phase_y(x_new, y_new)
            Ey_new = Ey_new_abs * np.exp(1j * Ey_new_phase)

            Ez_new_abs = f_interp_abs_z(x_new, y_new)
            Ez_new_phase = f_interp_phase_z(x_new, y_new)
            Ez_new = Ez_new_abs * np.exp(1j * Ez_new_phase)

        else:
            i_s = slice(i_x0, i_x1)
            j_s = slice(i_y0, i_y1)
            x_new = self.x[i_s]
            y_new = self.y[j_s]
            X_new, Y_new = np.meshgrid(x_new, y_new)
            Ex_new = self.Ex[i_s, j_s]
            Ey_new = self.Ey[i_s, j_s]
            Ez_new = self.Ez[i_s, j_s]

        if new_field is False:
            self.x = x_new
            self.y = y_new
            self.Ex = Ex_new
            self.Ey = Ey_new
            self.Ez = Ez_new
            self.X = X_new
            self.Y = Y_new
        else:
            field = Vector_field_XY(x=x_new,
                                    y=y_new,
                                    wavelength=self.wavelength)
            field.Ex = Ex_new
            field.Ey = Ey_new
            field.Ez = Ez_new
            return field

    @check_none('x','y','n')
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
        y_new = self.y
        x_new = self.x
        n_new = self.n

        diff1 = gradient(np.abs(n_new), axis=0)
        diff2 = gradient(np.abs(n_new), axis=1)

        # if np.abs(diff1 > min_incr) or np.abs(diff2 > min_incr):
        t = np.abs(diff1) + np.abs(diff2)

        ix, iy = (t > min_incr).nonzero()

        self.borders = x_new[ix], y_new[iy]

        if has_draw:
            plt.figure()
            extension = [self.x[0], self.x[-1], self.y[0], self.y[-1]]
            plt.imshow(t.transpose(), extent=extension, aspect='auto', alpha=0.5, cmap='gray')

        return self.borders


    @check_none('x','y','Ex','Ey','Ez',raise_exception=bool_raise_exception)
    def draw(self,
             kind: Draw_Vector_XY_Options = 'intensity',
             logarithm: float = 0,
             normalize: bool = False,
             cut_value: float | None = None,
             draw_borders: bool = True,
             num_ellipses: tuple[int, int] = (11, 11),
             amplification: float = 0.5,
             filename: str = '',
             draw: bool = True,
             **kwargs):
        """Draws electromagnetic field

        Args:
            kind (str):  'intensity', 'intensities', intensities_rz, 'phases', fields', 'stokes', 'param_ellipse', 'ellipses'
            logarithm (float): If >0, intensity is scaled in logarithm
            normalize (bool): If True, max(intensity)=1
            cut_value (float): If not None, cuts the maximum intensity to this value
            num_ellipses (int): number of ellipses for parameters_ellipse
            amplification (float): amplification of ellipses
            filename (str): if not '' stores drawing in file,
        """
        if self.n is not None:
            self.surface_detection()

        draw_borders = 0
        if draw is True:
            if kind == 'intensity':
                id_fig = self.__draw_intensity__(logarithm, normalize, cut_value, **kwargs)
            elif kind == 'intensities':
                id_fig = self.__draw_intensities__(logarithm, normalize, cut_value, **kwargs)

            elif kind == 'intensities_rz':
                id_fig = self.__draw_intensities_rz__(logarithm, normalize, cut_value, **kwargs)

            elif kind == 'phases':
                id_fig = self.__draw_phases__(**kwargs)

            elif kind == 'EH':
                id_fig = self.__draw_EH__(logarithm, normalize, cut_value, **kwargs)

            elif kind == 'E2H2':
                id_fig = self.__draw_E2H2__(logarithm, normalize, cut_value, **kwargs)

            elif kind == 'fields':
                id_fig = self.__draw_fields__(logarithm, normalize, cut_value, **kwargs)

            elif kind == 'stokes':
                id_fig = self.__draw_stokes__(logarithm, normalize, cut_value, **kwargs)

            elif kind == 'param_ellipse':
                id_fig = self.__draw_param_ellipse__(**kwargs)

            elif kind == 'ellipses':
                id_fig = self.__draw_ellipses__(logarithm, normalize, cut_value, num_ellipses, amplification, **kwargs)

                
            else:
                print("not good kind parameter in vector_fields_XY.draw()")
                id_fig = None

            if filename != '':
                plt.savefig(filename, dpi=100, bbox_inches='tight', pad_inches=0.1)

            return id_fig


    @check_none('x','y','Ex','Ey','Ez',raise_exception=bool_raise_exception)
    def __draw_intensity__(self,
                           logarithm: float,
                           normalize: bool,
                           cut_value: float,
                           draw_borders: bool=False,
                           color_intensity: str = CONF_DRAWING['color_intensity'],
                           **kwargs):
        """Draws the intensity

        Args:
            logarithm (float): If >0, intensity is scaled in logarithm
            normalize (bool): If True, max(intensity)=1
            cut_value (float): If not None, cuts the maximum intensity to this value
        """

        intensity = self.get('intensity')

        intensity = reduce_matrix_size(self.reduce_matrix, self.x, self.y, intensity)

        intensity = normalize_draw(intensity, logarithm, normalize, cut_value)

        plt.figure()
        h1 = plt.subplot(1, 1, 1)
        self.__draw1__(intensity, color_intensity, "")
        draw_edges(self, plt, draw_borders, **kwargs)
        plt.subplots_adjust(left=0,
                            bottom=0,
                            right=1,
                            top=1,
                            wspace=0.05,
                            hspace=0)
        plt.tight_layout()
        return h1


    @check_none('x','y','Ex','Ey','Ez',raise_exception=bool_raise_exception)
    def __draw_phases__(self, color_phase: str = CONF_DRAWING['color_phase'], draw_borders: bool=False,
                           **kwargs):
        """internal funcion: draws intensity X,Y.

        Args:
            logarithm (float): If >0, intensity is scaled in logarithm
            normalize (bool): If True, max(intensity)=1
            cut_value (float): If not None, cuts the maximum intensity to this value
        """

        Ex_r = reduce_matrix_size(self.reduce_matrix, self.x, self.y, self.Ex)
        Ey_r = reduce_matrix_size(self.reduce_matrix, self.x, self.y, self.Ey)
        Ez_r = reduce_matrix_size(self.reduce_matrix, self.x, self.y, self.Ez)
        tx, ty = rcParams['figure.figsize']

        intensity1 = np.abs(Ex_r)**2
        intensity2 = np.abs(Ey_r)**2
        intensity3 = np.abs(Ez_r)**2

        intensity_max = np.max(
            (intensity1.max(), intensity2.max(), intensity3.max()))

        if intensity3.max() < percentage_Ez * intensity_max:
            plt.figure(figsize=(2 * tx, ty))

            h1 = plt.subplot(1, 2, 1)

            phase = np.angle(Ex_r)
            intensity = np.abs(Ex_r)**2

            phase[intensity < percentage_intensity * (intensity.max())] = 0

            self.__draw1__(phase/degrees, color_phase, "$\phi_x$")
            plt.clim(-180, 180)

            h2 = plt.subplot(1, 2, 2)

            phase = np.angle(Ey_r)
            intensity = np.abs(Ey_r)**2

            phase[intensity < percentage_intensity * (intensity.max())] = 0

            self.__draw1__(phase/degrees, color_phase, "$\phi_y$")
            plt.clim(-180, 180)

            plt.subplots_adjust(left=0,
                                bottom=0,
                                right=1,
                                top=1,
                                wspace=0.05,
                                hspace=0)
            plt.tight_layout()
            plt.ylabel('')
            plt.gca().set_yticks([])
            rcParams['figure.figsize']= tx, ty

            return h1, h2
        else:

            plt.figure(figsize=(3 * tx, ty))

            h1 = plt.subplot(1, 3, 1)

            phase = np.angle(Ex_r)
            intensity = np.abs(Ex_r)**2

            phase[intensity < percentage_intensity * (intensity.max())] = 0

            self.__draw1__(phase/degrees, color_phase, "$\phi_x$")
            plt.clim(-180, 180)

            h2 = plt.subplot(1, 3, 2)

            phase = np.angle(Ey_r)
            intensity = np.abs(Ey_r)**2

            phase[intensity < percentage_intensity * (intensity.max())] = 0

            self.__draw1__(phase/degrees, color_phase, "$\phi_y$")
            plt.clim(-180, 180)
            plt.ylabel('')
            plt.gca().set_yticks([])
            
            h3 = plt.subplot(1, 3, 3)

            phase = np.angle(Ez_r)
            intensity = np.abs(Ez_r)**2

            phase[intensity < percentage_intensity * (intensity.max())] = 0

            self.__draw1__(phase/degrees, color_phase, "$\phi_z$")
            plt.clim(-180, 180)
            plt.ylabel('')
            plt.gca().set_yticks([])
            plt.subplots_adjust(left=0,
                                bottom=0,
                                right=1,
                                top=1,
                                wspace=0.05,
                                hspace=0)
            plt.tight_layout()

            rcParams['figure.figsize']= tx, ty

            return h1, h2, h3


    @check_none('x','y','Ex','Ey','Ez',raise_exception=False)
    def __draw_intensities__(self,
                             logarithm: float,
                             normalize: bool,
                             cut_value: float,
                             draw_borders: bool=False,
                             color_intensity: str = CONF_DRAWING['color_intensity'],
                             **kwargs):
        """internal funcion: draws phase X,Y, Z.

        Args:
            logarithm (float): If >0, intensity is scaled in logarithm
            normalize (bool): If True, max(intensity)=1
            cut_value (float): If not None, cuts the maximum intensity to this value
        """

        Ex_r = reduce_matrix_size(self.reduce_matrix, self.x, self.y, self.Ex)
        Ey_r = reduce_matrix_size(self.reduce_matrix, self.x, self.y, self.Ey)
        Ez_r = reduce_matrix_size(self.reduce_matrix, self.x, self.y, self.Ez)
        tx, ty = rcParams['figure.figsize']

        intensity1 = np.abs(Ex_r)**2
        intensity1 = normalize_draw(intensity1, logarithm, normalize,
                                    cut_value)

        intensity2 = np.abs(Ey_r)**2
        intensity2 = normalize_draw(intensity2, logarithm, normalize,
                                    cut_value)
    
        
        intensity3 = np.abs(Ez_r)**2
        intensity3 = normalize_draw(intensity3, logarithm, normalize,
                                    cut_value)

        intensity_max = np.max(
            (intensity1.max(), intensity2.max(), intensity3.max()))

        if intensity3.max() < percentage_Ez * intensity_max:
            plt.figure(figsize=(tx, ty))

            h1 = plt.subplot(1, 2, 1)

            self.__draw1__(intensity1, color_intensity, "$I_x$")
            plt.clim(0, intensity_max)

            h2 = plt.subplot(1, 2, 2)

            self.__draw1__(intensity2, color_intensity, "$I_y$")
            plt.clim(0, intensity_max)

            plt.ylabel('')
            plt.gca().set_yticks([])
        

            plt.subplots_adjust(left=0,
                                bottom=0,
                                right=1,
                                top=1,
                                wspace=0.05,
                                hspace=0)
            plt.tight_layout()

            return h1, h2
        else:

            plt.figure(figsize=(1.5 * tx, 1 * ty))

            h1 = plt.subplot(1, 3, 1)

            self.__draw1__(intensity1, color_intensity, "$I_x$")
            # plt.clim(0, intensity_max)

            h2 = plt.subplot(1, 3, 2)

            self.__draw1__(intensity2, color_intensity, "$I_y$")
            # plt.clim(0, intensity_max)

            plt.ylabel('')
            plt.gca().set_yticks([])
        
            h3 = plt.subplot(1, 3, 3)

            self.__draw1__(intensity3, color_intensity, "$I_z$")
            # plt.clim(0, intensity_max)

            plt.ylabel('')
            plt.gca().set_yticks([])
        
            plt.subplots_adjust(left=0,
                                bottom=0,
                                right=1,
                                top=1,
                                wspace=0.05,
                                hspace=0)
            plt.tight_layout()

            return h1, h2, h3


    @check_none('x','y','Ex','Ey','Ez',raise_exception=bool_raise_exception)
    def __draw_intensities_rz__(
            self,
            logarithm: float,
            normalize: bool,
            cut_value: float,
            draw_borders: bool=False,
            color_intensity: str = CONF_DRAWING['color_intensity'],
            **kwargs):
        """internal funcion: draws intensity X,Y.

        Args:
            logarithm (float): If >0, intensity is scaled in logarithm
            normalize (bool): If True, max(intensity)=1
            cut_value (float): If not None, cuts the maximum intensity to this value
        """

        Ex_r = reduce_matrix_size(self.reduce_matrix, self.x, self.y, self.Ex)
        Ey_r = reduce_matrix_size(self.reduce_matrix, self.x, self.y, self.Ey)
        Ez_r = reduce_matrix_size(self.reduce_matrix, self.x, self.y, self.Ez)
        tx, ty = rcParams['figure.figsize']

        intensity_r = np.abs(Ex_r)**2 + np.abs(Ey_r)**2
        intensity_r = normalize_draw(intensity_r, logarithm, normalize, cut_value)

        intensity_z = np.abs(Ez_r)**2
        intensity_z = normalize_draw(intensity_z, logarithm, normalize, cut_value)
        
        intensity_max = np.max((intensity_r, intensity_z))

        plt.figure(figsize=(tx, ty))

        h1 = plt.subplot(1, 2, 1)

        self.__draw1__(intensity_r, color_intensity, "$I_r$")
        plt.clim(0, intensity_max)

        h2 = plt.subplot(1, 2, 2)

        self.__draw1__(intensity_z, color_intensity, "$I_z$")
        plt.ylabel('')
        plt.gca().set_yticks([])
        
        plt.subplots_adjust(left=0,
                            bottom=0,
                            right=1,
                            top=1,
                            wspace=0.05,
                            hspace=0)
        plt.tight_layout()

        return h1, h2


    @check_none('x','y','Ex','Ey','Ez',raise_exception=bool_raise_exception)
    def __draw_fields__(self,
                        logarithm: float,
                        normalize: bool,
                        cut_value: float,
                        draw_borders: bool=False,
                        color_intensity: str = CONF_DRAWING['color_intensity'],
                        color_phase: str = CONF_DRAWING['color_phase'],
                        **kwargs):
        """__internal__: draws amplitude and phase in 2x2 drawing

        Args:
            logarithm (float): If >0, intensity is scaled in logarithm
            normalize (bool): If True, max(intensity)=1
            title (str): title of figure
            cut_value (float): If not None, cuts the maximum intensity to this value

        """
        Ex_r = reduce_matrix_size(self.reduce_matrix, self.x, self.y, self.Ex)
        Ey_r = reduce_matrix_size(self.reduce_matrix, self.x, self.y, self.Ey)
        Ez_r = reduce_matrix_size(self.reduce_matrix, self.x, self.y, self.Ez)

        amplitude1 = np.sqrt(np.abs(Ex_r)**2)
        amplitude1 = normalize_draw(amplitude1, logarithm, normalize,
                                    cut_value)

        amplitude2 = np.sqrt(np.abs(Ey_r)**2)
        amplitude2 = normalize_draw(amplitude2, logarithm, normalize,
                                    cut_value)

        amplitude3 = np.sqrt(np.abs(Ez_r)**2)
        amplitude3 = normalize_draw(amplitude3, logarithm, normalize,
                                    cut_value)

        amplitude_max = np.max(
            (amplitude1.max(), amplitude2.max(), amplitude3.max()))

        tx, ty = rcParams['figure.figsize']

        if amplitude3.max() < percentage_Ez * amplitude_max:
            plt.figure(figsize=(1.1 * tx, 1.5 * ty))

            h1 = plt.subplot(2, 2, 1)

            self.__draw1__(amplitude1, color_intensity, "$A_x$")
            plt.clim(0, amplitude_max)
            plt.xlabel('')
            plt.gca().set_xticks([])
            h2 = plt.subplot(2, 2, 2)
            self.__draw1__(amplitude2, color_intensity, "$A_y$")
            plt.clim(0, amplitude_max)
            plt.xlabel('')
            plt.gca().set_xticks([])
            plt.ylabel('')
            plt.gca().set_yticks([])
            h3 = plt.subplot(2, 2, 3)
            phase = np.angle(self.Ex)
            phase[amplitude1 < percentage_intensity * (amplitude1.max())] = 0

            self.__draw1__(phase/degrees, color_phase, "$\phi_x$")
            plt.clim(-180, 180)

            h4 = plt.subplot(2, 2, 4)
            phase = np.angle(self.Ey)
            phase[amplitude2 < percentage_intensity * (amplitude2.max())] = 0

            self.__draw1__(phase/degrees, color_phase, "$\phi_y$")
            plt.clim(-180, 180)
            plt.ylabel('')
            plt.gca().set_yticks([])
            h4 = plt.gca()
            
            plt.subplots_adjust(left=0,
                                bottom=0,
                                right=1,
                                top=1,
                                wspace=0.05,
                                hspace=0)
            plt.tight_layout()
            return h1, h2, h3, h4
        else:
            plt.figure(figsize=( 1.5 * tx,  1.75 * ty))

            h1 = plt.subplot(2, 3, 1)

            self.__draw1__(amplitude1, color_intensity, "$A_x$")
            plt.clim(0, amplitude_max)
            plt.xlabel('')
            plt.gca().set_xticks([])
            h2 = plt.subplot(2, 3, 2)
            self.__draw1__(amplitude2, color_intensity, "$A_y$")
            plt.clim(0, amplitude_max)
            plt.xlabel('')
            plt.gca().set_xticks([])
            plt.ylabel('')
            plt.gca().set_yticks([])
            
            h3 = plt.subplot(2, 3, 3)
            self.__draw1__(amplitude3, color_intensity, "$A_z$")
            plt.clim(0, amplitude_max)
            plt.xlabel('')
            plt.gca().set_xticks([])
            plt.ylabel('')
            plt.gca().set_yticks([])
            
            h4 = plt.subplot(2, 3, 4)
            phase = np.angle(self.Ex)
            phase[amplitude1 < percentage_intensity * (amplitude1.max())] = 0

            self.__draw1__(phase/degrees, color_phase, "$\phi_x$")
            plt.clim(-180, 180)

            h5 = plt.subplot(2, 3, 5)
            phase = np.angle(self.Ey)
            phase[amplitude2 < percentage_intensity * (amplitude2.max())] = 0

            self.__draw1__(phase/degrees, color_phase, "$\phi_y$")
            plt.clim(-180, 180)
            plt.ylabel('')
            plt.gca().set_yticks([])
            
            
            h6 = plt.subplot(2, 3, 6)
            phase = np.angle(self.Ez)
            phase[amplitude3 < percentage_intensity * (amplitude3.max())] = 0
            self.__draw1__(phase/degrees, color_phase, "$\phi_z$")
            plt.clim(-180, 180)
            plt.ylabel('')
            plt.gca().set_yticks([])
            
            plt.subplots_adjust(left=0,
                                bottom=0,
                                right=1,
                                top=1,
                                wspace=0.025,
                                hspace=0)
            plt.tight_layout()
            return h1, h2, h3, h4, h5, h6


    @check_none('x','y','Ex','Ey','Ez','Hx','Hy','Hz',raise_exception=bool_raise_exception)
    def __draw_EH__(
        self,
        logarithm,
        normalize,
        cut_value,
        draw_borders: bool=False,
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

        E_x = self.Ex.transpose()
        E_x = normalize_draw(E_x, logarithm, normalize, cut_value)

        E_y = self.Ey.transpose()
        E_y = normalize_draw(E_y, logarithm, normalize, cut_value)

        E_z = self.Ez.transpose()
        E_z = normalize_draw(E_z, logarithm, normalize, cut_value)

        H_x = self.Hx.transpose()
        H_x = normalize_draw(H_x, logarithm, normalize, cut_value)

        H_y = self.Hy.transpose()
        H_y = normalize_draw(H_y, logarithm, normalize, cut_value)

        H_z = self.Hz.transpose()
        H_z = normalize_draw(H_z, logarithm, normalize, cut_value)

        tx, ty = rcParams["figure.figsize"]

        E_max = np.max((E_x.max(), E_y.max(), E_z.max()))
        H_max = np.max((H_x.max(), H_y.max(), H_z.max()))

        if draw_z is True:

            
            fig, axs = plt.subplots(
                nrows=2, ncols=3, sharex=True, sharey=True, figsize=(1.5 * tx, 1.5 * ty)
            )

            id_fig, ax, IDimage = draw2D_XY(
                E_x, self.x, self.y, ax=axs[0, 0], xlabel="", ylabel="y $(\mu m)$", color=cmap, title=r'E$_x$')
            plt.axis(scale)
            draw_edges(self, axs[0, 0], draw_borders, **kwargs)
            IDimage.set_clim(-E_max,E_max)

            id_fig, ax, IDimage = draw2D_XY(
                E_y, self.x, self.y, ax=axs[0, 1], xlabel="", ylabel="", color=cmap, title=r'E$_y$')
            plt.axis(scale)
            draw_edges(self, axs[0, 1], draw_borders, **kwargs)
            IDimage.set_clim(-E_max,E_max)

            id_fig, ax, IDimage = draw2D_XY(
                E_z, self.x, self.y, ax=axs[0, 2], xlabel="", ylabel="", color=cmap, title=r'E$_z$')
            plt.axis(scale)
            draw_edges(self, axs[0, 2], draw_borders, **kwargs)
            IDimage.set_clim(-E_max,E_max)
            # ax.colorbar()


            id_fig, ax, IDimage = draw2D_XY(
                H_x, self.x, self.y, ax=axs[1, 0], xlabel="x $(\mu m)$", ylabel="y $(\mu m)$", color=cmap, title=r'H$_x$')
            plt.axis(scale)
            draw_edges(self, axs[1, 0], draw_borders, **kwargs)
            IDimage.set_clim(-H_max,H_max)

            id_fig, ax, IDimage = draw2D_XY(
                H_y, self.x, self.y, ax=axs[1, 1], xlabel="x $(\mu m)$", ylabel="", color=cmap, title=r'H$_y$')
            plt.axis(scale)
            draw_edges(self, axs[1, 1], draw_borders, **kwargs)
            IDimage.set_clim(-H_max,H_max)

            id_fig, ax, IDimage = draw2D_XY(
                H_z, self.x, self.y, ax=axs[1,2], xlabel="x $(\mu m)$", ylabel="", color=cmap, title=r'H$_z$')
            # ax.colorbar()
            plt.axis(scale)
            draw_edges(self, axs[1,2], draw_borders, **kwargs)
            IDimage.set_clim(-H_max,H_max)

        
            plt.tight_layout()
        else: 
            fig, axs = plt.subplots(
                nrows=2, ncols=2, sharex=True, sharey=True, figsize=(1 * tx, 1.5 * ty)
            )

            id_fig, ax, IDimage = draw2D_XY(
                E_x, self.x, self.y, ax=axs[0, 0], xlabel="", ylabel="y $(\mu m)$", color=cmap, title=r'E$_x$')
            draw_edges(self, axs[0,0], draw_borders, **kwargs)
            IDimage.set_clim(-E_max,E_max)
            id_fig, ax, IDimage = draw2D_XY(
                E_y, self.x, self.y, ax=axs[0, 1],  xlabel="x $(\mu m)$", ylabel="y $ (\mu m)$", color=cmap, title=r'E$_y$')
            draw_edges(self, axs[0,1], draw_borders, **kwargs)
            IDimage.set_clim(-E_max,E_max)


            id_fig, ax, IDimage = draw2D_XY(
                H_x, self.x, self.y, ax=axs[1, 0], xlabel="", ylabel="", color=cmap, title=r'H$_x$')
            draw_edges(self, axs[1,0], draw_borders, **kwargs)
            IDimage.set_clim(-H_max,H_max)
            id_fig, ax, IDimage = draw2D_XY(
                H_y, self.x, self.y, ax=axs[1, 1],  xlabel="x $ (\mu m)$", ylabel="", color=cmap, title=r'H$_y$')
            draw_edges(self, axs[1,1], draw_borders, **kwargs)
            IDimage.set_clim(-H_max,H_max)

        fig.subplots_adjust(right=1.25)
        cb_ax = fig.add_axes([0.2, 0, 0.6, 0.025])
        cbar = fig.colorbar(id_fig, cmap=cmap, cax=cb_ax, orientation='horizontal', shrink=0.5)
        plt.tight_layout()
            
        return self



    @check_none('x','y','Ex','Ey','Ez','Hx','Hy','Hz',raise_exception=bool_raise_exception)
    def __draw_E2H2__(
        self,
        logarithm,
        normalize,
        cut_value,
        draw_borders: bool=False,
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

        E2_x = np.abs(self.Ex.transpose())**2
        E2_x = normalize_draw(E2_x, logarithm, normalize, cut_value)

        E2_y = np.abs(self.Ey.transpose())**2
        E2_y = normalize_draw(E2_y, logarithm, normalize, cut_value)

        E2_z = np.abs(self.Ez.transpose())**2
        E2_z = normalize_draw(E2_z, logarithm, normalize, cut_value)

        H2_x = np.abs(self.Hx.transpose())**2
        H2_x = normalize_draw(H2_x, logarithm, normalize, cut_value)

        H2_y = np.abs(self.Hy.transpose())**2
        H2_y = normalize_draw(H2_y, logarithm, normalize, cut_value)

        H2_z = np.abs(self.Hz.transpose())**2
        H2_z = normalize_draw(H2_z, logarithm, normalize, cut_value)

        tx, ty = rcParams["figure.figsize"]

        E2_max = np.max((E2_x.max(), E2_y.max(), E2_z.max()))
        H2_max = np.max((H2_x.max(), H2_y.max(), H2_z.max()))

        if draw_z is True:

            
            fig, axs = plt.subplots(
                nrows=2, ncols=3, sharex=True, sharey=True, figsize=(1.5 * tx, 1.5 * ty)
            )

            id_fig, ax, IDimage = draw2D_XY(
                E2_x, self.x, self.y, ax=axs[0, 0], xlabel="", ylabel="y $(\mu m)$", color=cmap, title=r'E$^2_x$')
            plt.axis(scale)
            #draw_edges(self, axs[0, 0], draw_borders, color='k.')
            IDimage.set_clim(0,E2_max)

            id_fig, ax, IDimage = draw2D_XY(
                E2_y, self.x, self.y, ax=axs[0, 1], xlabel="", ylabel="", color=cmap, title=r'E$^2_y$')
            plt.axis(scale)
            #draw_edges(self, axs[1, 0], draw_borders, color='k.')
            IDimage.set_clim(0,E2_max)

            id_fig, ax, IDimage = draw2D_XY(
                E2_z, self.x, self.y, ax=axs[0, 2], xlabel="", ylabel="", color=cmap, title=r'E$^2_z$')
            plt.axis(scale)
            #draw_edges(self, axs[2, 0], draw_borders, color='k.')
            IDimage.set_clim(0,E2_max)
            # ax.colorbar()

            id_fig, ax, IDimage = draw2D_XY(
                H2_x, self.x, self.y, ax=axs[1, 0], xlabel="x $(\mu m)$", ylabel="y $(\mu m)$", color=cmap, title=r'H$^2_x$')
            plt.axis(scale)
            #draw_edges(self, axs[0, 1], draw_borders, color='k.')
            IDimage.set_clim(0,H2_max)

            id_fig, ax, IDimage = draw2D_XY(
                H2_y, self.x, self.y, ax=axs[1, 1], xlabel="x $(\mu m)$", ylabel="", color=cmap, title=r'H$^2_y$')
            plt.axis(scale)
            #draw_edges(self, axs[1, 1], draw_borders, color='k.')
            IDimage.set_clim(0,H2_max)

            id_fig, ax, IDimage = draw2D_XY(
                H2_z, self.x, self.y, ax=axs[1,2], xlabel="x $(\mu m)$", ylabel="", color=cmap, title=r'H$^2_z$')
            # ax.colorbar()
            plt.axis(scale)
            #draw_edges(self, axs[2, 1], draw_borders, color='k.')
            IDimage.set_clim(0,H2_max)

        
            plt.tight_layout()
        else: 
            fig, axs = plt.subplots(
                nrows=2, ncols=2, sharex=True, sharey=True, figsize=(1 * tx, 1.5 * ty)
            )

            id_fig, ax, IDimage = draw2D_XY(
                E2_x, self.x, self.y, ax=axs[0, 0], xlabel="", ylabel="y $(\mu m)$", color=cmap, title=r'E$_x$')
            #draw_edges(self, plt,  draw_borders, color='k.')
            IDimage.set_clim(0,E2_max)
            id_fig, ax, IDimage = draw2D_XY(
                E2_y, self.x, self.y, ax=axs[0, 1],  xlabel="x $(\mu m)$", ylabel="y $ (\mu m)$", color=cmap, title=r'E$_y$')
            #draw_edges(self, plt,  draw_borders, color='k.')
            IDimage.set_clim(0,E2_max)


            id_fig, ax, IDimage = draw2D_XY(
                H2_x, self.x, self.y, ax=axs[1, 0], xlabel="", ylabel="", color=cmap, title=r'H$_x$')
            #draw_edges(self, plt,  draw_borders, color='k.')
            IDimage.set_clim(0,H2_max)
            id_fig, ax, IDimage = draw2D_XY(
                H2_y, self.x, self.y, ax=axs[1, 1],  xlabel="x $ (\mu m)$", ylabel="", color=cmap, title=r'H$_y$')
            #draw_edges(self, plt,  draw_borders, color='k.')
            IDimage.set_clim(0,H2_max)

        fig.subplots_adjust(right=1.25)
        cb_ax = fig.add_axes([0.2, 0, 0.6, 0.025])
        cbar = fig.colorbar(id_fig, cmap=cmap, cax=cb_ax, orientation='horizontal', shrink=0.5)
        plt.tight_layout()
            
        return self



    @check_none('x','y','Ex','Ey','Ez',raise_exception=bool_raise_exception)
    def __draw_stokes__(self,
                        logarithm: float,
                        normalize: bool,
                        cut_value: float,
                        draw_borders: bool=False,
                        color_intensity: str = CONF_DRAWING['color_intensity'],
                        color_stokes: str = CONF_DRAWING['color_stokes']):
        """__internal__: computes and draws CI, CQ, CU, CV parameters
        """

        tx, ty = rcParams['figure.figsize']

        S0, S1, S2, S3 = self.get('stokes')
        S0 = normalize_draw(S0, logarithm, normalize, cut_value)
        S1 = normalize_draw(S1, logarithm, normalize, cut_value)
        S2 = normalize_draw(S2, logarithm, normalize, cut_value)
        S3 = normalize_draw(S3, logarithm, normalize, cut_value)

        intensity_max = S0.max()

        plt.figure(figsize=(1.75 * tx, 0.95 * ty))
        h1 = plt.subplot(1,4, 1)
        self.__draw1__(S0, color_intensity, "$S_0$")
        plt.clim(0, intensity_max)

        h2 = plt.subplot(1,4, 2)
        self.__draw1__(S1, color_stokes, "$S_1$")
        plt.clim(-intensity_max, intensity_max)
        plt.ylabel('')
        plt.gca().set_yticks([])
        
        h3 = plt.subplot(1,4, 3)
        self.__draw1__(S2, color_stokes, "$S_2$")
        plt.clim(-intensity_max, intensity_max)
        plt.ylabel('')
        plt.gca().set_yticks([])    
        
        h4 = plt.subplot(1,4, 4)
        self.__draw1__(S3, color_stokes, "$S_3$")
        plt.clim(-intensity_max, intensity_max)
        plt.ylabel('')
        plt.gca().set_yticks([])    

        plt.tight_layout()

        return (h1, h2, h3, h4)


    @check_none('x','y','Ex','Ey','Ez',raise_exception=bool_raise_exception)
    def __draw_param_ellipse__(self,
                               color_intensity: str = CONF_DRAWING['color_intensity'],
                               color_phase: str = CONF_DRAWING['color_phase'],
                               draw_borders: bool=False):
        """__internal__: computes and draws polariations ellipses
        """
        A, B, theta, h = self.polarization_ellipse(pol_state=None, matrix=True)

        A = reduce_matrix_size(self.reduce_matrix, self.x, self.y, A)
        B = reduce_matrix_size(self.reduce_matrix, self.x, self.y, B)
        theta = reduce_matrix_size(self.reduce_matrix, self.x, self.y, theta)
        h = reduce_matrix_size(self.reduce_matrix, self.x, self.y, h)

        tx, ty = rcParams['figure.figsize']

        plt.figure(figsize=(1.1 * tx, 1.75 * ty))

        max_intensity = max(A.max(), B.max())

        h1 = plt.subplot(2, 2, 1)
        self.__draw1__(A, color_intensity, "$A$")
        plt.clim(0, max_intensity)
        h2 = plt.subplot(2, 2, 2)
        self.__draw1__(B, color_intensity, "$B$")
        plt.clim(0, max_intensity)

        h3 = plt.subplot(2, 2, 3)
        self.__draw1__(theta/degrees, color_phase, "$\phi$")
        plt.clim(-180, 180)
        h4 = plt.subplot(2, 2, 4)
        self.__draw1__(h, "gist_heat", "$h$")
        plt.tight_layout()
        return (h1, h2, h3, h4)



    @check_none('x','y','Ex','Ey','Ez',raise_exception=bool_raise_exception)
    def __draw_ellipses__(self,
                          logarithm: float = 0.,
                          normalize: bool = False,
                          cut_value: float = '',
                          num_ellipses: tuple[int, int] = (21, 21),
                          amplification: float = 0.75,
                          draw_borders: bool=False,
                          color_line: str = 'w',
                          line_width: float = 0.5,
                          draw_arrow: bool = True,
                          head_width: float = .25,
                          ax: bool = False,
                          color_intensity: str = CONF_DRAWING['color_intensity']):
        """
        __draw_ellipses: Draw ellipses

        Args:
            logarithm (float, optional): _description_. Defaults to 0..
            normalize (bool, optional): _description_. Defaults to False.
            cut_value (float, optional): _description_. Defaults to ''.
            num_ellipses (tuple[int, int], optional): number of ellipses for parameters_ellipse. Defaults to (21, 21).
            amplification (float, optional): _description_. Defaults to 0.75.
            color_line (str, optional): _description_. Defaults to 'w'.
            line_width (float, optional): _description_. Defaults to 0.5.
            draw_arrow (bool, optional): _description_. Defaults to True.
            head_width (float, optional): _description_. Defaults to .25.
            ax (bool, optional): _description_. Defaults to False.
            color_intensity (str, optional): _description_. Defaults to CONF_DRAWING['color_intensity'].

            TODO: change color_line to two colors, one for right and another for left

        """
        
        percentage_intensity = CONF_DRAWING['percentage_intensity']
        intensity_max = (np.abs(self.Ex)**2 + np.abs(self.Ey)**2).max()

        Dx = self.x[-1] - self.x[0]
        Dy = self.y[-1] - self.y[0]


        size_x = Dx / (num_ellipses[0])
        size_y = Dy / (num_ellipses[1])
        x_centers = size_x/2 + size_x * np.array(range(0, num_ellipses[0]))
        y_centers = size_y/2 + size_y * np.array(range(0, num_ellipses[1]))

        num_x, num_y = len(self.x), len(self.y)
        ix_centers = num_x / (num_ellipses[0])
        iy_centers = num_y / (num_ellipses[1])

        ix_centers = (np.round(
            ix_centers/2 +
            ix_centers * np.array(range(0, num_ellipses[0])))).astype('int')
        iy_centers = (np.round(
            iy_centers/2 +
            iy_centers * np.array(range(0, num_ellipses[1])))).astype('int')

        Ix_centers, Iy_centers = np.meshgrid(ix_centers.astype('int'), iy_centers.astype('int'))

        verbose = False
        if verbose is True:
            print(num_x, num_y, ix_centers, iy_centers)
            print(Dx, Dy, size_x, size_y)
            print(x_centers, y_centers)
            print(Ix_centers, Iy_centers)

        E0x = self.Ex[Iy_centers, Ix_centers]
        E0y = self.Ey[Iy_centers, Ix_centers]

        angles = np.linspace(0, 360*degrees, 64)

        if ax is False:
            self.draw('intensity',
                      logarithm=logarithm,
                      color_intensity=color_intensity)
            ax = plt.gca()

        for i, xi in enumerate(ix_centers):
            for j, yj in enumerate(iy_centers):
                Ex = np.real(E0x[j, i] * np.exp(1j * angles))
                Ey = np.real(E0y[j, i] * np.exp(1j * angles))

                max_r = np.sqrt(np.abs(Ex)**2 + np.abs(Ey)**2).max()
                size_dim = min(size_x, size_y)

                if max_r > 0 and max_r**2 > percentage_intensity * intensity_max:

                    Ex = Ex / max_r * size_dim * amplification/2 + self.x[int(xi)]
                    Ey = Ey / max_r * size_dim * amplification/2 + self.y[int(yj)]
    
                    ax.plot(Ex, Ey, color_line, lw=line_width)
                    if draw_arrow:
                        ax.arrow(Ex[0],
                                 Ey[0],
                                 Ex[0] - Ex[1],
                                 Ey[0] - Ey[1],
                                 width=0,
                                 head_width=head_width,
                                 fc=color_line,
                                 ec=color_line,
                                 length_includes_head=False)
                # else:
                #     print(max_r, intensity_max,
                #           percentage_intensity * intensity_max)


    @check_none('x','y','Ex','Ey','Ez',raise_exception=bool_raise_exception)
    def __draw1__(self,
                  image: NDArrayFloat,
                  colormap: str,
                  title: str = '',
                  has_max: bool = False,
                  colorbar: bool = True,
                  only_image: bool = False):
        """Draws image.

        Args:
            image (numpy.array): array with drawing
            colormap (str): colormap
            title (str): title of drawing
        """
        extension = [self.x[0], self.x[-1], self.y[0], self.y[-1]]

        h = plt.imshow(image,
                       interpolation='bilinear',
                       aspect='auto',
                       origin='lower',
                       extent=extension)
        h.set_cmap(colormap)
        plt.axis('scaled')
        plt.axis(extension)

        if only_image is True:
            plt.axis('off')
            return h

        plt.title(title, fontsize=16)

        if has_max is True:
            text_up = "{}".format(image.max())
            plt.text(self.x.max(),
                     self.y.max(),
                     text_up,
                     fontsize=14,
                     bbox=dict(edgecolor='white',
                               facecolor='white',
                               alpha=0.75))

            text_down = "{}".format(image.min())
            plt.text(self.x.max(),
                     self.y.min(),
                     text_down,
                     fontsize=14,
                     bbox=dict(edgecolor='white',
                               facecolor='white',
                               alpha=0.75))

        if colorbar is True:
            plt.colorbar(orientation='horizontal', shrink=0.66, pad=0.15)

        plt.xlabel(r"$x  (\mu m)$")
        plt.ylabel(r"$y  (\mu m)$")
        h.set_clim(0, image.max())
        plt.subplots_adjust(0.01, 0.01, 0.99, 0.95, 0.05, 0.05)

        return h



def _compute1Elipse__(x0: float, y0: float, A: float, B: float, theta: float,
                      h: float = 0, amplification: float = 1):
    """computes polarization ellipse for drawing

    Args:
        x0 (float): position x of ellipse
        y0 (float): position y of ellipse
        A (float): axis 1 of ellipse
        B (float): axis 2 of ellipse
        theta (float): angle of ellipse
        h (float): to remove
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



def draw2D_XY(
        image,
        x,
        y,
        ax=None,
        xlabel="r$x  (\mu m)$",
        ylabel="r$y  (\mu m)$",
        title="",
        color="YlGnBu",  # YlGnBu  seismic
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
        reduction_y = int(num_y / 500)

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

    IDimage.set_cmap(color)

    return id_fig, ax, IDimage

