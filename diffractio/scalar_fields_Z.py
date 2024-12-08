# !/usr/bin/env python3


# ----------------------------------------------------------------------
# Name:        scalar_fields_Z.py
# Purpose:     Class for unidimensional scalar fields in the z-axis
#
# Author:      Luis Miguel Sanchez Brea
#
# Created:     2024
# Licence:     GPLv3
# ----------------------------------------------------------------------

"""
This module generates Scalar_field_Z class

The main atributes are:
    * self.z (numpy.array): linear array with equidistant positions. The number of data is preferibly :math:`2^n` .
    * self.wavelength (float): wavelength of the incident field.
    * self.u (numpy.array): equal size than  x. complex field

There are also some secondary atributes:
    * self.quality (float): quality of RS algorithm
    * self.info (str): description of data
    * self.type (str): Class of the field
    * self.date (str): date


*Class for unidimensional scalar fields*

*Definition of a scalar field*
    * instantiation, duplicate,  clear_field, print
    * save and load data



*Drawing functions*
    * draw

*Args:*
    * intensity, average intensity
    * get_edges_transitions (mainly for pylithography)

"""

import copy
import multiprocessing

from numpy import (angle, exp, linspace, pi, shape, zeros)
from scipy.interpolate import interp1d


from .__init__ import degrees, mm, np, plt
from .config import bool_raise_exception, Draw_Z_Options, get_scalar_options
from .utils_typing import npt, Any, NDArray,  NDArrayFloat, NDArrayComplex
from .utils_common import add, get_date, load_data_common, save_data_common, check_none, oversampling, get_scalar, rmul
from .utils_drawing import normalize_draw
from .utils_math import nearest

from .utils_optics import field_parameters, normalize_field, FWHM1D

num_max_processors = multiprocessing.cpu_count()

class Scalar_field_Z():
    """Class for unidimensional scalar fields in z axis.

    Args:
        z (numpy.array): linear array with equidistant positions.
        wavelength (float): wavelength of the incident field
        n_background (float): refractive index of background
        info (str): String with info about the simulation

    Attributes:
        self.z (numpy.array): Linear array with equidistant positions.
            The number of data is preferibly :math:`2^n`.
        self.wavelength (float): Wavelength of the incident field.
        self.u (numpy.array): Complex field. The size is equal to self.z.
        self.quality (float): Quality of RS algorithm.
        self.info (str): Description of data.
        self.type (str): Class of the field.
        self.date (str): Date when performed.
    """

    def __init__(self, z: NDArrayFloat | None = None, wavelength: float | None = None,
                 n_background: float = 1., info: str = ""):
        self.z = z
        self.wavelength = wavelength
        self.n_background = n_background
        if z is not None:
            self.u = zeros(shape(self.z), dtype=complex)
        else:
            self.u = None
        self.quality = 0
        self.info = info
        self.type = 'Scalar_field_Z'
        self.date = get_date()


    @check_none('z','u',raise_exception=bool_raise_exception)
    def __str__(self):
        """Represents main data of the atributes."""

        Imin = (np.abs(self.u)**2).min()
        Imax = (np.abs(self.u)**2).max()
        phase_min = (np.angle(self.u)).min()/degrees
        phase_max = (np.angle(self.u)).max()/degrees
        print("{}\n - z:  {},   u:  {}".format(self.type, self.z.shape,
                                               self.u.shape))
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


    @check_none('z','u',raise_exception=bool_raise_exception)
    def __add__(self, other):
        """Adds two Scalar_field_Z.

        Args:
            other (Scalar_field_Z): 2nd field to add
 
        Returns:
            Scalar_field_Z: `u3 = u1 + u2`
        """

        u = add(self, other, kind='source')

        return u


    @check_none('z','u',raise_exception=bool_raise_exception)
    def __sub__(self, other):
        """Substract two Scalar_field_x. For example two light sources or two masks.

        Args:
            other (Scalar_field_X): field to substract

        Returns:
            Scalar_field_X: `u3 = u1 - u2`

        """

        u3 = Scalar_field_Z(self.z, self.wavelength)
        u3.u = self.u - other.u
        return u3




    @check_none('z','u',raise_exception=bool_raise_exception)
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


        t = rmul(self, number, kind='amplitude')
            
        return t


    @check_none('z','u',raise_exception=bool_raise_exception)
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

    def size(self, verbose: bool = False):
        """returns the size of the instance in MB.

        Args:
            verbose (bool, optional): prints size in Mb. Defaults to False.

        Returns:
            float: size in MB
        """

        return get_instance_size_MB(self, verbose)

        

    @check_none('z','u',raise_exception=bool_raise_exception)
    def duplicate(self, clear: bool = False):
        """Duplicates the instance"""
        new_field = copy.deepcopy(self)
        if clear is True:
            new_field.clear_field()
        return new_field


    @check_none('u',raise_exception=bool_raise_exception)
    def conjugate(self, new_field: bool = True):
        """Conjugates the field
        """

        if new_field is True:
            u_new = self.duplicate()
            u_new.u = np.conj(self.u)
            return u_new
        else:
            self.u = np.conj(self.u)


    @check_none('u',raise_exception=bool_raise_exception)
    def clear_field(self):
        """Removes the field so that self.u = 0. """
        self.u = np.zeros_like(self.u, dtype=complex)


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
        """Load data from a file to a Scalar_field_X.
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


    @check_none('z','u',raise_exception=bool_raise_exception)
    def oversampling(self, factor_rate: int | tuple):
        """Overfample function has been implemented in scalar X, XY, XZ, and XYZ frames reduce the pixel size of the masks and fields. 
        This is also performed with the cut_resample function. However, this function oversamples with integer factors.
        
        Args:
            factor_rate (int | tuple, optional): factor rate. Defaults to 2.
        """

        self = oversampling(self, factor_rate)
        
    
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

    @check_none('z','u',raise_exception=bool_raise_exception)
    def cut_resample(self,
                     z_limits: NDArrayFloat | None = None,
                     num_points: int | None = None,
                     new_field: bool = False,
                     interp_kind: str = 'linear'):
        """Cuts the field to the range (z0,z1). If one of this z0,z1 positions is out of the self.z range it does nothing.
        It is also valid for resampling the field, just write z0,z1 as the limits of self.z

        Args:
            z_limits (numpy.array): (z0,z1) - starting and final points to cut, if '' - takes the current limit z[0] and z[-1]
            num_points (int): it resamples z, and u [],'',0,None -> it leave the points as it is
            new_field (bool): if True it returns a new Scalar_field_z
            interp_kind (str): 'linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic'

        Returns:
            (Scalar_field_Z): if new_field is True
        """

        if z_limits is None:
            # used only for resampling
            z0 = self.z[0]
            z1 = self.z[-1]
        else:
            z0, z1 = z_limits

        if z0 < self.z[0]:
            z0 = self.z[0]
        if z1 > self.z[-1]:
            z1 = self.z[-1]

        i_z0, _, _ = nearest(self.z, z0)
        i_z1, _, _ = nearest(self.z, z1)

        if num_points not in ([], '', 0, None):
            z_new = linspace(z0, z1, num_points)
            f_interp_abs = interp1d(self.z,
                                    np.abs(self.u),
                                    kind=interp_kind,
                                    bounds_error=False,
                                    fill_value=0)

            f_interp_phase = interp1d(self.z,
                                      np.imag(self.u),
                                      kind=interp_kind,
                                      bounds_error=False,
                                      fill_value=0)

            u_new_abs = f_interp_abs(z_new)
            u_new_phase = f_interp_phase(z_new)
            u_new = u_new_abs * np.ezp(1j * u_new_phase)

        else:
            i_s = slice(i_z0, i_z1)
            z_new = self.z[i_s]
            u_new = self.u[i_s]

        if new_field is False:
            self.z = z_new
            self.u = u_new
        elif new_field is True:
            field = Scalar_field_Z(z=z_new, wavelength=self.wavelength)
            field.u = u_new
            return field


    @check_none('u',raise_exception=bool_raise_exception)
    def normalize(self, kind='amplitude', new_field: bool = False):
        """Normalizes the field so that intensity.max()=1.

        Args:
            kind (str): 'amplitude', or 'intensity'
            new_field (bool): If False the computation goes to self.u. If True a new instance is produced

        Returns
            u (numpy.array): normalized optical field
        """
        return normalize_field(self, kind, new_field)


    @check_none('u',raise_exception=bool_raise_exception)
    def intensity(self):
        """Intensity.

        Returns:
            (numpy.array): Intensity
        """

        intensity = (np.abs(self.u)**2)
        return intensity


    @check_none('u',raise_exception=bool_raise_exception)
    def average_intensity(self, verbose: bool = False):
        """Returns the average intensity as: (np.abs(self.u)**2).sum() / num_data

        Args:
            verbose (bool): If True it prints the value of the average intensity.

        Returns:
            (float): average intensity.
        """
        average_intensity = (np.abs(self.u)**2).mean()
        if verbose is True:
            print("average intensity={} W/m").format(average_intensity)

        return average_intensity


    @check_none('z','u',raise_exception=bool_raise_exception)
    def FWHM1D(self, percentage: float = 0.5, remove_background: bool = None,
               has_draw: bool = False):
        """
        FWHM1D

        Args:
            percentage (float): value between 0 and 1. 0.5 means that the width is computed at half maximum.
            remove_background (str): 'min', 'mean', None
            has_draw (bool): If true it draws

        Returns:
            width (float): width, in this z case: DOF

        """

        intensities = np.abs(self.u)**2

        width = FWHM1D(self.z, intensities, percentage, remove_background,
                       has_draw)

        return np.squeeze(width)


    @check_none('z','u',raise_exception=bool_raise_exception)
    def DOF(self, percentage: float = 0.5, remove_background: str | None = None,
            has_draw: bool = False):
        """Determines Depth-of_focus (DOF) in terms of the width at different distances

        Args:
            percentage (float): value between 0 and 1. 0.5 means that the width is computed at half maximum.
            remove_background (str): 'min', 'mean', None
            has_draw (bool): If true it draws

        Returns:
            width (float): width, in this z case: D

        References:

            B. E. A. Saleh and M. C. Teich, Fundamentals of photonics. john Wiley & sons, 2nd ed. 2007. Eqs (3.1-18) (3.1-22) page 79

        Returns:

            (float): Depth of focus
            (float): beam waist
            (float, float, float): postions (z_min, z_0, z_max) of the depth of focus
        """

        return self.FWHM1D(percentage, remove_background, has_draw)

    def draw(self,
             kind: Draw_Z_Options = 'intensity',
             logarithm: float = 0.,
             normalize: bool = False,
             cut_value: float | None = None,
             z_scale: str = 'um',
             unwrap: bool = False,
             filename: str = ''):
        """Draws z field. There are several data from the field that are extracted, depending of 'kind' parameter.

        Args:
            kind (str): type of drawing: 'amplitude', 'intensity', 'field', 'phase'
            logarithm (float): If >0, intensity is scaled in logarithm
            normalize (bool): If True, max(intensity)=1
            cut_value (float): If not None, cuts the maximum intensity to this value
            unwrap (bool): If True, unwraps the phase.
            filename (str): if not '' stores drawing in file,
        """

        if self.z is None:
            print('could not draw file: self.z=None')
            return
        if z_scale == 'mm':
            z_drawing = self.z / mm
            zlabel = r'$z\,(mm)$'

        else:
            z_drawing = self.z
            zlabel = r'$z\,(\mu m)$'

        amplitude, intensity, phase = field_parameters(self.u)

        if unwrap:
            phase = np.unwrap(phase)

        plt.figure()

        if kind == 'intensity':
            y = intensity
        elif kind == 'phase':
            y = phase
        elif kind in ('amplitude', 'field'):
            y = amplitude

        if kind in ('intensity', 'amplitude', 'field'):
            y = normalize_draw(y, logarithm, normalize, cut_value)

        if kind == 'field':
            plt.subplot(211)
            plt.plot(z_drawing, y, 'k', lw=2)
            plt.xlabel(zlabel)
            plt.ylabel(r'$A\,(arb.u.)$')
            plt.xlim(left=z_drawing[0], right=z_drawing[-1])
            plt.ylim(bottom=0)

            plt.subplot(212)
            plt.plot(z_drawing, phase, 'k', lw=2)
            plt.xlabel(zlabel)
            plt.ylabel(r'$phase\,(radians)$')
            plt.xlim(left=z_drawing[0], right=z_drawing[-1])

        elif kind in ('amplitude', 'intensity', 'phase'):
            plt.plot(z_drawing, y, 'k', lw=2)
            plt.xlabel(zlabel)
            plt.ylabel(kind)
            plt.xlim(left=z_drawing[0], right=z_drawing[-1])

        if filename != '':
            plt.savefig(filename, dpi=100, bbox_inches='tight', pad_inches=0.1)

        if kind == 'intensity':
            plt.ylim(bottom=0)

        elif kind == 'phase':
            if unwrap == False:
                plt.ylim(-pi, pi)
