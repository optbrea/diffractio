# !/usr/bin/env python3

# ----------------------------------------------------------------------
# Name:        scalar_fields_X.py
# Purpose:     Class and functions for unidimensional scalar fields
#
# Author:      Luis Miguel Sanchez Brea
#
# Created:     2024
# Licence:     GPLv3
# ----------------------------------------------------------------------


"""
This module generates Scalar_field_X class and several functions for multiprocessing. It is required also for generating masks and fields.

The main atributes are:
    * self.x (numpy.array): linear array with equidistant positions. The number of data is preferibly :math:`2^n` .
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
    * add, substract sources
    * multiply masks and sources
    * save and load data

*Functions for generation of masks*
    * insert masks, insert_array_masks - insert other masks inside the mask
    * filter -

*Propagation*
    * fft, ifft - fourier transform
    * RS - Rayleigh Sommerfeld. It allows amplification of the field
    * BPM - Beam propagation method

*Drawing functions*
    * draw

*Args:*
    * intensity, average intensity
    * get_edges_transitions (mainly for pylithography)

*Functions*
    * kernelRS, kernelRSinverse

*Multiprocessing*
    * polychromatic_multiprocessing
    * extended_source_multiprocessing
    * extended_polychromatic_source

"""
# flake8: noqa


import copy
import copyreg
import multiprocessing
import time
import types


from numpy import angle, array, concatenate, exp, linspace, pi, shape, sqrt, zeros
from numpy.lib.scimath import sqrt as csqrt
from scipy.fftpack import fft, fftshift, ifft
from scipy.interpolate import interp1d
from scipy.special import hankel1

from .__init__ import degrees, mm, np, plt

from .config import bool_raise_exception, Draw_X_Options, get_scalar_options, empty_types
from .utils_typing import npt, Any, NDArray,  NDArrayFloat, NDArrayComplex
from .utils_common import (get_date, load_data_common, save_data_common, check_none, add, rmul,
                           oversampling, get_scalar, get_instance_size_MB)
from .utils_drawing import normalize_draw
from .utils_math import (fft_filter, get_edges, nearest, reduce_to_1, Bluestein_dft_x, get_k, nearest2)
from .utils_multiprocessing import (_pickle_method, _unpickle_method,
                                    execute_multiprocessing)
from .utils_optics import field_parameters, normalize_field

copyreg.pickle(types.MethodType, _pickle_method, _unpickle_method)

num_max_processors = multiprocessing.cpu_count()

class Scalar_field_X():
    """Class for unidimensional scalar fields.

    Args:
        x (numpy.array): linear array with equidistant positions.
            The number of data is preferibly :math:`2^n` .
        wavelength (float): wavelength of the incident field
        n_background (float): refractive index of background
        info (str): String with info about the simulation

    Attributes:
        self.x (numpy.array): Linear array with equidistant positions.
            The number of data is preferibly :math:`2^n`.
        self.wavelength (float): Wavelength of the incident field.
        self.u (numpy.array): Complex field. The size is equal to self.x.
        self.quality (float): Quality of RS algorithm.
        self.info (str): Description of data.
        self.type (str): Class of the field.
        self.date (str): Date when performed.
    """

    def __init__(self, x: NDArrayFloat | None = None, wavelength: float  = 0,
                 n_background: float = 1., info: str = ""):
        self.x = x
        self.wavelength = wavelength
        self.n_background = n_background
        if x is not None:
            self.u = np.zeros(shape(self.x), dtype=complex)
        else:
            self.u = None

        self.quality = 0
        self.info = info
        self.type = "Scalar_field_X"
        self.date = get_date()

    @check_none('x', 'u', raise_exception=bool_raise_exception)
    def __str__(self):
        """Represents main data of the atributes."""


        Imin = (np.abs(self.u) ** 2).min()
        Imax = (np.abs(self.u) ** 2).max()
        phase_min = (np.angle(self.u)).min()/degrees
        phase_max = (np.angle(self.u)).max()/degrees
        print("{}\n - x:  {},   u:  {}".format(self.type, self.x.shape, self.u.shape))
        print(
            " - xmin:       {:2.2f} um,  xmax:      {:2.2f} um,  Dx:   {:2.2f} um".format(
                self.x[0], self.x[-1], self.x[1] - self.x[0]
            )
        )
        print(" - Imin:       {:2.2f},     Imax:      {:2.2f}".format(Imin, Imax))
        print(
            " - phase_min:  {:2.2f} deg, phase_max: {:2.2f} deg".format(
                phase_min, phase_max
            )
        )

        print(" - wavelength: {:2.2f} um".format(self.wavelength))
        print(" - date:       {}".format(self.date))
        if self.info != "":
            print(" - info:       {}".format(self.info))
        return ""

    @check_none('x', 'u', raise_exception=bool_raise_exception)
    def __add__(self, other):
        """Adds two Scalar_field_x. For example two light sources or two masks.

        Args:
            other (Scalar_field_X): 2nd field to add
            kind (str): instruction how to add the fields: ['source', 'mask', 'phases', 'distances', 'no_overlap].
            - 'source': adds the fields as they are
            - 'mask': adds the fields as complex numbers and then normalizes so that the maximum amplitude is 1.
            - 'phases': adds the phases and then normalizes so that the maximum amplitude is 1.
            - 'np_overlap': adds the fields as they are. If the sum of the amplitudes is greater than 1, an error is produced
            - 'distances': adds the fields as they are. If the fields overlap, the field with the smallest distance is kept.

        Returns:
            Scalar_field_X:
        """

        if self.type == 'Scalar_mask_X':
            t = add(self, other, kind='mask')
        elif self.type == 'Scalar_source_X' or 'Scalar_field_X':
            t = add(self, other, kind='source')
            
        return t


    @check_none('x', 'u', raise_exception=bool_raise_exception)
    def add(self, other, kind):
        """Adds two Scalar_field_x. For example two light sources or two masks.

        Args:
            other (Vector_field_X): 2nd field to add
            kind (str): instruction how to add the fields: ['source', 'mask', 'phases', 'no_overlap', 'distances'].
                - 'source': adds the fields as they are
                - 'mask': adds the fields as complex numbers and then normalizes so that the maximum amplitude is 1.
                - 'phases': adds the phases and then normalizes so that the maximum amplitude is 1.
                - 'np_overlap': adds the fields as they are. If the sum of the amplitudes is greater than 1, an error is produced
                - 'distances': adds the fields as they are. If the fields overlap, the field with the smallest distance is kept.

        Returns:
            Scalar_field_X: `u3 = u1 + u2`
        """

        t = add(self, other, kind)
           
        return t

    @check_none('x', 'u', raise_exception=bool_raise_exception)
    def __sub__(self, other):
        """Substract two Scalar_field_x. For example two light sources or two masks.

        Args:
            other (Scalar_field_X): field to substract

        Returns:
            Scalar_field_X: `u3 = u1 - u2`

        TODO: It can be improved for masks (not having less than 1)
        """

        u3 = Scalar_field_X(self.x, self.wavelength)
        u3.u = self.u - other.u
        return u3

    @check_none('x', 'u', raise_exception=bool_raise_exception)
    def __mul__(self, other):
        """Multiply two fields. For example  :math: `u_1(x)= u_0(x)*t(x)`

        Args:
            other (Scalar_field_X): field to multiply

        Returns:
            Scalar_field_X: :math: `u_1(x)= u_0(x)*t(x)`
        """

        new_field = Scalar_field_X(self.x, self.wavelength)
        new_field.u = self.u * other.u
        return new_field


    @check_none('x', 'u', raise_exception=bool_raise_exception)
    def __rmul__(self, number: float | complex | int):
        """Multiply a field by a number.  For example  :math: `u_1(x)= m * u_0(x)`.

        Args:
            number (float | complex | int): number to multiply the field.
            kind (str): instruction how to add the fields: ['intensity', 'amplitude', 'phase'].
                - 'intensity': Multiply the intensity of the field by the number.
                - 'amplitude': Multiply the amplitude of the field by the number.
                - 'phase': Multiply the phase of the field by the number.

        Returns:
            Scalar_field_X:
        """

        if self.type == 'Scalar_mask_X':
            t = rmul(self, number, kind='intensity')
        elif self.type == 'Scalar_source_X' or 'Scalar_field_X':
            t = rmul(self, number, kind='amplitude')
            
        return t


    @check_none('x', 'u', raise_exception=bool_raise_exception)
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

        

    @check_none('u')
    def conjugate(self, new_field: bool = True):
        """Conjugates the field"""



        if new_field is True:
            u_new = self.duplicate()
            u_new.u = np.conj(self.u)
            return u_new
        else:
            self.u = np.conj(self.u)

    @check_none('x', 'u')
    def oversampling(self, factor_rate: int | tuple):
        """Overfample function has been implemented in scalar X, XY, XZ, and XYZ frames reduce the pixel size of the masks and fields. 
        This is also performed with the cut_resample function. However, this function oversamples with integer factors.
        
        Args:
            factor_rate (int | tuple, optional): factor rate. Defaults to 2.
        """

        self = oversampling(self, factor_rate)


    def duplicate(self, clear: bool = False):
        """Duplicates the instance

        Args:
            clear (bool): If True, clears the self.u argument, else it keeps the same than in self.

        Returns
            new_instance (scalar_field_X): duplicated instance.
        """

        new_field = copy.deepcopy(self)
        if clear is True:
            new_field.clear_field()
        return new_field

    def reduce_to_1(self):
        """All the values greater than 1 pass to 1. This is used for Scalar_masks when we add two masks."""

        self = reduce_to_1(self)

    @check_none('u', raise_exception=bool_raise_exception)
    def clear_field(self):
        """Removes the field so that self.u = 0."""
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
            final_filename = save_data_common(
                self, filename, add_name, description, verbose
            )
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
                if verbose is True:
                    print(dict0.keys())
            else:
                raise Exception("no dictionary in load_data")


    @check_none('x', 'u', raise_exception=bool_raise_exception)
    def cut_resample(
            self, x_limits: NDArrayFloat | None = None, num_points: int | None = None,
            new_field: bool = False, interp_kind: str = "linear"):
        """Cuts the field to the range (x0,x1). If one of this x0,x1 positions is out of the self.x range it does nothing.
        It is also valid for resampling the field, just write x0,x1 as the limits of self.x

        Args:
            x_limits (numpy.array): (x0,x1) - starting and final points to cut, if '' - takes the current limit x[0] and x[-1]
            num_points (int): it resamples x, and u [], '',0,None -> it leave the points as it is
            new_field (bool): if True it returns a new Scalar_field_X
            interp_kind (str): 'linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic'

        Returns:
            (Scalar_field_X): if new_field is True
        """
        

        if x_limits is None:
            # used only for resampling
            x0 = self.x[0]
            x1 = self.x[-1]
        else:
            x0, x1 = x_limits

        if x0 < self.x[0]:
            x0 = self.x[0]
        if x1 > self.x[-1]:
            x1 = self.x[-1]

        i_x0, _, _ = nearest(self.x, x0)
        i_x1, _, _ = nearest(self.x, x1)

        if num_points not in empty_types:
            x_new = linspace(x0, x1, num_points)
            f_interp_abs = interp1d(
                self.x,
                np.abs(self.u),
                kind=interp_kind,
                bounds_error=False,
                fill_value=0,
            )

            f_interp_phase = interp1d(
                self.x,
                np.imag(self.u),
                kind=interp_kind,
                bounds_error=False,
                fill_value=0,
            )

            u_new_abs = f_interp_abs(x_new)
            u_new_phase = f_interp_phase(x_new)
            u_new = u_new_abs * np.exp(1j * u_new_phase)

        else:
            i_s = slice(i_x0, i_x1)
            x_new = self.x[i_s]
            u_new = self.u[i_s]

        if new_field is False:
            self.x = x_new
            self.u = u_new
        elif new_field is True:
            field = Scalar_field_X(x=x_new, wavelength=self.wavelength)
            field.u = u_new
            return field

    def incident_field(self, u0):
        """Incident field for the experiment. It takes a Scalar_source_X field.

        Args:
            u0 (Scalar_source_X): field produced by Scalar_source_X (or a X field)
        """
        self.u = u0.u
        
        
    @check_none('u', raise_exception=bool_raise_exception)
    def get(self, kind: get_scalar_options):
        """Get parameters from Scalar field.

        Args:
            kind (str): 'intensity', 'phase', 'field'

        Returns:
            matrices with required values
        """

        data = get_scalar(self, kind)
        return data
        


    def normalize(self, kind='amplitude', new_field: bool = False):
        """Normalizes the field so that intensity.max()=1.

        Args:
            kind (str): 'amplitude', or 'intensity'
            new_field (bool): If False the computation goes to self.u. If True a new instance is produced.
        
        Returns
            u (numpy.array): normalized optical field
        """
        return normalize_field(self, kind, new_field)


    @check_none('u', raise_exception=bool_raise_exception)
    def inverse_amplitude(self, new_field: bool = False):
        """Inverts the amplitude of the mask, phase is equal as initial

        Args:
            new_field (bool): If True it returns a Scalar_mask_X object, else, it modifies the existing object.


        Returns:
            Scalar_mask_X:  If new_field is True, it returns a Scalar_mask_X object.
        """
        from diffractio.scalar_masks_X import Scalar_mask_X



        amplitude = np.abs(self.u)
        phase = np.angle(self.u)

        new_amplitude = (1 - amplitude) * np.exp(1.0j * phase)

        if new_field is False:
            self.u = new_amplitude
        else:
            new = Scalar_mask_X(self.x, self.wavelength)
            new.u = new_amplitude
            return new

    @check_none('x', 'u', raise_exception=bool_raise_exception)
    def inverse_phase(self, new_field: bool = False):
        """Inverts the phase of the mask, amplitude is equal as initial

        Args:
            new_field (bool): If True it returns a Scalar_mask_X object, else, it modifies the existing object.


        Returns:
            Scalar_mask_X:  If new_field is True, it returns a Scalar_mask_X object.
        """

        
        amplitude = np.abs(self.u)
        phase = np.angle(self.u)

        new_amplitude = amplitude * np.exp(-1.0j * phase)

        if new_field is False:
            self.u = new_amplitude
        else:
            from .scalar_masks_X import Scalar_mask_X
            new = Scalar_mask_X(self.x, self.wavelength)
            new.u = new_amplitude
            return new

    @check_none('u', raise_exception=bool_raise_exception)
    def filter(self, size: float):
        """ Filters the field with a slit of a certain size.

        Args:
            size (float): size of mask for filtering
        """

        from .scalar_masks_X import Scalar_mask_X  # Do not write up

        slit = Scalar_mask_X(self.x, self.wavelength)
        slit.slit(x0=0, size=size)
        self.u = fft_filter(self.u, slit.u)

    @check_none('x', 'u', raise_exception=bool_raise_exception)
    def insert_mask(self, t1, x0_mask1: float, clean: bool = True, kind_position: str = "left"):
        """Insert mask t1 in mask self. It is performed using interpolation.

        Args:
            t1 (Scalar_field_X): field X (shorter)
            x0_mask1 (float): location of starting point at t0 of init point of t1.
            clean (bool): if True remove previous fields, else substitute
            kind_position (str):  'left' 'center'
        """

        # reallocating the mask
        if kind_position == "left":
            t1.x = t1.x - t1.x[0] + x0_mask1
        elif kind_position == "center":
            t1.x = t1.x - (t1.x[0] + t1.x[-1])/2 + x0_mask1

        # interpolation is different for real and imag
        f_interp_real = interp1d(
            t1.x, np.real(t1.u), kind="nearest", bounds_error=False, fill_value=0
        )

        f_interp_imag = interp1d(
            t1.x, np.imag(t1.u), kind="nearest", bounds_error=False, fill_value=0
        )

        # interpolates all the range
        u_new_real = f_interp_real(self.x)
        u_new_imag = f_interp_imag(self.x)
        u_new = u_new_real + 1j * u_new_imag

        if clean is True:
            self.clear_field()
            self.u = u_new
        else:
            i_pos = (self.x > t1.x[0]) * (self.x < t1.x[-1])
            self.u[i_pos] = u_new[i_pos]

    @check_none('x', 'u', raise_exception=bool_raise_exception)
    def pupil(self, x0, radius):
        """Place a pupil in the field.


        Args:
            x0 (float): center of pupil.
            radius (float): radius of pupil.

        """
        xmin = x0 - radius
        xmax = x0 + radius

        pupil = np.zeros_like(self.x)
        ix = (self.x < xmax) & (self.x > xmin)
        pupil[ix] = 1

        self.u = self.u * pupil

    def insert_array_masks(self, t1, x_pos, clean=True, kind_position="left"):
        """Insert several identical masks t1 in self.u according to positions x_pos

        Args:
            t1 (Scalar_field_X): mask to insert.
            x_pos (numpy.array): array with locations.
            clean (bool): elliminate preview fields in self.
            kind_position (str): 'left', 'center': positions are at left or center.
        """

        self.insert_mask(t1, x_pos[0], clean=clean, kind_position=kind_position)

        for xi in x_pos[1:]:
            self.insert_mask(t1, xi, clean=False, kind_position=kind_position)

    def repeat_structure(self, num_repetitions, position="center", new_field=True):
        """Repeat the structure n times.

        Args:
            num_repetitions (int): Number of repetitions of the mask
            position (string or number): 'center', 'previous' or initial position. Initial x
            new_field (bool): If True, a new mask is produced, else, the mask is modified.

        """
        
        u0 = self.u
        x0 = self.x
        wavelength = self.wavelength

        u_new = np.tile(u0, num_repetitions)

        x_min = x0[0]
        x_max = x0[-1]

        x_new = np.linspace(
            num_repetitions * x_min, num_repetitions * x_max, num_repetitions * len(x0)
        )

        if position == "center":
            center_x = (x_new[-1] + x_new[0])/2
            x_new = x_new - center_x
        elif position == "previous":
            x_new = x_new - x_new[0] + x0[0]

        elif isinstance(position, (int, float)):
            x_new = x_new - x_new[0] + position

        if new_field is True:
            t_new = Scalar_field_X(x=x_new, wavelength=wavelength)
            t_new.u = u_new
            return t_new
        else:
            self.u = u_new
            self.x = x_new

    @check_none('x', 'u', raise_exception=bool_raise_exception)
    def fft(self, z: float | None = None, shift: bool = True,
            remove0: bool = False, matrix: bool = False,
            new_field: bool = False, verbose: bool = False):
        """Far field diffraction pattern using Fast Fourier Transform (FFT).

        Args:
            z (float): distance to the observation plane or focal of lens. If None the exit is in radians
            shift (bool): if True, fftshift is performed
            remove0 (bool): if True, central point is removed
            matrix (bool):  if True only matrix is returned.  If False, returns Scalar_field_X
            new_field (bool): if True returns Scalar_field_X, else it puts in self
            verbose (bool): if True, prints info

        Returns:
            (array or Scalar_field_X or None): FFT of the input field
        """
        


        ttf1 = fft(self.u)
        if remove0 is True:
            ttf1[0] = 0

        if shift is True:
            ttf1 = fftshift(ttf1)

        if matrix is True:
            return ttf1

        # x scaling - Infor
        num_x = self.x.size
        delta_x = self.x[1] - self.x[0]
        freq_nyquist_x = 1 / (2 * delta_x)
        kx = np.linspace(-freq_nyquist_x, freq_nyquist_x, num_x) * self.wavelength

        if z in (None, "", [], 0):
            x_new = kx  # exit in angles (degrees)
            if verbose is True:
                print("z={}".format(z))
                print("x0={},x1={}".format(x_new[0], x_new[-1]))
        else:
            x_new = kx * z  # exit distances at a obsrvation plane z
            if verbose is True:
                print("z={}".format(z))
                print("x0={},x1={}".format(x_new[0], x_new[-1]))

        if new_field is True:
            field_output = Scalar_field_X(self.x, self.wavelength)
            field_output.u = ttf1
            field_output.x = x_new
            return field_output
        else:
            self.u = ttf1
            self.x = x_new

    @check_none('x', 'u', raise_exception=bool_raise_exception)
    def ifft(self, z: float | None = None, shift: bool = True,
             remove0: bool = False, matrix: bool = False,
             new_field: bool = False, verbose: bool = False):
        """Inverse Fast Fourier Transform (ifft) of the field.

        Args:
            z (float): distance to the observation plane or focal of lens
            shift (bool): if True, fftshift is performed
            remove0 (bool): if True, central point is removed
            matrix (bool):  if True only matrix is returned. If False, returns Scalar_field_X
            new_field (bool): if True returns Scalar_field_X, else it puts in self
            verbose (bool): if True, prints info

        Returns:
            (array or Scalar_field_X or None): FFT of the input field
        """

        ttf1 = ifft(self.u)
        if remove0 is True:
            ttf1[0] = 0

        if shift is True:
            ttf1 = fftshift(ttf1)

        if matrix is True:
            return ttf1

        # x scaling - Infor
        num_x = self.x.size
        delta_x = self.x[1] - self.x[0]
        freq_nyquist_x = 1 / (2 * delta_x)
        kx = np.linspace(-freq_nyquist_x, freq_nyquist_x, num_x) * self.wavelength

        if z in (None, "", [], 0):
            x_new = kx  # exit in angles (radians)
            if verbose is True:
                print("x0={},x1={}".format(x_new[0], x_new[-1]))
        else:
            x_new = kx * z  # exit distances at a obsrvation plane z
            if verbose is True:
                print("x0={},x1={}".format(x_new[0], x_new[-1]))
        if new_field is True:
            field_output = Scalar_field_X(self.x, self.wavelength)
            field_output.u = ttf1
            if verbose is True:
                print("x0={},x1={}".format(x_new[0], x_new[-1]))
            field_output.x = x_new
            return field_output
        else:
            self.u = ttf1
            self.x = x_new
            if verbose is True:
                print("x0={},x1={}".format(x_new[0], x_new[-1]))

    @check_none('x', 'u', raise_exception=bool_raise_exception)
    def _RS_(self, z: float, n: float, matrix: bool = False,
             new_field: bool = True, fast: bool = False, kind: str = "z",
             xout: None | NDArrayFloat = None, verbose: bool = True):
        r"""Fast-Fourier-Transform  method for numerical integration of diffraction Rayleigh-Sommerfeld formula. `Thin Element Approximation` is considered for determining the field just after the mask:

        :math:`\mathbf{E}_{0}(\zeta,\eta)=t(\zeta,\eta)\mathbf{E}_{inc}(\zeta,\eta)`

        The near field approach is performed according to  :math:`\mathbf{E}(x,y,z) = \frac{1}{i\lambda z}e^{i k z}\iint\mathbf{E}_{0}(\zeta,\eta)e^{i\frac{k}{2z}\left[\left(x-\zeta\right)^{2}+\left(y-\eta\right)^{2}\right]}d\zeta d\eta`

        If we have a field of size N*M, the result of propagation is also a field N*M. Nevertheless, there is a parameter `amplification` which allows us to determine the field in greater observation planes (jN)x(jM).

        One adventage of this approach is that it returns a quality parameter: if self.quality>1, propagation is right.

        Args:
            z (float): distance to observation plane. I z<0 inverse propagation is executed
            n (float): refractive index
            matrix (bool): if True returns a matrix with result. It is much faster than new_field=True
            new_field (bool): if False the computation goes to self.u. If True a new instance is produced
            fast (bool): Instead of using Hankle function for RS kernel uses exponential
            xout (numpy.array): for amplification
            verbose (bool): if True it writes to shell

        Returns:
            (Scalar_field_X or None): if New_field is True Scalar_field_X, else None

        References:
                F. Shen and A. Wang, “Fast-Fourier-transform based numerical integration method for the Rayleigh-Sommerfeld diffraction formula,” Appl. Opt., vol. 45, no. 6, pp. 1102–1110, 2006.
        """

        if xout is None:
            xout = self.x[0]

        xout = self.x + xout - self.x[0]

        nx = len(xout)
        dx = xout[1] - xout[0]

        # parametro de quality
        dr_real = dx
        rmax = xout.max()
        dr_ideal = (
            sqrt(
                (self.wavelength / n) ** 2 +
                rmax**2 +
                2 * (self.wavelength / n) * sqrt(rmax**2 + z**2)
            ) -
            rmax
        )
        self.quality = dr_ideal / dr_real/2

        if verbose is True:
            if self.quality.min() > 1:
                print("Good result: factor {:2.2f}".format(self.quality))
            else:
                print("- Needs denser sampling: factor {:2.2f}".format(self.quality))

        precise = False

        if precise:
            # matrix W para integracion simpson
            a = [2, 4]
            num_rep = int(round((nx)/2) - 1)

            b = array(a * num_rep)
            W = concatenate(((1,), b, (2, 1))) / 3.0

            if float(nx)/2 == round(nx/2):  # es par
                i_central = num_rep + 1
                W = concatenate((W[:i_central], W[i_central + 1:]))
        else:
            W = 1

        # field
        U = zeros(2 * nx - 1, dtype=complex)
        U[0:nx] = array(W * self.u)

        xext = self.x[0] - xout[::-1]  # da la vuelta
        xext = xext[0:-1]
        xext = concatenate((xext, self.x - xout[0]))
        if z > 0:
            H = kernelRS(xext, self.wavelength, z, n, kind=kind, fast=fast)
        else:
            H = kernelRSinverse(xext, self.wavelength, z, n, kind=kind, fast=fast)

        # calculo de la transformada de Fourier
        S = ifft(fft(U) * fft(H)) * dx
        Usalida = S[nx - 1:]

        # los calculos se pueden dejar en la instancia o crear un new field
        if matrix is True:
            return Usalida

        if new_field is True:
            field_output = Scalar_field_X(self.x, self.wavelength)
            field_output.u = Usalida
            field_output.quality = self.quality
            return field_output
        else:
            self.u = Usalida

    @check_none('x', 'u', raise_exception=bool_raise_exception)
    def RS(self, z: float, amplification: int = 1, n: float = 1.,
           new_field: bool = True, matrix: bool = False, xout: None | NDArrayFloat = None,
           fast: bool = False, kind: str = "z", verbose: bool = True):
        """Fast-Fourier-Transform  method for numerical integration of diffraction Rayleigh-Sommerfeld formula. Is we have a field of size N*M, the result of propagation is also a field N*M. Nevertheless, there is a parameter `amplification` which allows us to determine the field in greater observation planes (jN)x(jM).

        Args:
            z (float): Distance to observation plane. if z<0 inverse propagation is executed
            n (float): refractive index
            new_field (bool): if False the computation goes to self.u, if True a new instance is produced
            matrix (bool): If True, the result of the function is a numpy.array
            fast (bool): Instead of using Hankle function for RS kernel uses expotential
            amplification (int): number of frames in x direction
            kind (str): 'z'. In some circunstamces the function is used for other integrals
            verbose (bool): if True it writes to shell

        Returns:
            If New_field is True:  Scalar_field_X,
            If matrix is True: numpy.array()
            Else: None
            
        Info:
            This approach a quality parameter: If self.quality>1, propagation is right.
        """


        width_x = self.x[-1] - self.x[0]
        num_pixels = len(self.x)

        positions_x = (
            -amplification * width_x/2 + array(list(range(amplification))) * width_x
        )

        x0 = linspace(
            -amplification * width_x/2,
            amplification * width_x/2,
            num_pixels * amplification,
        )

        if xout is not None:
            positions_x = positions_x + xout
            x0 = x0 + xout - width_x/2

        u_field = np.zeros_like(x0, dtype=complex)
        qualities = np.zeros((amplification))
        for i, xi in zip(list(range(len(positions_x))), np.flipud(positions_x)):
            u3 = self._RS_(
                z=z,
                n=n,
                matrix=False,
                new_field=True,
                fast=fast,
                kind=kind,
                xout=xi,
                verbose=verbose,
            )
            xshape = slice(i * num_pixels, (i + 1) * num_pixels)
            u_field[xshape] = u3.u
            qualities[i] = u3.quality

        if matrix is True:
            return u_field

        if new_field is True:
            field_output = Scalar_field_X(x=x0, wavelength=self.wavelength)
            field_output.u = u_field
            field_output.quality = qualities
            return field_output
        else:
            self.x = x0
            self.u = u_field
            self.quality = qualities

    @check_none('x')
    def CZT(self, z: float | NDArrayFloat, xout: float | NDArrayFloat | None,
            verbose: float = False):
        """Chirped z-transform.
        
        The code for this algoritm is based on "Hu, Yanlei, et al. "Efficient full-path optical calculation of scalar and 
        vector diffraction using the Bluestein method." Light: Science & Applications 9.1 (2020): 119."
        
        However, the convolution Kernel has been changed to Rayleigh-Sommerfeld.

        Args:
            z (float or np.array): diffraction distance
            xout (float or np.array): x array with positions of the output plane
            verbose (bool): If True it prints information.


        Returns:
            u_out: Complex amplitude of the outgoing light beam

        References:
            [Light: Science and Applications, 9(1), (2020)]
        """
        
        if xout is None:
            xout = self.x

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

        dx = self.x[1] - self.x[0]

        if num_z == 1:
            delta_out = np.zeros(2)
            if num_x > 1:
                delta_out[0] = (xend - xstart) / (num_x - 1)

            # calculating scalar diffraction below
            R = np.sqrt(xout**2 + z**2)
            if z>=0:
                F0 = (0.5j * k * z / R) * hankel1(1, k * R)
            else:
                F0 = (-0.5j * k * z / R) * hankel1(1, k * R).conjugate()

            R = np.sqrt(self.x**2 + z**2)
            if z>=0:
                F = (0.5j * k * z / R) * hankel1(1, k * R)
            else:
                F = (-0.5j * k * z / R) * hankel1(1, k * R).conjugate()
                
            u0 = self.u * F

            fs = self.wavelength * z / dx  # dimension of the imaging plane
            fx1 = xstart + fs/2
            fx2 = xend + fs/2
            u0 = Bluestein_dft_x(u0, fx1, fx2, fs, num_x)

            k_factor = np.sqrt(np.abs(z) * self.wavelength) * dx

            u0 = F0 * u0 * k_factor
            # obtain the complex amplitude of the outgoing light beam

            if num_x == 1:
                return 1j * u0
            else:
                u_out = Scalar_field_X(xout, self.wavelength)
                u_out.u = 1j * u0

        else:
            u_zs = np.zeros((len(z), num_x), dtype=complex)
            num_z = len(z)
            for i, z_now in enumerate(z):
                if verbose is True:
                    print("{}/{}".format(i, num_z), sep="\r", end="\r")
                delta_out = np.zeros(2)
                if num_x > 1:
                    delta_out[0] = (xend - xstart) / (num_x - 1)

                # calculating scalar diffraction below
                
                R = np.sqrt(xout**2 + z_now**2)
                if z.all()>=0:
                    F0 = (0.5j * k * z_now / R) * hankel1(1, k * R)
                else:
                    F0 = (-0.5j * k * z_now / R) * hankel1(1, k * R).conjugate()

                R = np.sqrt(self.x**2 + z_now**2)
                if z.all()>=0:
                    F = (0.5j * k * z_now / R) * hankel1(1, k * R)
                else:
                    F = (-0.5j * k * z_now / R) * hankel1(1, k * R).conjugate()

                u0 = self.u * F

                fs = self.wavelength * z_now / dx  # dimension of the imaging plane
                fx1 = xstart + fs/2
                fx2 = xend + fs/2
                u0 = Bluestein_dft_x(u0, fx1, fx2, fs, num_x)

                u0 = F0 * u0  # obtain the complex amplitude of the outgoing light beam

                k_factor = np.sqrt(np.abs(z_now) * self.wavelength) * dx

                u_zs[i, :] = u0 * k_factor

            if num_x == 1:
                from diffractio.scalar_fields_Z import Scalar_field_Z

                u_out = Scalar_field_Z(z=z, wavelength=self.wavelength)
                u_out.u = 1j * u_zs
                return u_out
            else:
                from diffractio.scalar_fields_XZ import Scalar_field_XZ

                u_out = Scalar_field_XZ(xout, z, self.wavelength)
                u_out.u = 1j * u_zs

        return u_out


    def CZT_points(self, z: NDArrayFloat, x: NDArrayFloat,  verbose: float = False, has_draw: bool = False) -> NDArrayFloat:
        """Chirped z-transform for arrays of points z = f(x)
        
        The code for this algoritm is based on "Hu, Yanlei, et al. "Efficient full-path optical calculation of scalar and 
        vector diffraction using the Bluestein method." Light: Science & Applications 9.1 (2020): 119."
        
        However, the convolution Kernel has been changed to Rayleigh-Sommerfeld.

        Args:
            z (float or np.array): diffraction distance
            x (float or np.array): x array with positions of the output plane
            verbose (bool): If True it prints information.
            has_draw (bool): If True it draws the far field pattern.

        Returns:
            u_out: Complex amplitude of the outgoing light beam

        References:
            [Light: Science and Applications, 9(1), (2020)]
        """

        if len(z) != len(x):
            raise ValueError(f"z and x arrays must have the same length. Got z: {len(z)}, x: {len(x)}")
    
        
        k = 2 * np.pi / self.wavelength

        num_z = len(z)

        u_out = np.zeros_like(x, dtype=complex)

        dx = self.x[1] - self.x[0]

        for i in range(num_z):
            if verbose is True:
                print("{}/{}".format(i, num_z), sep="\r", end="\r")
                
            x_now = x[i]
            z_now = z[i]

            xstart = x_now
            xend = x_now

            # calculating scalar diffraction below
            R = np.sqrt(x_now**2 + z_now**2)
            if z_now>=0:
                F0 = (0.5j * k * z_now / R) * hankel1(1, k * R)
            else:
                F0 = (-0.5j * k * z_now / R) * hankel1(1, k * R).conjugate()


            R = np.sqrt(self.x**2 + z_now**2)
            if z_now>=0:
                F = (0.5j * k * z_now / R) * hankel1(1, k * R)
            else:
                F = (-0.5j * k * z_now / R) * hankel1(1, k * R).conjugate()
                
            u0 = self.u * F

            fs = self.wavelength * z_now / dx
            fx1 = xstart + fs/2
            fx2 = xend + fs/2

            u0 = Bluestein_dft_x(u0, fx1, fx2, fs, 1)
            k_factor = np.sqrt(np.abs(z_now) * self.wavelength) * dx
            u0 = F0 * u0 * k_factor

            u_out[i] = 1j * u0 

        if has_draw:

            I_far = np.abs(u_out)**2
            I_far /= np.max(I_far)

            plt.figure()
            plt.semilogy(x/mm, I_far, 'b')
            plt.grid('on')
            plt.xlabel(r' $x (mm)$')
            plt.ylabel(r'$\log_{10}(I/I_{max})$')
            plt.title('Far field pattern')

        return u_out



    def CZT_angular(self,  theta: NDArrayFloat | None, radius_obs: float, verbose: float = False, has_draw: bool | int = False):
        """Chirped z-transform for arrays of points R = f(theta)
        
        The code for this algoritm is based on "Hu, Yanlei, et al. "Efficient full-path optical calculation of scalar and 
        vector diffraction using the Bluestein method." Light: Science & Applications 9.1 (2020)."
        

        Args:
            radius_obs (float): radius of the observation circle
            theta (float or np.array): theta array with observation angles
            verbose (bool): If True it prints information.
            has_draw (bool or int): If True it draws the intensity in polar and cartesian coordinates. If 1, only cartesian. If 2, only polar.
        Returns:
            u_out: Complex amplitude of the outgoing light beam
        """

        x = radius_obs * np.sin(theta)
        z = radius_obs * np.cos(theta)

        u_theta = self.CZT_points(z, x, verbose=verbose)

        # Calculate intensity for drawing
        I_far = np.abs(u_theta)**2
        I_far /= I_far.max()

        if has_draw in (True, 1, 'all'):
            plt.figure()
            plt.semilogy(theta/degrees, I_far, 'b')
            plt.grid('on')
            plt.xlabel(r' $\theta (^\circ)$')
            plt.ylabel(r'$\log_{10}(I/I_{max})$')
            plt.xlim(theta[0]/degrees, theta[-1]/degrees)
            plt.ylim(1e-6, 1)
            plt.tight_layout()


            
        if has_draw in (2, 'all'):

            
            # Enhanced polar plot
            fig, ax = plt.subplots( subplot_kw=dict(projection='polar'))
            
            # Plot the data
            ax.plot(theta, np.log10(I_far), 'b-', linewidth=1, label=r'$\log_{10}(I/I_{max})$')
            
            # Grid settings
            ax.grid(True, alpha=1)
            ax.set_rgrids(np.arange(-6, 1, 2), 
                        labels=[f'{i} dB' for i in range(-6, 1, 2)], 
                        angle=45, fontsize=10)
            
            # # Angular grid with degree labels
            # theta_deg_ticks = np.arange(theta[0]/degrees, theta[-1]/degrees+0.1, 15/degrees)
            # ax.set_thetagrids(theta_deg_ticks, 
            #                 labels=[f'{int(t)}°' for t in theta_deg_ticks], 
            #                 fontsize=10)
            
            # # Set limits
            ax.set_ylim(-6, 0)
            ax.set_xlim(-90*degrees, 90*degrees)
            
            # Title and labels
            # ax.set_title('Far field pattern - Polar View', fontsize=10)
                        
            plt.tight_layout()

        return u_theta   


    @check_none('x')
    def WPM(self, fn, zs: NDArrayFloat, num_sampling: tuple[int] | None = None,
            ROI: tuple[NDArray] | None = None, x_pos: float | None = None,
            z_pos: float | None = None, get_u_max: bool = False,
            has_edges: bool = True, pow_edge: int = 80, verbose: bool = False):
        """WPM method used for very dense sampling. It does not storages the intensity distribution at propagation, but only selected areas. The areas to be stored are:
            - global view with a desired sampling given by num_sampling.
            - intensity at the last plane.
            - Intensity at plane with maximum intensity (focus of a lens, for example)
            - Region of interest given by ROI.

        Args:
            fn (function): Function that returns the refractive index at a plane in a numpy.array
            zs (np.array): linspace with positions z where to evaluate the field.
            num_sampling (int, int) or None: If None, it does not storage intermediate data. Otherwise, it stores a small XZ matrix which size in num_sampling.
            ROI (np.array, np.array) or None: x and z arrays with positions and sampling to store. It is used when we need to store with a high density a small area of the total field.
            x_pos (float or None): If not None, a high density longitudinal array with the field at x_pos position is stored.
            z_pos (float or None): If not None, a high density transversal array with the field at z_pos is stored.
            get_u_max (bool): If True, returns field at z position with maximum intensity.
            has_edges (bool): If True absorbing edges are used.
            pow_edge (float): If has_edges, power of the supergaussian.
            verbose (bool): If True prints information.

        Returns:
            u_iter (Scalar_field_X): final field
            u_out_gv (Scalar_field_XZ): global view
            u_out_roi (Scalar_field_XZ): field at ROI positions
            u_axis_x (Scalar_field_X): field at position z_pos
            u_axis_z (Scalar_field_Z): field at position x_pos
            u_max (Scalar_field_X): field at position with maximum intensity
            z_max (float): position with maximum intensity

        References:

            1. M. W. Fertig and K.-H. Brenner, “Vector wave propagation method,” J. Opt. Soc. Am. A, vol. 27, no. 4, p. 709, 2010.

            2. S. Schmidt et al., “Wave-optical modeling beyond the thin-element-approximation,” Opt. Express, vol. 24, no. 26, p. 30188, 2016.

        """
        from diffractio.scalar_masks_XZ import Scalar_mask_XZ
        

        
        k0 = 2 * np.pi / self.wavelength
        dx = self.x[1] - self.x[0]
        dz = zs[1] - zs[0]

        kx = get_k(self.x, flavour="+")
        k_perp2 = kx**2

        if has_edges is False:
            has_filter = np.zeros_like(zs)
        elif isinstance(has_edges, int):
            has_filter = np.ones_like(zs)
        else:
            has_filter = has_edges

        width_edge = 0.95 * (self.x[-1] - self.x[0])/2
        x_center = (self.x[-1] + self.x[0])/2

        filter_function = np.exp(
            -((np.abs(self.x - x_center) / width_edge) ** pow_edge)
        )

        u_iter = self.duplicate()

        # STORING field at maximum intensity
        if get_u_max is True:
            u_max = self.duplicate()
            I_max = 0.0
            z_max = 0
        else:
            u_max = None
            z_max = None

        # Storing intensities at axis (x=x_pos)
        if x_pos is not None:
            from diffractio.scalar_fields_Z import Scalar_field_Z

            u_axis_x = Scalar_field_Z(zs, self.wavelength)
            index_x_axis, _, _ = nearest(self.x, x_pos)
        else:
            u_axis_x = None

            # Storing fields at axis (x=x_pos)
        if z_pos is not None:
            u_axis_z = self.duplicate()
            index_zpos, _, _ = nearest(zs, z_pos)
        else:
            u_axis_z = None

        # STORING global view matrices
        if num_sampling is not None:
            len_x, len_z = num_sampling
            xout_gv = np.linspace(self.x[0], self.x[-1], len_x)
            zout_gv = np.linspace(zs[0], zs[-1], len_z)
            u_out_gv = Scalar_mask_XZ(
                xout_gv, zout_gv, self.wavelength, info="from WPM_no_storage_1D"
            )
            indexes_x_gv, _, _ = nearest2(self.x, xout_gv)
            indexes_z_gv, _, _ = nearest2(zs, zout_gv)
            u_out_gv.n = fn(xout_gv, zout_gv, self.wavelength)
        else:
            u_out_gv = None

        # STORING ROI
        if ROI is not None:
            xout_roi, zout_roi = ROI
            u_out_roi = Scalar_mask_XZ(
                xout_roi, zout_roi, self.wavelength, info="from WPM_no_storage_1D"
            )
            indexes_x_roi, _, _ = nearest2(self.x, xout_roi)
            indexes_z_roi, _, _ = nearest2(zs, zout_roi)
            u_out_roi.n = fn(xout_roi, zout_roi, self.wavelength)
        else:
            u_out_roi = None

        t1 = time.time()
        num_steps = len(zs)
        iz_out_gv = 0
        iz_out_roi = 0

        for j in range(1, num_steps):
            if has_filter[j] == 0:
                filter_edge = 1
            else:
                filter_edge = filter_function

            refractive_index = fn(
                self.x,
                np.array(
                    [
                        zs[j - 1],
                    ]
                ),
                self.wavelength,
            )
            u_iter.u = (
                WPM_schmidt_kernel(u_iter.u, refractive_index, k0, k_perp2, dz) *
                filter_edge
            )

            if x_pos is not None:
                u_axis_x.u[j] = u_iter.u[index_x_axis]

            if z_pos is not None:
                if j == index_zpos:
                    u_axis_z.u = u_iter.u

            if num_sampling is not None:
                if j in indexes_z_gv:
                    u_out_gv.u[iz_out_gv,:] = u_iter.u[indexes_x_gv]
                    iz_out_gv = iz_out_gv + 1

            if ROI is not None:
                if j in indexes_z_roi:
                    u_out_roi.u[iz_out_roi,:] = u_iter.u[indexes_x_roi]
                    iz_out_roi = iz_out_roi + 1

            if get_u_max is True:
                current_intensity = np.max(np.abs(u_iter.u) ** 2)
                if current_intensity > I_max:
                    I_max = u_iter.intensity().max()
                    u_max.u = u_iter.u
                    z_max = zs[j]

            if verbose is True:
                print("{}/{}".format(j, num_steps), sep="\r", end="\r")

        t2 = time.time()
        if verbose is True:
            print(
                "Time = {:2.2f} s, time/loop = {:2.4} ms".format(
                    t2 - t1, (t2 - t1) / len(zs) * 1000
                )
            )

        return u_iter, u_out_gv, u_out_roi, u_axis_x, u_axis_z, u_max, z_max

    def to_far_field(self, angles: np.array,  z_obs: float | None = None,  has_draw: bool = True, has_logarithm=True, verbose: bool = True, **kwargs):
        r"""
        Compute the far field of a source or mask. 
        

        The function calculates the far field (in angles) of a mask or source using the CZT function. 
        In fact, the CZT function is used to compute the far field, and the x positions are computed considering far field.

        Args:
            angles (np.array): Angles to compute the far field.
            z_obs (float | None, optional): Distance where the far field is computed. If None, the distance is given by $\pi size^2 /\wavelength$. Defaults to None.
            has_draw (bool, optional): Draws the far field diffraction pattern, in angles. Defaults to False.
            has_logarithm (bool, optional): If True, the drawing is made in logarithm scale. Defaults to False.
            verbose (bool, optional): Prints information. Defaults to True.

        Returns:
            I_far (np.array): Intensity at the far field
        """

        if z_obs is None: 
            size_frame = self.x[-1]-self.x[0]
            z_obs = np.pi*size_frame**2/self.wavelength

        xout = np.sin(angles)*z_obs

        u_far = self.CZT(z=z_obs, xout=xout)
        I_far = u_far.intensity()
        
        if has_logarithm:
            function = plt.semilogy
        else:
            function = plt.plot

        if has_draw:
            plt.figure(**kwargs)
            function(angles/degrees, I_far, 'k')
            plt.ylim(0,I_far.max())
            plt.xlim(angles[0]/degrees, angles[-1]/degrees)
            plt.xlabel('angles (degrees)')
            plt.grid('on')
            
        if verbose:
            print("z_obs = {:2.2f} mm".format(z_obs/mm))
            
        return I_far



    @check_none('x', 'u', raise_exception=bool_raise_exception)
    def MTF(self,  frequencies: np.ndarray, incoherent: bool = True, has_draw: bool = False) -> tuple[np.ndarray, np.ndarray]:
        
        """Direct DFT evaluation at arbitrary frequencies. Exact but O(N*M).
        
        Args:
            frequencies (np.ndarray): frequencies where MTF is evaluated (in cycles/mm)
            incoherent (bool): If True, MTF is computed for incoherent light (using intensity). If False, coherent light (using field).
            has_draw (bool): If True, draws the MTF.

        Returns:
            fd (np.ndarray): frequencies (cycles/mm)
            mtf (np.ndarray): MTF values

        """
            
        if incoherent:
            s = np.abs(self.u)**2
        else:
            s = self.u

        
        # exponent matrix: shape (n_freqs, N)
        ex = np.exp(-2j * np.pi * np.outer(frequencies/1000, self.x))
        Svals = ex.dot(s)  
        norm = np.abs(Svals).max() if np.abs(Svals).max() > 0 else 1.0
        otf = Svals / norm
        mtf = np.abs(otf)


        if has_draw:
            plt.figure(); 

            plt.plot(frequencies, mtf, 'b', label='MTF')
            plt.ylabel('MTF')
            plt.xlabel('Frequency (cycles/mm)')
            plt.xlim(0, frequencies[-1])
            plt.ylim(-0.01, 1.01)
            plt.grid()
            plt.legend()
        
        return mtf


    @check_none('u', raise_exception=bool_raise_exception)
    def intensity(self):
        """Intensity.

        Returns:
            (numpy.array): Intensity
        """

        intensity = np.abs(self.u) ** 2
        return intensity

    def average_intensity(self, verbose: bool = False):
        """Returns the average intensity as: (np.abs(self.u)**2).sum() / num_data

        Args:
            verbose (bool): If True it prints the value of the average intensity.

        Returns:
            (float): average intensity.
        """
        average_intensity = (np.abs(self.u) ** 2).mean()
        if verbose is True:
            print("average intensity={} W/m").format(average_intensity)

        return average_intensity


    @check_none('u', raise_exception=bool_raise_exception)
    def get_edges(self, kind_transition: str = "amplitude", min_step: int = 0,
                  verbose: bool = False, filename: str = ""):
        """Determine locations of edges for a binary mask.

        Args:
            kind_transition:'amplitude' 'phase'
                if we see the amplitude or phase of the field
            min_step: minimum step for consider a transition

        Returns:
            type_transition: array with +1, -1 with rasing or falling edges
            pos_transition: positions x of transitions
            raising: positions of raising
            falling: positions of falling
        """

        pos_transitions, type_transitions, raising, falling = get_edges(
            self.x, self.u, kind_transition, min_step, verbose, filename
        )
        return pos_transitions, type_transitions, raising, falling


    @check_none('x', raise_exception=bool_raise_exception)
    def get_RS_minimum_z(self, n: float = 1., quality: int = 1, verbose: bool = True):
        """Determines the minimum available distance for RS algorithm. If higher or lower quality parameters is required you can add as a parameter

        Args:
            n (float): refractive index of the surrounding medium.
            quality (int, optional): quality. Defaults to 1.
            verbose (bool, optional): prints info. Defaults to True.

        Returns:
            z_min (float): z_min for quality_factor>quality
        """

        range_x = self.x[-1] - self.x[0]
        num_x = len(self.x)

        dx = range_x / num_x
        dr_real = dx
        rmax = range_x

        factor = (
            ((quality * dr_real + rmax) ** 2 - (self.wavelength / n) ** 2 - rmax**2) /
            2 *
            n /
            self.wavelength
        ) ** 2 - rmax**2

        if factor > 0:
            z_min = np.sqrt(factor)
        else:
            z_min = 0

        if verbose:
            if z_min > 1000:
                print("z min = {:2.2f} mm".format(z_min / mm))
            else:
                print("z min = {:2.2f} um".format(z_min))

        return z_min

    @check_none('x', 'u', raise_exception=bool_raise_exception)
    def draw(self, kind: Draw_X_Options = "intensity", logarithm: float = 0.,
        normalize: bool = False, cut_value: float | None = None, filename: str = "",
        scale: str = ""):
        """Draws X field. There are several data from the field that are extracted, depending of 'kind' parameter.

        Args:
            kind (str): type of drawing: 'amplitude', 'intensity', 'field', 'phase', 'fill', 'fft'
            logarithm (float): If >0, intensity is scaled in logarithm
            normalize (bool): If True, max(intensity)=1
            cut_value (float): If not None, cuts the maximum intensity to this value
            filename (str): if not '' stores drawing in file,
            scale (str): '', 'scaled', 'equal', scales the XY drawing
        """

        if self.x is None:
            print("could not draw file: self.x=None")
            return

        amplitude, intensity, phase = field_parameters(self.u)
        phase[intensity < 0.001 * intensity.max()] = 0

        plt.figure()

        if kind == "intensity":
            y = intensity
        elif kind == "phase":
            y = phase
        elif kind in ("amplitude", "fft", "fill", "field"):
            y = amplitude

        if kind in ("intensity", "amplitude", "fft", "fill", "field"):
            y = normalize_draw(y, logarithm, normalize, cut_value)

        if kind == "field":
            plt.subplot(211)
            plt.plot(self.x, y, "k")
            plt.xlabel(r"$x\,(\mu m)$")
            plt.ylabel(r"$A\,(arb.u.)$")
            plt.xlim(left=self.x[0], right=self.x[-1])
            plt.ylim(bottom=0)

            plt.subplot(212)
            plt.plot(self.x, phase, "k")
            plt.xlabel(r"$x\,(\mu m)$")
            plt.ylabel(r"$phase\,(radians)$")
            plt.xlim(left=self.x[0], right=self.x[-1])

        elif kind in ("amplitude", "intensity", "phase"):
            plt.plot(self.x, y, "k")
            plt.xlabel(r"$x\,(\mu m)$")
            plt.ylabel(kind)
            plt.xlim(left=self.x[0], right=self.x[-1])

        elif kind == "fft":
            plt.plot(self.x/degrees, y, "k")
            plt.xlim(left=self.x[0]/degrees, right=self.x[-1]/degrees)
            plt.xlabel(r"$\phi\,(degrees)$")
            plt.ylabel(kind)

        elif kind == "fill":
            # this is for binary maks, as gratings and I0s.
            plt.fill_between(self.x, 0, amplitude)
            plt.xlabel(r"$x\,(\mu m)$")
            plt.ylabel(kind)
            plt.xlim(left=self.x[0], right=self.x[-1])

        if scale != "":
            plt.axis(scale)

        if filename != "":
            plt.savefig(filename, dpi=100, bbox_inches="tight", pad_inches=0.1)

        if kind == "intensity":
            plt.ylim(bottom=0)
        elif kind == "phase":
            plt.ylim(-pi, pi)


def kernelRS(x: NDArrayFloat, wavelength: float, z: float,
             n: float = 1., kind: str = "z", fast: bool = False):
    r"""Kernel for RS propagation. It uses the hankel tansform.

    There is a 'fast' version based on :math:`hk_1 = \sqrt{2/(\pi \, k \, R)}  e^{i  (k \, R - 3  \pi / 4)}` which approximates the result.

    Args:
        x (numpy.array): positions x
        wavelength (float): wavelength of incident fields
        z (float): distance for propagation
        n (float): refractive index of background
        kind (str): 'z', 'x', '0': for simplifying vector propagation
        fast (bool): If True  The approximation is much faster. According to https://dlmf.nist.gov/10.2#E5

    Returns:
        (complex array): kernel

    References:
        - https://dlmf.nist.gov/10.2#E5
        - F. Shen and A. Wang, “Fast-Fourier-transform based numerical integration method for the Rayleigh-Sommerfeld diffraction formula,” Appl. Opt., vol. 45, no. 6, pp. 1102–1110, 2006.
    """
    k = 2 * pi * n / wavelength
    R = sqrt(x**2 + z**2)

    if fast is False:
        hk1 = hankel1(1, k * R)
    elif fast is True:
        hk1 = sqrt(2 / (pi * k * R)) * exp(1.0j * (k * R - 3 * pi / 4))

    if kind == "z":
        return (0.5j * k * z / R) * hk1
    elif kind == "x":
        return (0.5j * k * x / R) * hk1
    elif kind == "0":
        return (0.5j * k) * hk1


def kernelRSinverse(x: NDArrayFloat, wavelength: float, z: float,
                    n: float = 1., kind: str = "z", fast: bool = False):
    """Kernel for inverse RS propagation. See also kernelRS

    Args:
        x (numpy.array): positions x
        wavelength (float): wavelength of incident fields
        z (float): distance for propagation
        n (float): refractive index of background
        kind (str): 'z', 'x', '0': for simplifying vector propagation
        fast (bool): If True  The approximation is much faster. According to https://dlmf.nist.gov/10.2#E5

    Returns:
        complex array: kernel
    """
    k = 2 * pi * n / wavelength
    R = sqrt(x**2 + z**2)

    if fast is False:
        hk1 = hankel1(1, k * R).conjugate()
    elif fast is True:
        hk1 = sqrt(2 / (pi * k * R)) * exp(-1.0j * (k * R - 3 * pi / 4))

    if kind == "z":
        return (-0.5j * k * z / R) * hk1
    elif kind == "x":
        return (-0.5j * k * x / R) * hk1
    elif kind == "0":
        return (-0.5j * k) * hk1


def kernelFresnel(x: NDArrayFloat, wavelength: float, z: float, n: float = 1.):
    """
    Kernel for Fresnel propagation.

    Args:
        x(numpy.np.array): positions x
        wavelength(float): wavelength of incident fields
        z(float): distance for propagation
        n(float): refractive index of background

    Returns:
        complex np.array: kernel
    """

    k = 2 * np.pi * n / wavelength
    return np.exp(1.j * k * (z + x**2 / (2 * z))) / (1.j * wavelength * z)




def PWD_kernel(u: NDArrayComplex, n: NDArrayComplex, k0: float, k_perp2: NDArray[Any], dz: float):
    """
    Step for scalar (TE) Plane wave decomposition (PWD) algorithm.

    Args:
        u (np.array): fields
        n (np.array): refractive index
        k0 (float): wavenumber
        k_perp2 (np.array): transversal k**2
        dz (float): increment in distances

    Returns:
        numpy.array(): Field at at distance dz from the incident field

    References:
        1. Schmidt, S. et al. "Wave-optical modeling beyond the thin-element-approximation". 'Opt. Express' 24, 30188 (2016).
    """

    Ek = fftshift(fft(u))
    H = np.exp(1j * dz * csqrt(n**2 * k0**2 - k_perp2))

    return ifft(fftshift(H * Ek))


def WPM_schmidt_kernel(u, n: NDArrayComplex, k0: float, k_perp2: NDArrayComplex, dz: float):
    """
    Kernel for fast propagation of WPM method

    Args:
        u (np.array): fields
        n (np.array): refractive index
        k0 (float): wavenumber
        k_perp2 (np.array): transversal k**2
        dz (float): increment in distances

    References:

        1. M. W. Fertig and K.-H. Brenner, “Vector wave propagation method,” J. Opt. Soc. Am. A, vol. 27, no. 4, p. 709, 2010.

        2. S. Schmidt et al., “Wave-optical modeling beyond the thin-element-approximation,” Opt. Express, vol. 24, no. 26, p. 30188, 2016.
    """
    refractive_indexes = np.unique(n)

    u_final = np.zeros_like(u, dtype=complex)
    for m, n_m in enumerate(refractive_indexes):
        # print (m, n_m)
        u_temp = PWD_kernel(u, n_m, k0, k_perp2, dz)
        Imz = n == n_m
        u_final = u_final + Imz * u_temp

    return u_final


def polychromatic_multiprocessing(
        function_process,
        wavelengths: NDArrayFloat,
        spectrum: NDArrayFloat,
        num_processors: int = num_max_processors,
        verbose: bool = False):
    """
    It performs an analysis of polychromatic light. It needs a function with only one input parameter: wavelength.
    It determines the intensity for each wavelength and the final results is the summation of the intensities.

    Args:
        function_process (function): function with accepts params as input Args:
        wavelengths (array): wavelengths in the spectrum
        spectrum (array): weights for the spectrum. if None: uniform spectrum, else array with the same dimension as wavelengths
        num_processors (int): number of processors for the computation
        verbose (bool): if True send information to shell

    Returns:
        intensity (array, complex): intensity = intensity + spectrum[i] * np.abs(u_s[i].u)**2
        u_s (Scalar_field_X): fields for each wavelength
        time_proc (float): time interval in the processing
    """

    if not isinstance(spectrum, np.ndarray):
        spectrum = np.ones_like(wavelengths)

    if type(wavelengths) in (list, np.ndarray):
        u_s, time_proc = execute_multiprocessing(
            function_process, wavelengths, num_processors, verbose
        )
        print(len(u_s))
        intensity = np.zeros_like(u_s[0].u, dtype=float)
        for i in range(len(wavelengths)):
            intensity = intensity + spectrum[i] * np.abs(u_s[i].u) ** 2
        intensity = intensity / spectrum.sum()
    else:
        time1 = time.time()
        u_s = function_process(wavelengths)
        time2 = time.time()
        intensity = np.abs(u_s.u) ** 2
        time_proc = time2 - time1

    return intensity, u_s, time_proc


def extended_source_multiprocessing(
        function_process, x0s: NDArrayFloat,
        num_processors: int = num_max_processors,
        verbose: bool = False):
    """
    It performs an analysis of extendes source light. It needs a function with only an input parameter, that is x0s positions of sources. It determines the intensity for each wavelength and it is added.

    Args:
        function_process (function): function with accepts params as input Args:
        x0s (array): positions of sources
        num_processors (int): number of processors for the computation
        verbose (bool): if True send information to shell

    Returns:
        - intensity (array, complex): intensity = intensity + spectrum[i] * np.abs(u_s[i].u)**2
        - u_s (Scalar_field_X): fields for each wavelength
        - time_proc (float): time interval in the processing
    """

    if type(x0s) in (list, np.ndarray):
        u_s, time_proc = execute_multiprocessing(
            function_process, x0s, num_processors, verbose
        )
        intensity = np.zeros_like(u_s[0].u, dtype=float)
        for i in range(len(x0s)):
            intensity = intensity + np.abs(u_s[i].u) ** 2
        intensity = intensity / len(x0s)
    else:
        time1 = time.time()
        u_s = function_process(x0s)
        time2 = time.time()
        intensity = np.abs(u_s.u) ** 2
        time_proc = time2 - time1
        if verbose is True:
            print("num_proc: {}, time={}".format(1, time_proc))

    return intensity, u_s, time_proc


def extended_polychromatic_source(
    function_process,
    x0s: NDArrayFloat,
    wavelengths: NDArrayFloat,
    spectrum: NDArrayFloat,
    num_processors: int = num_max_processors,
    verbose: bool = False,
):
    """It performs an analysis of extendes source light. It needs a function with only an input parameter, that is x0s positions of sources. It determines the intensity for each wavelength and it is added.

    Args:
        function_process (function): function with accepts params as input Args:
        x0s (array): positions of sources
        wavelengths (array): wavelengths in the spectrum
        spectrum (array): weights for the spectrum. If None: uniform spectrum, else array with the same dimension as wavelengths
        num_processors (int): number of processors for the computation
        verbose (bool): if True send information to shell
     Returns:        - intensity (array, complex): intensity = intensity + spectrum[i] * np.abs(u_s[i].u)**2
        - u_s (Scalar_field_X): fields for each wavelength         - time_proc (float): time interval in the processing
    """

    dict_Parameters = []
    for i, wavelength in enumerate(wavelengths):
        for j, x0 in enumerate(x0s):
            dict_Parameters.append(dict(x0=x0, wavelength=wavelength, ij=(i, j)))

    u_s, time_proc = execute_multiprocessing(
        function_process, dict_Parameters, num_processors, verbose
    )
    intensity = np.zeros_like(u_s[0].u, dtype=float)
    for k in range(len(u_s)):
        # print( len(u_s), dict_Parameters[k]['ij'])
        i_wavelength = dict_Parameters[k]["ij"][0]
        intensity = intensity + spectrum[i_wavelength] * np.abs(u_s[k].u) ** 2
    intensity = intensity / (spectrum.sum() * len(x0s))

    return intensity, u_s, time_proc


def quality_factor(range_x: NDArrayFloat, num_x: int,
                   z: float, wavelength: float,
                   n: float = 1., verbose: bool = False):
    """Determine the quality factor for RS algorithm

    Args:
        x (np.array): x array with positions
        z (float): observation distance
        wavelength (float): wavelength)
        n (float): refractive index

    Returns:
        _type_: _description_
    """

    dx = range_x / num_x
    dr_real = dx
    rmax = range_x

    dr_ideal = (
        np.sqrt(
            (wavelength / n) ** 2 +
            rmax**2 +
            2 * (wavelength / n) * np.sqrt(rmax**2 + z**2)
        ) -
        rmax
    )
    quality = dr_ideal / dr_real

    if verbose:
        print("Quality factor = {:2.2f}".format(quality))

    return quality


def get_RS_minimum_z(range_x: float, num_x: int, wavelength: float, n: float = 1.,
                     quality=1., verbose: bool = True):
    """_summary_

    Args:
        range_x (float): range_x
        num_x (int): num_x
        z (float): z
        wavelength (float): wavelength
        quality (int, optional): quality. Defaults to 1.
        verbose (bool, optional): prints info. Defaults to True.

    Returns:
        _type_: _description_
    """

    dr_real = range_x / num_x
    rmax = range_x

    zmin = np.sqrt(
        (
            ((quality * dr_real + rmax) ** 2 - (wavelength / n) ** 2 - rmax**2) /
            2 *
            n /
            wavelength
        ) **
        2 -
        rmax**2
    )

    if verbose:
        print("z min = {:2.2f}".format(zmin))

    return zmin
