# !/usr/bin/env python3

# ----------------------------------------------------------------------
# Name:        scalar_fields_XY.py
# Purpose:     Class and methods for handling XY scalar fields
#
# Author:      Luis Miguel Sanchez Brea
#
# Created:     2024
# Licence:     GPLv3
# ----------------------------------------------------------------------


"""
This module generates Scalar_field_XY class.

It can be considered an extension of Scalar_field_X for visualizing XY fields

For the case of Rayleigh sommefeld it is not necessary to compute all z positions but the final.

Nevertheless, for BPM method, intermediate computations are required. In this class, intermediate results are stored.

It is required also for generating masks and fields.



The main atributes are:
    * self.x - x positions of the field
    * self.y - y positions of the field
    * self.wavelength - wavdelength of the incident field. The field is monochromatic
    * self.u (numpy.np.array): equal size to x * y. complex field
    * self.X (numpy.np.array): equal size to x * y. complex field
    * self.Y (numpy.np.array): equal size to x * y. complex field
    * self.quality (float): quality of RS algorithm
    * self.info (str): description of data
    * self.type (str): Class of the field
    * self.date (str): date of execution

The magnitude is related to microns: `micron = 1.`

*Class for XY scalar fields*

*Definition of a scalar field*
    * instatiation, duplicate
    * save, load data
    * cut_resample, binarize, discretize
    * get_phase, get_amplitude, remove_amplitude, remove_phase, amplitude2phase, phase2amplitude

*Propagation*
    * fft, ifft, RS, RS_simple, RS_amplificacion

*Drawing functions*
    * draw, draw_profile,
    * video, progresion

*Parameters*
    * profile

*Functions outside the class*
    * draw_several_fields
    * draw2D
    * several_propagations
    * kernelRS, kernelRSinverse, kernelFresnel, WPM_schmidt_kernel
"""

import copy
import datetime
import time
from typing import Tuple

import matplotlib.animation as animation
import scipy.ndimage
from matplotlib import rcParams
from numpy import gradient
from numpy.lib.scimath import sqrt as csqrt
from scipy.fftpack import fft2, fftshift, ifft2
from scipy.interpolate import RectBivariateSpline
from scipy.signal import czt

from .__init__ import np, plt
from .__init__ import degrees, mm, seconds, um

from .config import bool_raise_exception, CONF_DRAWING, Draw_XY_Options, Save_mask_Options, get_scalar_options
from .config import options_pupil

from .utils_typing import npt, Any, NDArray,  NDArrayFloat, NDArrayComplex
from .utils_common import get_date, load_data_common, save_data_common, add, check_none, oversampling, get_scalar, rmul
from .utils_drawing import (draw2D, normalize_draw, prepare_drawing,
                            reduce_matrix_size)
from .utils_math import get_edges, get_k, nearest, nearest2, reduce_to_1, rotate_image, Bluestein_dft_xy
from .utils_optics import beam_width_2D, field_parameters, normalize_field
from .scalar_fields_X import Scalar_field_X
from .scalar_fields_XZ import Scalar_field_XZ
from .scalar_fields_Z import Scalar_field_Z
from .utils_common import get_instance_size_MB

try:
    import screeninfo
except:
    print("screeninfo not imported.")

try:
    import cv2
except:
    print("cv2 not imported. Function send_image_screen cannot be used")

percentage_intensity_config = CONF_DRAWING['percentage_intensity']

class Scalar_field_XY():
    """Class for working with XY scalar fields.

    Args:
        x (numpy.np.array): linear np.array with equidistant positions. The number of data is preferibly :math:`2^n` .
        y (numpy.np.array): linear np.array wit equidistant positions for y values
        wavelength (float): wavelength of the incident field
        info (str): String with info about the simulation

    Attributes:
        self.x (numpy.np.array): linear np.array with equidistant positions. The number of data is preferibly :math:`2^n` .
        self.y (numpy.np.array): linear np.array wit equidistant positions for y values
        self.wavelength (float): wavelength of the incident field.
        self.u (numpy.np.array): (x,z) complex field
        self.info (str): String with info about the simulation
    """

# flake8: noqa

    def __init__(self, x: NDArrayFloat | None = None, y: NDArrayFloat | None = None,
                 wavelength: float  = 0., n_background: float = 1., info: str = ""):
        self.x = x
        self.y = y
        self.wavelength = wavelength
        self.n_background = n_background
        if x is not None and y is not None:
            self.X, self.Y = np.meshgrid(x, y)
            self.u = np.zeros(np.shape(self.X), dtype=complex)
        else:
            self.X = None
            self.Y = None
            self.u = None
        
        self.n = None
        self.borders = None
            
        self.info = info
        self.reduce_matrix = 'standard'  # 'None, 'standard', (5,5)
        self.type = 'Scalar_field_XY'
        self.date = get_date()
        self.quality = 0
        self.CONF_DRAWING = CONF_DRAWING

    @check_none('x', 'y', 'u', raise_exception=bool_raise_exception)
    def __str__(self):
        """Represents main data of the atributes"""

        Imin = (np.abs(self.u)**2).min()
        Imax = (np.abs(self.u)**2).max()
        phase_min = (np.angle(self.u)).min()/degrees
        phase_max = (np.angle(self.u)).max()/degrees
        print("{}\n - x:  {},   y:  {},   u:  {}".format(
            self.type, self.x.shape, self.y.shape, self.u.shape))
        print(
            " - xmin:       {:2.2f} um,  xmax:      {:2.2f} um,  Dx:   {:2.2f} um"
            .format(self.x[0], self.x[-1], self.x[1] - self.x[0]))
        print(
            " - ymin:       {:2.2f} um,  ymax:      {:2.2f} um,  Dy:   {:2.2f} um"
            .format(self.y[0], self.y[-1], self.y[1] - self.y[0]))
        print(" - Imin:       {:2.2f},     Imax:      {:2.2f}".format(
            Imin, Imax))
        print(" - phase_min:  {:2.2f} deg, phase_max: {:2.2f} deg".format(
            phase_min, phase_max))

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

        Returns:
            Scalar_field_X: `u3 = u1 + u2`
        """

        if self.type == 'Scalar_mask_XY':
            t = add(self, other, kind='mask')
        elif self.type == 'Scalar_source_XY' or 'Scalar_field_XY':
            t = add(self, other, kind='source')

        t.type = self.type
            
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

    @check_none('x', 'y', 'u', raise_exception=bool_raise_exception)
    def __sub__(self, other):
        """Substract two Scalar_field_XY. For example two light sources or two masks

        Args:
            other (Scalar_field_X): field to substract

        Returns:
            Scalar_field_X: `u3 = u1 - u2`

        TODO:  It can be improved for masks (not having less than 1)
        """
        u3 = Scalar_field_XY(self.x, self.y, self.wavelength)
        u3.u = self.u - other.u
        return u3
    
    @check_none('x', 'y', 'u', raise_exception=bool_raise_exception)
    def __mul__(self, other):
        """Multiply two fields. For example  :math:`u_1(x)= u_0(x)*t(x)`

        Args:
            other (Scalar_field_X): field to multiply

        Returns:
            Scalar_field_X: :math:`u_1(x)= u_0(x)*t(x)`
        """
        new_field = Scalar_field_XY(self.x, self.y, self.wavelength)
        new_field.u = self.u * other.u
        new_field.type = self.type

        return new_field


    @check_none('x', 'y', 'u', raise_exception=bool_raise_exception)
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

        if self.type == 'Scalar_mask_XY':
            t = rmul(self, number, kind='intensity')
        elif self.type == 'Scalar_source_XY' or 'Scalar_field_XY':
            t = rmul(self, number, kind='amplitude')
            
        return t


    @check_none('x', 'y', 'u', raise_exception=bool_raise_exception)
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

    @check_none('x', 'y', 'X', 'Y', 'u', raise_exception=bool_raise_exception)
    def __rotate__(self, angle: float,
                   position: tuple[float, float] | None = None):
        """Rotation of X,Y with respect to position

        Args:ing
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

    def size(self, verbose: bool = False):
        """returns the size of the instance in MB.

        Args:
            verbose (bool, optional): prints size in Mb. Defaults to False.

        Returns:
            float: size in MB
        """

        return get_instance_size_MB(self, verbose)

        

    @check_none('u', raise_exception=bool_raise_exception)
    def conjugate(self, new_field: bool = True):
        """Conjugates the field
        """

        if new_field is True:
            u_new = self.duplicate()
            u_new.u = np.conj(self.u)
            return u_new
        else:
            self.u = np.conj(self.u)

    def duplicate(self, clear: bool = False):
        """Duplicates the instance"""
        # new_field = Scalar_field_XY(self.x, self.y, self.wavelength)
        # new_field.u = self.u
        # tipo = type(self)
        # new_field = tipo(self)
        new_field = copy.deepcopy(self)
        if clear is True:
            new_field.clear_field()
        return new_field

    def reduce_to_1(self):
        """All the values greater than 1 pass to 1. This is used for Scalar_masks when we add two masks.
        """

        self = reduce_to_1(self)


    @check_none('x', 'y', raise_exception=bool_raise_exception)
    def rotate(self, angle: float, position: tuple[float, float] | None = None,
               new_field: bool = False):
        """Rotation of X,Y with respect to position. If position is not given, rotation is with respect to the center of the image

        Args:
            angle (float): angle to rotate, in radians
            position (float, float): position of center of rotation
        """

        if position is None:
            x0 = (self.x[-1] + self.x[0])/2
            y0 = (self.y[-1] + self.y[0])/2
        else:
            x0, y0 = position

        center_rotation = y0, x0

        u_real_rotate = rotate_image(self.x, self.y, np.real(self.u), -angle * 180 / np.pi, center_rotation)
        u_imag_rotate = rotate_image(self.x, self.y, np.imag(self.u), -angle * 180 / np.pi, center_rotation)
        u_rotate = u_real_rotate + 1j * u_imag_rotate

        if new_field is True:
            u_new = Scalar_field_XY(self.x, self.y, self.wavelength)
            u_new.u = u_rotate
            return u_new
        else:
            self.u = u_rotate



    @check_none('x', 'y', 'X', 'Y', raise_exception=bool_raise_exception)
    def apodization(self, power: int = 10):
        """Multiply field by an apodizer. The apodizer is a super_gauss function.

        Args:
            power (int, optional): size of apodization. Defaults to 10.

        """


        width_x = (self.x[-1] - self.x[0])/2
        width_y = (self.y[-1] - self.y[0])/2

        center_x = (self.x[-1] + self.x[0])/2
        center_y = (self.y[-1] + self.y[0])/2

        t = np.exp(-((self.X - center_x) / width_x)**power -
                   ((self.Y - center_y) / width_y)**power)
        self.u = t

    @check_none('u', raise_exception=bool_raise_exception)
    def clear_field(self):
        """Removes the field: self.u=0.
        """
        
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
            final_filename = save_data_common(self, filename, add_name, description, verbose)
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

        if verbose is True:
            print(dict0.keys())

    @check_none('u', raise_exception=bool_raise_exception)
    def save_mask(self,
                  filename: str = "",
                  kind: Save_mask_Options = 'amplitude',
                  binarize: bool = False,
                  cmap: str = 'gray',  # or cm
                  info: str = ""):
        """Create a mask in a file, for example, ablation or litography engraver

        Args:
            filename (str): file name
            kind (str): save amplitude, phase or intensity. If intensity it is normalized to (0,1)
            binarize (bool): If True convert the mask in (0,1) levels
            cmap (str): colormap. (gray)
            info (str): info of the mask

        Returns:
            float: area (in um**2)
        """

        # filename for txt
        name = filename.split(".")
        nombreTxt = name[0] + ".txt"

        # image
        plt.figure()
        filter = np.abs(self.u) > 0

        if kind == 'amplitude':
            mask = np.abs(self.u)

        elif kind == 'phase':
            mask = np.angle(self.u)
            mask = (mask - mask.min()) / (mask.max() - mask.min())
            mask = mask * filter

        elif kind == 'intensity':
            mask = np.abs(self.u)**2
            cmap = 'hot'

        if binarize is True:
            mask_min = mask.min()
            mask_max = mask.max()
            mask_mean = (mask_max + mask_min)/2

            mask2 = np.zeros_like(mask)
            mask2[mask < mask_mean] = 0
            mask2[mask >= mask_mean] = 1

            mask = mask2

        plt.imsave(filename, mask, cmap=cmap, dpi=100, origin='lower')
        plt.close()

        # important data
        ofile = open(nombreTxt, "w")
        ofile.write("filename %s\n" % filename)
        ofile.write("date: {}\n".format(datetime.date.today()))
        if info is not None:
            ofile.write("\ninfo:\n")
            ofile.write(info)
        ofile.write("\n\n")
        ofile.write("mask length: %i x %i\n" % (len(self.x), len(self.y)))
        ofile.write("x0 = %f *um, x1 = %f *um, Deltax = %f *um\n" %
                    (self.x.min(), self.x[-1], self.x[1] - self.x[0]))
        ofile.write("y0 = %f *um, y1 = %f *um, Deltay = %f *um\n" %
                    (self.y.min(), self.y[-1], self.y[1] - self.y[0]))

        ofile.write("wavelength = %f *um" % self.wavelength)
        ofile.close()

        return mask


    @check_none('x', 'y', 'u')
    def oversampling(self, factor_rate: int | tuple, new_field: bool = False):
        """oversampling. It has been implemented in scalar X, XY, XZ, and XYZ frames reduce the pixel size of the masks and fields. 
        This is also performed with the cut_resample function. However, this function oversamples with integer factors.
        
        Args:
            factor_rate (int | tuple, optional): factor rate. Defaults to 2.
            new_field (bool, optional): if True, returns a new field. Defaults to False.
        """



        if new_field is True:
            cls = self.duplicate()
            cls = oversampling(cls, factor_rate)
            return cls
        else:
            self = oversampling(self, factor_rate)


    @check_none('x', 'y', 'u', raise_exception=bool_raise_exception)
    def cut_resample(self,
                     x_limits: tuple[float, float] = '',
                     y_limits: tuple[float, float] = '',
                     num_points: int = [],
                     new_field: bool = False,
                     interp_kind: tuple[int, int] = (3, 1)):
        """it cut the field to the range (x0,x1). If one of this x0,x1 positions is out of the self.x range it do nothing. It is also valid for resampling the field, just write x0,x1 as the limits of self.x

        Args:
            x_limits (float,float): (x0,x1) starting and final points to cut. if '' - takes the current limit x[0] and x[-1]
            y_limits (float,float): (y0,y1) - starting and final points to cut. if '' - takes the current limit y[0] and y[-1]
            num_points (int): it resamples x, y and u. [], '',,None -> it leave the points as it is
            new_field (bool): it returns a new Scalar_field_XY
            interp_kind: numbers between 1 and 5
        """
        
        if x_limits == '':
            # used only for resampling
            x0 = self.x[0]
            x1 = self.x[-1]
        else:
            x0, x1 = x_limits

        if y_limits == '':
            # used only for resampling
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
        # new_num_points = i_x1 - i_x0
        i_y0, _, _ = nearest(self.y, y0)
        i_y1, _, _ = nearest(self.y, y1)

        kxu, kxn = interp_kind

        if num_points not in ([], '', 0, None):
            num_points_x, num_points_y = num_points
            x_new = np.linspace(x0, x1, num_points_x)
            y_new = np.linspace(y0, y1, num_points_y)
            X_new, Y_new = np.meshgrid(x_new, y_new)

            f_interp_abs = RectBivariateSpline(self.y, self.x,np.abs(self.u),kx=kxu,ky=kxu,s=0)
            f_interp_phase = RectBivariateSpline(self.y,  self.x,  np.angle(self.u),  kx=kxu,  ky=kxu,  s=0)
            u_new_abs = f_interp_abs(y_new, x_new)
            u_new_phase = f_interp_phase(y_new, x_new)
            u_new = u_new_abs * np.exp(1j * u_new_phase)

        else:
            i_s = slice(i_x0, i_x1)
            j_s = slice(i_y0, i_y1)
            x_new = self.x[i_s]
            y_new = self.y[j_s]
            X_new, Y_new = np.meshgrid(x_new, y_new)
            u_new = self.u[i_s, j_s]

        if new_field is False:
            self.x = x_new
            self.y = y_new
            self.u = u_new
            self.X = X_new
            self.Y = Y_new
        elif new_field is True:
            field = Scalar_field_XY(x=x_new,
                                    y=y_new,
                                    wavelength=self.wavelength)
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


    @check_none('x', 'y', 'u', raise_exception=bool_raise_exception)
    def get_mask_size(self, threshold=0, has_draw=False, verbose=False):
        """get_mask_size. Computes the size of the mask in x and y directions.

        Args:
            threshold (float, optional): % of maximum intensity to determine size, if not binary. Defaults to 0.
            has_draw (bool, optional): it draws profiles. Defaults to False.
            verbose (bool, optional): It shows the size. Defaults to False.

        Returns:
            (float, float): size in micrometers of the mask in x and y directions
        """

        dx = self.x[1] - self.x[0]
        dy = self.y[1] - self.y[0]

        intensity = self.intensity()

        profile_x = intensity.sum(axis=0)
        profile_x = profile_x / np.max(profile_x)
        profile_x_normalized = profile_x / np.max(profile_x)
        profile_x_normalized[profile_x_normalized<threshold] = 0
        profile_x_normalized[profile_x_normalized>=threshold] = 1

        profile_y = intensity.sum(axis=1)
        profile_y = profile_y / np.max(profile_y)
        profile_y_normalized = profile_y / np.max(profile_y)
        profile_y_normalized[profile_y_normalized<threshold] = 0
        profile_y_normalized[profile_y_normalized>=threshold] = 1

        if has_draw:
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            axes[0].plot(self.x, profile_x)
            axes[0].plot(self.x, profile_x_normalized)
            axes[0].set_title("Intensity profile_x")
            axes[0].set_xlabel(r"$x (\mu m)$")
            axes[1].plot(self.y, profile_y)
            axes[1].plot(self.y, profile_y_normalized)
            axes[1].set_title("Intensity profile_y")
            axes[0].set_ylabel(r"$y (\mu m)$")
            plt.tight_layout()
            plt.show()  

        non_zero_indices_x = np.nonzero(profile_x_normalized)[0]
        if len(non_zero_indices_x) == 0:
            return None, None
        first_non_zero_index_x = non_zero_indices_x[0]
        last_non_zero_index_x = non_zero_indices_x[-1]

        non_zero_indices_y = np.nonzero(profile_y_normalized)[0]
        if len(non_zero_indices_y) == 0:
            return None, None
        first_non_zero_index_y = non_zero_indices_y[0]
        last_non_zero_index_y = non_zero_indices_y[-1]

        size_x = (last_non_zero_index_x - first_non_zero_index_x) * dx
        size_y = (last_non_zero_index_y - first_non_zero_index_y) * dy
        
        
        if verbose:
            if size_x > 1*mm or size_y > 1*mm:
                print("mask size: {:2.2f} mm, {:2.2f} mm".format(size_x/mm, size_y/mm))
            else:
                print("mask size: {:2.2f} um, {:2.2f} um".format(size_x/um, size_y/um))


        return size_x, size_y


    @check_none('x', 'y', 'u', raise_exception=bool_raise_exception)
    def pupil(self,  r0: tuple[float, float] | None = None,
              radius: float | tuple[float, float] | None = None,
              angle: float = 0*degrees, kind: options_pupil ='direct'):
        """place a pupil in the field. If r0 or radius are None, they are computed using the x,y parameters.

        Args:
            r0 (float, float): center of circle/ellipse
            radius (float, float) or (float): radius of circle/ellipse
            angle (float): angle of rotation in radians
            kind (str): 'direct' or 'inverse'. If direct, the pupil is dark. If inverse, the pupil is clear (removes everyting outside radius, but tranmission is 1).
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
        self.u = self.u * pupil0
        if kind == 'inverse':
            self.u[~ipasa] = 1
        return self
    """
    def fft_proposal(self,
                     z=0,
                     shift=True,
                     remove0=True,
                     matrix: bool = False,
                     new_field: bool = False,
                     cut_square=True):
        Fast Fourier Transform (FFT) of the field.
        Args:
            z (float): distance to the observation plane or focal of lens
                       if z==0, no x,y scaled is produced
            shift (bool): if True, fftshift is performed
            remove0 (bool): if True, central point is removed
            matrix (bool):  if True only matrix is returned. if False, returns Scalar_field_X
            new_field (bool): if True returns Scalar_field_X, else it puts in self
            cut_square (bool): when the mask is not squared, the fft is computed using the squared XY version of the mask, removing the laterals.

        Returns:
            (np.array or Scalar_field_X or None): FFT of the input field
        
        k = 2 * np.pi / self.wavelength
        num_x = self.x.size
        num_y = self.y.size
        #print(num_x, num_y, np.bitwise_and(num_x != num_y, cut_square == True))
        u_field = self.u
        range_x = self.x[1] - self.x[0]
        range_y = self.y[1] - self.y[0]
        # if np.bitwise_and(num_x != num_y, cut_square == True):
        #     num_final = min(num_x, num_y)
        #     print(num_final)
        #     u_field = fftshift(u_field)
        #     u_field = u_field[0:num_final, 0:num_final]

        ttf1 = np.fft.fft2(u_field)
        # print(ttf1.shape)

        delta_x = self.x[1] - self.x[0]
        freq_nyquist_x = 1 / (2 * delta_x)
        kx = np.linspace(-freq_nyquist_x, freq_nyquist_x,
                         num_x) * self.wavelength

        delta_y = self.y[1] - self.y[0]
        freq_nyquist_y = 1 / (2 * delta_y)
        ky = np.linspace(-freq_nyquist_y, freq_nyquist_y,
                         num_y) * self.wavelength

        if remove0 is True:
            ttf1[0, 0] = 0

        if shift is True:
            ttf1 = np.fft.fftshift(ttf1)

        if matrix is True:
            return ttf1

        if z is None:
            x_new = kx  # exit in angles (radians)
            y_new = ky  # exit in angles (radians)
        elif z == 0:
            x_new = self.y * num_x / num_y
            y_new = self.x
        else:
            x_new = kx * z  # exit distances at a observation plane z
            y_new = ky * z  # exit distances at a observation plane z

        if new_field is True:
            field_output = Scalar_field_XY(x_new, y_new, self.wavelength)
            field_output.u = ttf1
            return field_output
        else:
            self.u = ttf1
            self.x = x_new
            self.y = y_new
            self.X, self.Y = np.meshgrid(self.x, self.y)
    """

    @check_none('x', 'y', 'u', raise_exception=bool_raise_exception)
    def fft(self, z: float = 0, shift: bool = True, remove0: bool = True,
            matrix: bool = False, new_field: bool = False):
        """Fast Fourier Transform (FFT) of the field.

        Args:
            z (float): distance to the observation plane or focal of lens
                       if z==0, no x,y scaled is produced
            shift (bool): if True, fftshift is performed
            remove0 (bool): if True, central point is removed
            matrix (bool):  if True only matrix is returned. if False, returns Scalar_field_X
            new_field (bool): if True returns Scalar_field_X, else it puts in self

        Returns:
            (np.array or Scalar_field_X or None): FFT of the input field
        """

        k = 2 * np.pi / self.wavelength

        ttf1 = np.fft.fft2(self.u)

        num_x = self.x.size
        delta_x = self.x[1] - self.x[0]
        freq_nyquist_x = 1 / (2 * delta_x)
        kx = np.linspace(-freq_nyquist_x, freq_nyquist_x, num_x) * self.wavelength

        num_y = self.y.size
        delta_y = self.y[1] - self.y[0]
        freq_nyquist_y = 1 / (2 * delta_y)
        ky = np.linspace(-freq_nyquist_y, freq_nyquist_y,
                         num_y) * self.wavelength

        if remove0 is True:
            ttf1[0, 0] = 0

        if shift is True:
            ttf1 = np.fft.fftshift(ttf1)

        if matrix is True:
            return ttf1

        if z is None:
            x_new = kx  # exit in angles (radians)
            y_new = ky  # exit in angles (radians)
        elif z == 0:
            x_new = self.x
            y_new = self.y
        else:
            x_new = kx * z  # exit distances at a observation plane z
            y_new = ky * z  # exit distances at a observation plane z

        if new_field is True:
            field_output = Scalar_field_XY(x_new, y_new, self.wavelength)
            field_output.u = ttf1
            return field_output
        else:
            self.u = ttf1
            self.x = x_new
            self.y = y_new
            self.X, self.Y = np.meshgrid(self.x, self.y)

    
    # @check_none('x', 'y', 'u', raise_exception=bool_raise_exception)
    # def ifft_proposal(self, z: float = 0*mm, shift: bool = True, remove0: bool = True,
    #                   matrix: bool = False, new_field: bool = False):
    #     """Fast Fourier Transform (fft) of the field.

    #     Args:
    #         z (float): distance to the observation plane or focal of lens
    #         shift (bool): if True, fftshift is performed
    #         remove0 (bool): if True, central point is removed
    #         matrix (bool):  if True only matrix is returned. If False, returns Scalar_field_X
    #         new_field (bool): if True returns Scalar_field_X, else puts in self

    #     Returns:
    #         (np.array or Scalar_field_X or None): FFT of the input field
    #     """
        
    #     k = 2 * np.pi / self.wavelength
    #     num_x = self.x.size
    #     num_y = self.y.size
    #     u_field = self.u

    #     range_x = self.x[1] - self.x[0]
    #     range_y = self.y[1] - self.y[0]

    #     ttf1 = np.fft.ifft2(u_field)

    #     delta_x = self.x[1] - self.x[0]
    #     freq_nyquist_x = 1 / (2 * delta_x)
    #     kx = np.linspace(-freq_nyquist_x, freq_nyquist_x,
    #                      num_x) * self.wavelength

    #     delta_y = self.y[1] - self.y[0]
    #     freq_nyquist_y = 1 / (2 * delta_y)
    #     ky = np.linspace(-freq_nyquist_y, freq_nyquist_y,
    #                      num_y) * self.wavelength

    #     if remove0 is True:
    #         ttf1[0, 0] = 0

    #     if shift is True:
    #         ttf1 = np.fft.fftshift(ttf1)

    #     if matrix is True:
    #         return ttf1

    #     if z is None:
    #         x_new = kx  # exit in angles (radians)
    #         y_new = ky  # exit in angles (radians)
    #     elif z == 0:
    #         x_new = self.y
    #         y_new = self.x * num_x / num_y
    #     else:
    #         x_new = kx * z  # exit distances at a observation plane z
    #         y_new = ky * z  # exit distances at a observation plane z

    #     if new_field is True:
    #         field_output = Scalar_field_XY(x_new, y_new, self.wavelength)
    #         field_output.u = ttf1
    #         return field_output
    #     else:
    #         self.u = ttf1
    #         self.x = x_new
    #         self.y = y_new
    #         self.X, self.Y = np.meshgrid(self.x, self.y)

    @check_none('x', 'y', 'u', raise_exception=bool_raise_exception)
    def ifft(self, z: float = 0*mm, shift: bool = True, remove0: bool = True,
             matrix: bool = False, new_field: bool = False):
        """Fast Fourier Transform (fft) of the field.

        Args:
            z (float): distance to the observation plane or focal of lens
            shift (bool): if True, fftshift is performed
            remove0 (bool): if True, central point is removed
            matrix (bool):  if True only matrix is returned. If False, returns Scalar_field_X
            new_field (bool): if True returns Scalar_field_X, else puts in self

        Returns:
            (np.array or Scalar_field_X or None): FFT of the input field
        """
        
        k = 2 * np.pi / self.wavelength

        ttf1 = np.fft.ifft2(self.u)
        ttf1 = ttf1

        if remove0 is True:
            ttf1[0, 0] = 0

        if shift is True:
            ttf1 = np.fft.fftshift(ttf1)

        if matrix is True:
            return ttf1

        # x scaling - Jose Antonio Gómez Pedrero
        num_x = self.x.size
        delta_x = self.x[1] - self.x[0]
        freq_nyquist_x = 1 / (2 * delta_x)
        kx = np.linspace(-freq_nyquist_x, freq_nyquist_x, num_x) * self.wavelength

        num_y = self.y.size
        delta_y = self.y[1] - self.y[0]
        freq_nyquist_y = 1 / (2 * delta_y)
        ky = np.linspace(-freq_nyquist_y, freq_nyquist_y,
                         num_y) * self.wavelength

        if z is None:
            x_new = kx  # exit in angles (radians)
            y_new = ky  # exit in angles (radians)
        elif z == 0:
            x_new = self.x
            y_new = self.y
        else:
            x_new = kx * z  # exit distances at a observation plane z
            y_new = ky * z  # exit distances at a observation plane z

        if new_field is True:
            field_output = Scalar_field_XY(x_new, y_new, self.wavelength)
            field_output.u = ttf1
            return field_output
        else:
            self.u = ttf1
            self.x = x_new
            self.y = y_new
            self.X, self.Y = np.meshgrid(self.x, self.y)


    @check_none('x', 'y', 'u', raise_exception=bool_raise_exception)
    def _RS_(self, z: float, n: float, new_field: bool = True, out_matrix: bool = False,
             kind: str = 'z', xout: float | None = None, yout: float | None = None,
             verbose: bool = False):
        r"""Fast-Fourier-Transform  method for numerical integration of diffraction Rayleigh-Sommerfeld formula. `Thin Element Approximation` is considered for determining the field just after the mask: :math:`\mathbf{E}_{0}(\zeta,\eta)=t(\zeta,\eta)\mathbf{E}_{inc}(\zeta,\eta)` Is we have a field of size N*M, the result of propagation is also a field N*M. Nevertheless, there is a parameter `amplification` which allows us to determine the field in greater observation planes (jN)x(jM).

        Args:
            z (float): distance to observation plane.
                if z<0 inverse propagation is executed
            n (float): refractive index
            new_field (bool): if False the computation goes to self.u
                              if True a new instance is produced

            xout (float), init point for amplification at x
            yout (float), init point for amplification at y
            verbose (bool): if True it writes to shell

        Returns:
            if New_field is True: Scalar_field_X
            else None

        Note:
            One adventage of this approach is that it returns a quality parameter: if self.quality>1, propagation is right.


        References:
            F. Shen and A. Wang, “Fast-Fourier-transform based numerical integration method for the Rayleigh-Sommerfeld diffraction formula,” Appl. Opt., vol. 45, no. 6, pp. 1102–1110, 2006.

        """

        if xout is None:
            xout = self.x[0]
        if yout is None:
            yout = self.y[0]

        xout = self.x + xout - self.x[0]
        yout = self.y + yout - self.y[0]

        nx = len(xout)
        ny = len(yout)
        dx = xout[1] - xout[0]
        dy = yout[1] - yout[0]

        dr_real = np.sqrt(dx**2 + dy**2)
        rmax = np.sqrt((xout**2).max() + (yout**2).max())
        dr_ideal = np.sqrt((self.wavelength / n)**2 + rmax**2 + 2 *
                           (self.wavelength / n) * np.sqrt(rmax**2 + z**2)) - rmax
        self.quality = dr_ideal / dr_real

        if verbose is True:
            if (self.quality.min() >= 0.99):
                print('Good result: factor {:2.2f}'.format(self.quality),
                      end='\r')
            else:
                print('- Needs denser sampling: factor {:2.2f}\n'.format(
                    self.quality))
        
        # could not make work properly           
        precise = 0 
        if precise:
            a = [4, 2]
            num_repx = int(round((nx)/2) - 1)
            num_repy = int(round((ny)/2) - 1)
            bx = np.array(a * num_repx)
            by = np.array(a * num_repy)
            cx = np.concatenate(((1, ), bx, (2, 1))) / 3.
            cy = np.concatenate(((1, ), by, (2, 1))) / 3.

            if float(nx)/2 == round(nx/2):  # es par
                i_centralx = num_repx + 1
                cx = np.concatenate((cx[:i_centralx], cx[i_centralx + 1:]))
            if float(ny)/2 == round(ny/2):  # es par
                i_centraly = num_repy + 1
                cy = np.concatenate((cy[:i_centraly], cy[i_centraly + 1:]))

            W = (cx[:, np.newaxis] * cy[np.newaxis, :]).T

        else:
            W = 1

        U = np.zeros((2 * ny - 1, 2 * nx - 1), dtype=complex)
        U[0:ny, 0:nx] = np.array(W * self.u)

        xext = self.x[0] - xout[::-1]
        xext = xext[0:-1]
        xext = np.concatenate((xext, self.x - xout[0]))

        yext = self.y[0] - yout[::-1]
        yext = yext[0:-1]
        yext = np.concatenate((yext, self.y - yout[0]))

        Xext, Yext = np.meshgrid(xext, yext)

        if z > 0:
            H = kernelRS(Xext, Yext, self.wavelength, z, n, kind=kind)
        else:
            H = kernelRSinverse(Xext, Yext, self.wavelength, z, n, kind=kind)

        S = ifft2(fft2(U) * fft2(H)) * dx * dy
        Usalida = S[ny - 1:, nx - 1:]  

        if out_matrix is True:
            return Usalida

        if new_field is True:
            field_output = Scalar_field_XY(self.x, self.y, self.wavelength)
            field_output.u = Usalida
            field_output.quality = self.quality
            return field_output
        else:
            self.u = Usalida

    @check_none('x', 'y', raise_exception=bool_raise_exception)
    def RS(self, z: float, amplification: tuple[int, int] = (1, 1), n: float = 1.,
            new_field: bool = True, matrix: bool = False,
            xout: float | None = None, yout: float | None = None,
            kind: str = 'z', verbose: bool = False):
        """Fast-Fourier-Transform  method for numerical integration of diffraction Rayleigh-Sommerfeld formula. Is we have a field of size N*M, the result of propagation is also a field N*M. Nevertheless, there is a parameter `amplification` which allows us to determine the field in greater observation planes (jN)x(jM).

        Args:
            amplification (int, int): number of frames in x and y direction
            z (float): distance to observation plane. if z<0 inverse propagation is executed
            n (float): refractive index
            new_field (bool): if False the computation goes to self.u, if True a new instance is produced.
            matrix(Bool): if True returns a matrix, else a Scalar_field_XY
            xout (float): If not None, the sampling area is moved. This is the left position
            yout (float): If not None, the sampling area y moved. This is the lower position.
            kind (str):
            verbose (bool): if True it writes to shell

        Returns:
            if New_field is True: Scalar_field_X, else None.

        Note:
            One advantage of this approach is that it returns a quality parameter: if self.quality>1, propagation is right.

        References:
            F. Shen and A. Wang, “Fast-Fourier-transform based numerical integration method for the Rayleigh-Sommerfeld diffraction formula,” Appl. Opt., vol. 45, no. 6, pp. 1102–1110, 2006.
        """

        amplification_x, amplification_y = amplification
        width_x = self.x[-1] - self.x[0]
        width_y = self.y[-1] - self.y[0]
        num_pixels_x = len(self.x)
        num_pixels_y = len(self.y)

        if amplification_x * amplification_y > 1:

            posiciones_x = -amplification_x * width_x/2 + np.array(
                list(range(amplification_x))) * width_x
            posiciones_y = -amplification_y * width_y/2 + np.array(
                list(range(amplification_y))) * width_y

            X0 = np.linspace(-amplification_x * width_x/2,
                             amplification_x * width_x/2,
                             num_pixels_x * amplification_x)
            Y0 = np.linspace(-amplification_y * width_y/2,
                             amplification_y * width_y/2,
                             num_pixels_y * amplification_y)

            U_final = Scalar_field_XY(x=X0, y=Y0, wavelength=self.wavelength)

            for i, xi in zip(list(range(len(posiciones_x))),
                             np.flipud(posiciones_x)):
                for j, yi in zip(list(range(len(posiciones_y))),
                                 np.flipud(posiciones_y)):
                    # num_ventana = j * amplification_x + i + 1
                    u3 = self._RS_(z=z,
                                   n=n,
                                   new_field=False,
                                   kind=kind,
                                   xout=xi,
                                   yout=yi,
                                   out_matrix=True,
                                   verbose=verbose)
                    xshape = slice(i * num_pixels_x, (i + 1) * num_pixels_x)
                    yshape = slice(j * num_pixels_y, (j + 1) * num_pixels_y)
                    U_final.u[yshape, xshape] = u3

            if matrix is True:
                return U_final.u
            else:
                if new_field is True:
                    return U_final
                else:
                    self.u = U_final.u
                    self.x = X0
                    self.y = Y0
        else:

            if xout is None:
                u_s = self._RS_(z,
                                n,
                                new_field=new_field,
                                out_matrix=True,
                                kind=kind,
                                xout=xout,
                                yout=yout,
                                verbose=verbose)
            else:
                u_s = self._RS_(z,
                                n,
                                new_field=new_field,
                                out_matrix=True,
                                kind=kind,
                                xout=-xout + self.x[0] - width_x/2,
                                yout=-yout + self.y[0] - width_y/2,
                                verbose=verbose)

            if matrix is True:
                return u_s

            if new_field is True:
                U_final = Scalar_field_XY(x=self.x,
                                          y=self.y,
                                          wavelength=self.wavelength)
                U_final.u = u_s
                if xout is not None:
                    U_final.x = self.x + xout - self.x[0]
                    U_final.y = self.y + yout - self.y[0]
                    U_final.X, U_final.Y = np.meshgrid(self.x, self.y)

                return U_final
            else:
                self.u = u_s
                self.x = self.x + xout - self.x[0]
                self.y = self.y + yout - self.y[0]
                self.X, self.Y = np.meshgrid(self.x, self.y)

    @check_none('x', 'y', raise_exception=bool_raise_exception)
    def WPM(self,
            fn,
            zs: NDArrayFloat,
            num_sampling: list = (512, 512),
            ROI: tuple[NDArrayFloat] | None = (None, None),
            r_pos: float | None = None,
            z_pos: float | None = None,
            get_u_max: bool = True,
            has_edges: bool = True,
            pow_edge: int = 80,
            matrix: bool = False,
            verbose: bool = False):
        """WPM method used for very dense sampling. It does not storages the intensity distribution at propagation, but only selected areas. The areas to be stored are:
            - global view with a desired sampling given by num_sampling.
            - intensity at the last plane.
            - Intensity at plane with maximum intensity (focus of a lens, for example)
            - Region of interest given by ROI.

        Args:
            fn (function): Function that returns the refractive index at a plane in a numpy.np.array
            zs (np.array): np.linspace with positions z where to evaluate the field.
            num_sampling (int, int) or None: If None, it does not storage intermediate data. Otherwise, it stores a small XZ matrix which size in num_sampling.
            ROI (np.array, np.array) or None: x and z arrays with positions and sampling to store. It is used when we need to store with a high density a small area of the total field.
            r_pos (float or None): If not None, a high density longitudinal np.array with the field at x_pos position is stored.
            z_pos (float or None): If not None, a high density transversal np.array with the field at z_pos is stored.
            get_u_max (bool): If True, returns field at z position with maximum intensity.
            has_edges (bool): If True absorbing edges are used.
            pow_edge (float): If has_edges, power of the supergaussian.
            verbose (bool): If True prints information.

        References:

            1. M. W. Fertig and K.-H. Brenner, “Vector wave propagation method,” J. Opt. Soc. Am. A, vol. 27, no. 4, p. 709, 2010.

            2. S. Schmidt et al., “Wave-optical modeling beyond the thin-element-approximation,” Opt. Express, vol. 24, no. 26, p. 30188, 2016.

        """

        
        from diffractio.scalar_fields_XYZ import Scalar_field_XYZ
        
        k0 = 2 * np.pi / self.wavelength
        x = self.x
        y = self.y
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        dz = zs[1] - zs[0]

        u_iter = Scalar_field_XY(self.x, self.y, self.wavelength)

        kx = get_k(x, flavour='+')
        ky = get_k(y, flavour='+')

        KX, KY = np.meshgrid(kx, ky)

        k_perp2 = KX**2 + KY**2
        k_perp = np.sqrt(k_perp2)

        if has_edges is False:
            if hasattr(self, 'z'):
                has_filter = np.zeros_like(self.z)
            else:
                has_filter = np.zeros_like(zs)
        elif isinstance(has_edges, int):
            if hasattr(self, 'z'):
                has_filter = np.ones_like(self.z)
            else:
                has_filter = np.ones_line(zs)
        else:
            has_filter = has_edges

        width_edge = 0.95*(self.x[-1] - self.x[0])/2
        x_center = (self.x[-1] + self.x[0])/2
        y_center = (self.y[-1] + self.y[0])/2

        filter_x = np.exp(-(np.abs(self.X-x_center) / width_edge)**pow_edge)
        filter_y = np.exp(-(np.abs(self.Y-y_center) / width_edge)**pow_edge)
        filter_function = filter_x*filter_y

        u_iter = self.duplicate()

        # STORING field at maximum intensity
        if get_u_max is True:
            u_max = self.duplicate()
            I_max = 0.
            z_max = 0
        else:
            u_max = None
            z_max = None

        # Storing intensities at axis (x=0,0)
        intensities = np.zeros_like(zs)
        index_x_axis, _, _ = nearest(x, 0.)
        index_y_axis, _, _ = nearest(y, 0.)

        # Storing intensities at axis (x=x_pos)
        if r_pos is not None:
            from diffractio.scalar_fields_Z import Scalar_field_Z
            x_pos, y_pos = r_pos
            u_axis_x = Scalar_field_Z(zs, self.wavelength)
            index_x_axis, _, _ = nearest(x, x_pos)
            index_y_axis, _, _ = nearest(y, y_pos)
        else:
            u_axis_x = None

        # Storing fields at plane
        if z_pos is not None:
            u_axis_z = self.duplicate()
            index_zpos, _, _ = nearest(zs, z_pos)
        else:
            u_axis_z = None

        # STORING global view matrices
        if num_sampling is not None:
            len_x, len_y, len_z = num_sampling
            xout_gv = np.linspace(x[0], x[-1], len_x)
            yout_gv = np.linspace(y[0], y[-1], len_y)
            zout_gv = np.linspace(zs[0], zs[-1], len_z)
            u_out_gv = Scalar_field_XYZ(xout_gv,
                                        yout_gv,
                                        zout_gv,
                                        self.wavelength,
                                        info='from WPM_no_storage_2D')
            indexes_x_gv, _, _ = nearest2(x, xout_gv)
            indexes_y_gv, _, _ = nearest2(y, yout_gv)
            indexes_z_gv, _, _ = nearest2(zs, zout_gv)
            indexes_X_gv, indexes_Y_gv = np.meshgrid(indexes_x_gv, indexes_y_gv)

            u_out_gv.n = fn(xout_gv, yout_gv, zout_gv, self.wavelength)
        else:
            u_out_gv = None

        # STORING ROI
        if ROI is not None:
            xout_roi, yout_roi, zout_roi = ROI
            u_out_roi = Scalar_field_XYZ(xout_roi,
                                         yout_roi,
                                         zout_roi,
                                         self.wavelength,
                                         info='from WPM_no_storage_2D')
            indexes_x_roi, _, _ = nearest2(x, xout_roi)
            indexes_y_roi, _, _ = nearest2(y, yout_roi)
            indexes_z_roi, _, _ = nearest2(zs, zout_roi)

            indexes_X_roi, indexes_Y_roi = np.meshgrid(indexes_x_roi,        indexes_y_roi)
            # indexes_x_roi = np.unique(indexes_x_roi)
            # indexes_y_roi = np.unique(indexes_y_roi)
            # indexes_z_roi = np.unique(indexes_z_roi)

            u_out_roi.n = fn(xout_roi, yout_roi, zout_roi, self.wavelength)
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

            refractive_index = fn(x, y, np.array([
                zs[j - 1],
            ]), self.wavelength)

            u_iter.u = WPM_schmidt_kernel(u_iter.u, refractive_index, k0,
                                          k_perp2, dz) * filter_edge

            current_intensity = np.max(np.abs(u_iter.u)**2)
            intensities[j] = u_iter.intensity().transpose()[index_x_axis, index_y_axis]

            if r_pos is not None:
                u_axis_x.u[j] = u_iter.u[index_y_axis, index_x_axis]

            if z_pos is not None:
                if j == index_zpos:
                    u_axis_z.u = u_iter.u

            if num_sampling is not None:
                if j in indexes_z_gv:
                    u_out_gv.u[:, :, iz_out_gv] = u_iter.u[indexes_Y_gv,            indexes_X_gv]
                    u_out_gv.n[:, :, iz_out_gv] = refractive_index[indexes_Y_gv, indexes_X_gv]                   
                    iz_out_gv = iz_out_gv + 1

            if ROI is not None:
                if j in indexes_z_roi:
                    u_out_roi.u[:, :, iz_out_roi] = u_iter.u[indexes_Y_roi,              indexes_X_roi]
                    
                    u_out_roi.n[:, :, iz_out_roi] = refractive_index[indexes_Y_roi,              indexes_X_roi]                 
                    
                    iz_out_roi = iz_out_roi + 1

            if get_u_max is True:
                current_intensity = np.max(np.abs(u_iter.u)**2)
                if current_intensity > I_max:
                    I_max = u_iter.intensity().max()
                    u_max.u = u_iter.u
                    z_max = zs[j]

            if verbose is True:
                print("{}/{}".format(j, num_steps), sep='\r', end='\r')

        t2 = time.time()

        if verbose is True:
            print("Time = {:2.2f} s, time/loop = {:2.4} ms".format(
                t2 - t1, (t2 - t1) / num_steps * 1000))
            get_instance_size_MB(self, verbose)

        return u_iter, u_out_gv, u_out_roi, u_axis_x, u_axis_z, u_max, z_max

    @check_none('x', 'y', raise_exception=bool_raise_exception)
    def CZT(self, z: float, xout: NDArrayFloat | None = None,
            yout: NDArrayFloat | None = None, verbose: bool = False):
        """Chirped Z Transform algorithm for XY Scheme. z, xout, and yout parameters can be numbers or arrays.
        The output Scheme depends on this input parameters.

        The code for this algoritm is based on "Hu, Yanlei, et al. "Efficient full-path optical calculation of scalar and 
        vector diffraction using the Bluestein method." Light: Science & Applications 9.1 (2020): 119."
        
        However, the convolution Kernel has been changed to Rayleigh-Sommerfeld.
        
        Args:
            z (float): diffraction distance
            xout (np.array): x np.array with positions of the output plane
            yout (np.array): y np.array with positions of the output plane
            verbose (bool): If True, it prints some information

        Returns:
            u_out: Scalar_field_** depending of the input scheme. When all the parameters are numbers, it returns the complex field at that point.


        References:
                [Light: Science and Applications, 9(1), (2020)] 
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

        dx = self.x[1] - self.x[0]
        dy = self.y[1] - self.y[0]

        delta_out = np.zeros(2)
        if num_x > 1:
            delta_out[0] = (xend - xstart) / (num_x - 1)

        if num_y > 1:
            delta_out[1] = (yend - ystart) / (num_y - 1)

        Xout, Yout = np.meshgrid(xout, yout)

        if verbose:
            print("num x, num y, num z = {}, {}, {}".format(
                num_x, num_y, num_z))

        if num_z == 1:
            # calculating scalar diffraction below
            # F0 = np.exp(1j * k * z) / (1j * self.wavelength * z) * np.exp(
            #     1j * k/2 / z * (Xout**2 + Yout**2))
            # F = np.exp(1j * k/2 / z * (self.X**2 + self.Y**2))

            R1 = np.sqrt(Xout**2 + Yout**2 + z**2)
            R2 = np.sqrt(self.X**2 + self.Y**2 + z**2)

            if z > 0:
                F0 = 1 / (2 * np.pi) * np.exp(
                    1.j * k * R1) * z / R1**2 * (1 / R1 - 1.j * k)
                F = 1 / (2 * np.pi) * np.exp(
                    1.j * k * R2) * z / R2**2 * (1 / R2 - 1.j * k)
            else:
                F0 = 1 / (2 * np.pi) * np.exp(
                    -1.j * k * R1) * z / R1**2 * (1 / R1 + 1.j * k)
                F = 1 / (2 * np.pi) * np.exp(
                    -1.j * k * R2) * z / R2**2 * (1 / R2 + 1.j * k)

            k_factor = z * dx * dy * self.wavelength

            # using Bluestein method to calculate the complex amplitude of the outgoing light beam

            # one-dimensional FFT in one direction
            fsx = self.wavelength * z / dx  # dimension of the imaging plane
            fsy = self.wavelength * z / dy  # dimension of the imaging plane

            if num_x == 1 and num_y == 1:
                u0 = self.u * F
                fy1 = ystart + fsy/2
                fy2 = yend + fsy/2
                u0 = Bluestein_dft_xy(u0, fy1, fy2, fsy, num_y)
                # one-dimensional FFT in the other direction
                fx1 = xstart + fsx/2
                fx2 = xend + fsx/2
                u0 = Bluestein_dft_xy(u0, fx1, fx2, fsx, num_x)
                u0 = F0 * u0 * k_factor  # obtain the complex amplitude of the outgoing light beam

            elif num_x > 1 and num_y > 1:
                u0 = self.u * F

                fy1 = ystart + fsy/2
                fy2 = yend + fsy/2
                u0 = Bluestein_dft_xy(u0, fy1, fy2, fsy, num_y)

                fx1 = xstart + fsx/2
                fx2 = xend + fsx/2
                u0 = Bluestein_dft_xy(u0, fx1, fx2, fsx, num_x)
                u0 = F0 * u0 * k_factor  # obtain the complex amplitude of the outgoing light beam

            elif num_x > 1 and num_y == 1:
                u0 = self.u * F
                u0 = u0.transpose()

                # one-dimensional FFT in the other direction
                fx1 = xstart + fsx/2
                fx2 = xend + fsx/2
                u0 = Bluestein_dft_xy(u0, fx1, fx2, fsx, num_x)
                
                fy1 = ystart + fsy/2
                fy2 = yend + fsy/2
                u0 = Bluestein_dft_xy(u0, fy1, fy2, fsy, num_y)

                u0 = F0 * u0* k_factor  # obtain the complex amplitude of the outgoing light beam

            elif num_x == 1 and num_y > 1:
                u0 = self.u * F
                fy1 = ystart + fsy/2
                fy2 = yend + fsy/2
                u0 = Bluestein_dft_xy(u0, fy1, fy2, fsy, num_y)

                fx1 = xstart + fsx/2
                fx2 = xend + fsx/2
                u0 = Bluestein_dft_xy(u0, fx1, fx2, fsx, num_x)
                

                # one-dimensional FFT in the other direction
                u0 = F0 * u0.transpose() * k_factor  # obtain the complex amplitude of the outgoing light beam

            u0 = u0.squeeze()

            if num_x == 1 and num_y == 1:
                # just 1 number
                return 1j*u0.mean()

            elif num_x > 1 and num_y == 1:
                u_out = Scalar_field_X(xout, self.wavelength)
                u_out.u = 1j*u0.transpose()[:, 0]
                return u_out

            elif num_x == 1 and num_y > 1:
                u_out = Scalar_field_X(yout, self.wavelength)
                u_out.u = 1j*u0[:, 0]

                return u_out

            elif num_x > 1 and num_y > 1:
                from diffractio.scalar_fields_XY import Scalar_field_XY
                u_out = Scalar_field_XY(xout, yout, self.wavelength)
                u_out.u = 1j*u0
                return u_out

        elif num_z > 1:
            u_zs = np.zeros((num_y, num_x, num_z), dtype=complex)
            u_zs = u_zs.squeeze()
            Xout, Yout = np.meshgrid(xout, yout)

            for i, z_now in enumerate(z):
                if verbose is True:
                    print("{}/{}".format(i, num_z), sep="\r", end="\r")

                R1 = np.sqrt(Xout**2 + Yout**2 + z_now**2)
                R2 = np.sqrt(self.X**2 + self.Y**2 + z_now**2)

                if z_now > 0:
                    F0 = 1 / (2 * np.pi) * np.exp(
                        1.j * k * R1) * z_now / R1**2 * (1 / R1 - 1.j * k)
                    F = 1 / (2 * np.pi) * np.exp(
                        1.j * k * R2) * z_now / R2**2 * (1 / R2 - 1.j * k)
                else:
                    F0 = 1 / (2 * np.pi) * np.exp(
                        -1.j * k * R1) * z_now / R1**2 * (1 / R1 + 1.j * k)
                    F = 1 / (2 * np.pi) * np.exp(
                        -1.j * k * R2) * z_now / R2**2 * (1 / R2 + 1.j * k)

                u0 = self.u * F

                # one-dimensional FFT in one direction
                fsx = self.wavelength * z_now / dx
                fsy = self.wavelength * z_now / dy

                # if num_x > 1 and num_y == 1:
                #     fx1 = xstart + fsx/2
                #     fx2 = xend + fsx/2
                #     u0 = Bluestein_dft_xy(u0, fx1, fx2, fsx, num_x)

                #     fy1 = ystart + fsy/2
                #     fy2 = yend + fsy/2
                #     u0 = Bluestein_dft_xy(u0, fy1, fy2, fsy, num_y)

                if num_x > 1 and num_y == 1:
                    fy1 = ystart + fsy/2
                    fy2 = yend + fsy/2
                    u0 = Bluestein_dft_xy(u0, fy1, fy2, fsy, num_y)
                    u0=u0.transpose()

                    fx1 = xstart + fsx/2
                    fx2 = xend + fsx/2
                    u0 = Bluestein_dft_xy(u0, fx1, fx2, fsx, num_x)


                elif num_x == 1 and num_y > 1:
                    fy1 = ystart + fsy/2
                    fy2 = yend + fsy/2
                    u0 = Bluestein_dft_xy(u0, fy1, fy2, fsy, num_y)
                    fx1 = xstart + fsx/2
                    fx2 = xend + fsx/2
                    u0 = Bluestein_dft_xy(u0, fx1, fx2, fsx, num_x)
                    u0 = u0.transpose()

                elif num_x == 1 and num_y == 1:
                    fy1 = ystart + fsy/2
                    fy2 = yend + fsy/2
                    u0 = Bluestein_dft_xy(u0, fy1, fy2, fsy, num_y)

                    fx1 = xstart + fsx/2
                    fx2 = xend + fsx/2
                    u0 = Bluestein_dft_xy(u0, fx1, fx2, fsx, num_x)

                elif num_x > 1 and num_y > 1:
                    fy1 = ystart + fsy/2
                    fy2 = yend + fsy/2
                    u0 = Bluestein_dft_xy(u0, fy1, fy2, fsy, num_y)

                    fx1 = xstart + fsx/2
                    fx2 = xend + fsx/2
                    u0 = Bluestein_dft_xy(u0, fx1, fx2, fsx, num_x)
                
                u0 = F0 * u0
                k_factor = z_now * dx * dy * self.wavelength

                if num_x == 1 and num_y == 1:
                    u_zs[i] = u0.mean() * k_factor
                elif num_x > 1 and num_y == 1:
                    u_zs[:, i] = u0[0,:] * k_factor
                elif num_x == 1 and num_y > 1:
                    u_zs[:, i] = u0[:,0] * k_factor # cambiado
                elif num_x > 1 and num_y > 1:
                    u_zs[:, :, i] = u0 * k_factor

            if num_x == 1 and num_y == 1:
                u_out = Scalar_field_Z(z, self.wavelength)
                u_out.u = 1j*u_zs
                return u_out

            elif num_x > 1 and num_y == 1:
                u_out = Scalar_field_XZ(xout, z, self.wavelength)
                u_out.u = 1j*u_zs.transpose()
                return u_out

            elif num_x == 1 and num_y > 1:
                u_out = Scalar_field_XZ(yout, z, self.wavelength)
                u_out.u = 1j*u_zs.transpose()
                return u_out

            elif num_x > 1 and num_y > 1:
                from diffractio.scalar_fields_XYZ import Scalar_field_XYZ
                u_out = Scalar_field_XYZ(xout, yout, z, self.wavelength)
                u_out.u = 1j*u_zs
                return u_out

        return 1j*u_out


    @check_none('x', 'y', 'u', raise_exception=bool_raise_exception)
    def profile(self,
                point1: tuple[float, float] | str = '',
                point2: tuple[float, float] | str = '',
                npixels: int | None = None,
                kind: str = 'intensity',
                order: int = 2):
        """Determine profile in image. If points are not given, then image is shown and points are obtained clicking.

        Args:
            point1 (float, float): initial point. if '' get from click
            point2 (float, float): final point. if '' get from click
            npixels (int): number of pixels for interpolation
            kind (str): type of drawing: 'amplitude', 'intensity', 'phase'
            order (int): order for interpolation

        Returns:
            numpy.np.array: profile
            numpy.np.array: z values for profile
            (float, float): point1
            (float, float): point2
        """
        
        if npixels is None:
            npixels = len(self.x)

        if point1 == '' or point2 == '':
            self.draw(kind=kind)
            print("coordinates to given: click twice")
            point1, point2 = plt.ginput(2)

        x1, y1 = point1
        x2, y2 = point2

        ix1, value, distance = nearest(self.x, x1)
        ix2, value, distance = nearest(self.x, x2)
        iy1, value, distance = nearest(self.y, y1)
        iy2, value, distance = nearest(self.y, y2)

        x = np.linspace(ix1, ix2, npixels)
        y = np.linspace(iy1, iy2, npixels)

        if kind == 'intensity':
            image = np.abs(self.u)**2
        elif kind == 'amplitude':
            image = np.real(self.u)
        elif kind == 'phase':
            image = np.angle(self.u)  # / np.pi
            image[image == 1] = -1

        h = np.linspace(0, np.sqrt((y2 - y1)**2 + (x2 - x1)**2), npixels)
        h = h - h[-1]/2

        z_profile = scipy.ndimage.map_coordinates(image.transpose(),   np.vstack((x, y)),   order=order)
        z_profile[-1] = z_profile[-2]

        return h, z_profile, point1, point2

    @check_none('x', raise_exception=bool_raise_exception)
    def draw_profile(self,
                     point1: tuple[float, float] | str = '',
                     point2: tuple[float, float] | str = '',
                     npixels: int | None = None,
                     kind: str = 'intensity',
                     order: int = 2):
        """Draws profile in image. If points are not given, then image is shown and points are obtained clicking.

        Args:
            point1 (float): initial point. if '' get from click
            point2 (float): final point. if '' get from click
            npixels (int): number of pixels for interpolation
            kind (str): type of drawing: 'amplitude', 'intensity', 'phase'
            order (int): order for interpolation

        Returns:
            numpy.np.array: profile
            numpy.np.array: z values for profile
            (float, float): point1
            (float, float): point2
        """

        if npixels is None:
            npixels = len(self.x)

        h, z_profile, point1, point2 = self.profile(point1, point2, npixels, kind, order)

        plt.figure()
        plt.plot(h, z_profile, 'k', lw=2)
        plt.xlabel('h (profile)')
        plt.ylabel(kind)
        plt.axis([h.min(), h.max(), z_profile.min(), z_profile.max()])
        return h, z_profile, point1, point2


    def get_edges(self,
                  kind_transition: str = 'amplitude',
                  min_step: float = 0,
                  verbose: bool = False,
                  filename: str = ''):
        """
        Determine locations of edges for a binary mask. Valid for litography engraving of gratings.

        Args:
            kind_transition:'amplitude' 'phase'.
            min_step: minimum step for consider a transition

        Returns:
            type_transition: np.array with +1, -1 with rasing or falling edges
            pos_transition: positions x of transitions
            raising: positions of raising
            falling: positions of falling
        """

        pos_transitions, type_transitions, raising, falling = get_edges(
            self.x, self.u, kind_transition, min_step, verbose, filename)
        return pos_transitions, type_transitions, raising, falling


    @check_none('x', 'y', 'u', raise_exception=bool_raise_exception)
    def search_focus(self, kind: str = 'moments', verbose: bool = True):
        """Search for location of maximum intensity.

        Args:
            kind (str): 'moments' or 'maximum'
            verbose (bool): If True prints information.

        Returns:
            (x,y): positions of focus
        """

        if kind == 'maximum':
            intensity = np.abs(self.u)**2
            ix, iy = np.unravel_index(intensity.argmax(), intensity.shape)
            pos_x, pos_y = self.x[ix], self.y[iy]
        elif kind == 'moments':
            _, _, _, moments = beam_width_2D(self.x,
                                             self.y,
                                             np.abs(self.u)**2,
                                             has_draw=False)
            pos_x, pos_y, _, _, _ = moments

        if verbose is True:
            print(("x = {:2.3f} um, y = {:2.3f} um".format(pos_x, pos_y)))

        return pos_x, pos_y


    @check_none('x', 'y', 'u', raise_exception=bool_raise_exception)
    def MTF(self, fx: np.ndarray, fy: np.ndarray, incoherent: bool = False, has_draw: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        
        """Compute 2D MTF using CZT for arbitrary frequency grids.
        
        Args:
            fx: 1D array of desired fx frequency samples.  The frequencies are in cycles/mm.
            fy: 1D array of desired fy frequency samples. The frequencies are in cycles/mm.
            incoherent: If True, compute MTF from intensity; else from complex field.
            has_draw: If True, plot the resulting 2D MTF as an image.
            Returns:
                mtf2d: 2D array of MTF values (My x Mx).
        """

        fx = fx/1000  # convert from cycles/um to cycles/mm
        fy = fy/1000  # convert from cycles/um to cycles/mm

        Nx = len(self.x)
        Ny = len(self.y)

        # prepare intensity (incoherent) or complex field (coherent)
        if incoherent:
            s = np.abs(self.u)**2
        else:
            s = self.u.astype(np.complex128)

        # check uniform grids and compute spacing and origins
        x = self.x 
        y = self.y
        if x.size != Nx or y.size != Ny:
            raise ValueError('x and y lengths must match field shape (Nx, Ny)')
        dxs = np.diff(x)
        dys = np.diff(y)
        dx = dxs.mean()
        dy = dys.mean()
        if not np.allclose(dxs, dx, rtol=1e-6, atol=1e-12):
            raise ValueError('x must be (approximately) uniformly sampled')
        if not np.allclose(dys, dy, rtol=1e-6, atol=1e-12):
            raise ValueError('y must be (approximately) uniformly sampled')
        x0 = x[0]
        y0 = y[0]

        dfx = fx[1] - fx[0]
        dfy = fy[1] - fy[0]
        Mx = fx.size
        My = fy.size
        fx0 = fx[0]
        fy0 = fy[0]

        # CZT parameters for x-axis: w_x = exp(-2j*pi*dfx*dx) ; a_x = 1 (we handle x0 prefactor separately)
        w_x = np.exp(-2j * np.pi * dfx * dx)
        a_x = 1.0
        # CZT along x for each row: compute Mx freq samples per row
        # precompute phase factor for x0: pref_x[k] = exp(-2j*pi*(fx0 + k*dfx)*x0)
        pref_x = np.exp(-2j * np.pi * (fx0 + np.arange(Mx) * dfx) * x0)

        # Allocate intermediate array (Ny x Mx)
        Sx = np.zeros((Ny, Mx), dtype=complex)
        # For each row, compute CZT of sequence s[row, :] * exp(-2j*pi*fx0 * n * dx) with w = w_x and m = Mx
        n = np.arange(Nx)
        phase_f0_x = np.exp(-2j * np.pi * fx0 * n * dx)
        for iy in range(Ny):
            row = s[iy, :] * phase_f0_x
            Srow = czt(row, Mx, w=w_x, a=1.0)  # returns length Mx array
            Sx[iy, :] = pref_x * Srow  # apply x0 prefactor across k

        # Now for each fx index k we have a column Sx[:, k] as function of y; apply CZT along y to get fy samples
        w_y = np.exp(-2j * np.pi * dfy * dy)
        pref_y = np.exp(-2j * np.pi * (fy0 + np.arange(My) * dfy) * y0)
        mtf2d = np.zeros((My, Mx), dtype=float)
        m = np.arange(Ny)
        phase_f0_y = np.exp(-2j * np.pi * fy0 * m * dy)
        for kx in range(Mx):
            col = Sx[:, kx] * phase_f0_y  # length Ny complex
            # czt along y produces My samples for this kx
            Sy = czt(col, My, w=w_y, a=1.0)
            # apply y0 prefactor and store magnitude later
            Sy = pref_y * Sy
            mtf2d[:, kx] = np.abs(Sy)

        # normalize so OTF at (0,0) = 1 if (0,0) is inside sampled grid
        # find index of fx=0 and fy=0 if present (within tolerance) to normalize; otherwise normalize to max
        ix0 = np.argmin(np.abs(fx - 0.0)) if np.any(np.isclose(fx, 0.0, atol=1e-12)) else None
        iy0 = np.argmin(np.abs(fy - 0.0)) if np.any(np.isclose(fy, 0.0, atol=1e-12)) else None
        if ix0 is not None and iy0 is not None:
            norm = mtf2d[iy0, ix0] if mtf2d[iy0, ix0] != 0 else mtf2d.max()
        else:
            norm = mtf2d.max() if mtf2d.max() != 0 else 1.0
        mtf2d = mtf2d / norm


        if has_draw:
            """Plot 2D MTF as image with fx horizontal axis (cycles/unit) and fy vertical."""
            extent = (fx[0]*1000, fx[-1]*1000, fy[0]*1000, fy[-1]*1000)
            plt.figure(figsize=(6,5))
            plt.imshow(mtf2d, origin='lower', extent=extent, aspect='equal', cmap='hot', vmin=0, vmax=1)
            plt.xlabel('fx (cycles / mm)')
            plt.ylabel('fy (cycles / mm)')
            plt.title('MTF')
            plt.show()

        return mtf2d



    def MTF_profiles(self, fx: np.ndarray, fy: np.ndarray, incoherent: bool = False, has_draw: bool = True, mtf_ideal: Tuple[np.ndarray, np.ndarray] = None) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """Return central horizontal and vertical cuts through the 2D MTF and a radial average."""
       
        mtf2d = self.MTF(fx, fy, incoherent=incoherent, has_draw=False)
       
       
        # find central indices (closest to zero freq) if 0 included; else central index
        ix0 = np.argmin(np.abs(fx - 0.0)) if np.any(np.isclose(fx, 0.0, atol=1e-12)) else fx.size // 2
        iy0 = np.argmin(np.abs(fy - 0.0)) if np.any(np.isclose(fy, 0.0, atol=1e-12)) else fy.size // 2
        horiz = mtf2d[iy0, :]  # MTF vs fx at fy ~ 0
        vert = mtf2d[:, ix0]   # MTF vs fy at fx ~ 0
        # radial average
        FX, FY = np.meshgrid(fx, fy)
        FR = np.sqrt(FX**2 + FY**2)
        fr_flat = FR.ravel()
        mtf_flat = mtf2d.ravel()
        fmax = fr_flat.max()
        nbins = max(200, int(min(fx.size, fy.size)))
        bins = np.linspace(0, fmax, nbins+1)
        frad = 0.5*(bins[:-1] + bins[1:])
        mtf_rad = np.empty(nbins)
        for i in range(nbins):
            sel = (fr_flat >= bins[i]) & (fr_flat < bins[i+1])
            if np.any(sel):
                mtf_rad[i] = mtf_flat[sel].mean()
            else:
                mtf_rad[i] = np.nan
        
        if np.isnan(mtf_rad[0]):
            mtf_rad[0] = mtf_rad[1]

        mtf_rad = np.nan_to_num(mtf_rad)


        if has_draw:
            plt.figure()
            plt.plot(fx, horiz, 'r', label='Horizontal cut (fy=0)')
            plt.plot(fy, vert, 'b', label='Vertical cut (fx=0)')
            if mtf_ideal is not None:
                freqs_ideal, mtf_ideal_vals = mtf_ideal
                plt.plot(freqs_ideal, mtf_ideal_vals, 'k--', label='MTF ideal')
            plt.xlabel('Frequency (cycles/mm)')
            plt.ylabel('MTF')
            plt.title('1D MTF Profiles')
            plt.xlim(0, max(fx[-1], fy[-1]))
            plt.ylim(-0.01, 1.01)
            plt.grid()
            plt.legend()


            plt.figure()
            plt.plot(frad, mtf_rad, 'b-', label='radial average')
            if mtf_ideal is not None:
                freqs_ideal, mtf_ideal_vals = mtf_ideal
                plt.plot(freqs_ideal, mtf_ideal_vals, 'k--', label='MTF ideal')
            plt.xlabel('Spatial Frequency (cycles/mm)')
            plt.ylabel('Radial Average MTF')
            plt.xlim(0, frad[-1])
            plt.ylim(-0.01, 1.01)
            plt.grid()
            plt.legend()


        return (fx, horiz), (fy, vert), (frad, mtf_rad)



    @check_none('x', 'y', 'u', raise_exception=bool_raise_exception)
    def beam_width_4s(self, has_draw: bool = True, verbose: bool = False):
        """Returns the beam width parameters according to ISO11146.

        Args:
            has_draw (bool): If True, it draws
            verbose (bool): If True, it prints data

        Returns:
            (float): dx width x
            (float): dy width y
            (float): principal_axis, angle
            (str): (x_mean, y_mean, x2_mean, y2_mean, xy_mean), Moments

        References:

            * https://en.wikipedia.org/wiki/Beam_diameter
            * http://www.auniontech.com/ueditor/file/20170921/1505982360689799.pdf
        """
            
        dx, dy, principal_axis, (x_mean, y_mean, x2_mean, y2_mean,
                                 xy_mean) = beam_width_2D(self.x, self.y, np.abs(self.u)**2, has_draw=False)

        if has_draw is True:
            from matplotlib.patches import Ellipse

            self.draw()
            ellipse = Ellipse(xy=(x_mean, y_mean),
                              width=dy,
                              height=dx,
                              angle=-principal_axis/degrees)
            ellipse2 = Ellipse(xy=(x_mean, y_mean),
                               width=dy/2,
                               height=dx/2,
                               angle=-principal_axis/degrees)

            ellipse3 = Ellipse(xy=(x_mean, y_mean),
                               width=dy / 4,
                               height=dx / 4,
                               angle=-principal_axis/degrees)

            ax = plt.gca()
            ax.add_artist(ellipse)
            ellipse.set_clip_box(ax.bbox)
            ellipse.set_facecolor('none')
            ellipse.set_alpha(0.75)
            ellipse.set_edgecolor('yellow')
            ellipse.set_linewidth(1.5)

            ax.add_artist(ellipse2)
            ellipse2.set_clip_box(ax.bbox)
            ellipse2.set_facecolor('none')
            ellipse2.set_alpha(0.75)
            ellipse2.set_edgecolor('red')
            ellipse2.set_linewidth(1.5)

            ax.add_artist(ellipse3)
            ellipse3.set_clip_box(ax.bbox)
            ellipse3.set_facecolor('none')
            ellipse3.set_alpha(0.75)
            ellipse3.set_edgecolor('black')
            ellipse3.set_linewidth(1.5)

            x0 = self.x[0]
            y0 = self.y[0]
            plt.plot(x0, y0, 'yellow', label=r'4$\sigma$')
            plt.plot(x0, y0, 'red', label=r'2$\sigma$')
            plt.plot(x0, y0, 'black', label=r'1$\sigma$')
            plt.legend()

        if verbose is True:
            print(f"Beam waist (1s): dx = {dx/um:.2f} um, dy = {dy/um:.2f} um")
            print(f"Principal axis angle: {principal_axis/degrees:.2f} degrees")
            print(("x_mean = {:2.3f} um, y_mean = {:2.3f} um").format(
                x_mean, y_mean))
            print(("x2_mean = {:2.3f} um^2, y2_mean = {:2.3f} um^2, "
                   "xy_mean = {:2.3f} um^2").format(
                       x2_mean, y2_mean, xy_mean))

        return dx, dy, principal_axis, (x_mean, y_mean, x2_mean, y2_mean,
                                        xy_mean)


    @check_none('u', raise_exception=bool_raise_exception)
    def intensity(self):
        """Returns intensity."""
        
        intensity = np.abs(self.u)**2
        return intensity

    @check_none('u', raise_exception=bool_raise_exception)
    def average_intensity(self, verbose: bool = False):
        """Returns average intensity as: (np.abs(self.u)**2).sum() / num_data.

        Args:
            verbose(bool): If True prints data.
        """
        
        average_intensity = (np.abs(self.u)**2).mean()
        if verbose is True:
            print(("average intensity={} W/m").format(average_intensity))

        return average_intensity


    @check_none('u', raise_exception=bool_raise_exception)
    def send_image_screen(self, id_screen: int, kind: str = 'amplitude'):
        """Takes the images and sends the images to a screen in full size.

        Args:
            id_screen(hdl): handle to screen
            kind('str'): 'amplitude', 'intensity', 'phase'
        """

        amplitude, intensity, phase = field_parameters(self.u)

        if kind == 'amplitude':
            image = amplitude
        elif kind == 'intensity':
            image = intensity
        elif kind == 'phase':

            phase = (phase + np.pi) % (2 * np.pi) - np.pi

            image = phase + np.pi
            image[0, 0] = 0
            image[0, 1] = 2 * np.pi
            image = image / (2 * np.pi)

        print(("send_image_screen: max={}. min={}".format(
            image.max(), image.min())))

        screen = screeninfo.get_monitors()[id_screen]
        window_name = 'projector'
        cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
        cv2.moveWindow(window_name, screen.x - 1, screen.y - 1)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN)
        cv2.imshow(window_name, image)
        cv2.waitKey()
        cv2.destroyAllWindows()

    @check_none('u', raise_exception=bool_raise_exception)
    def get_amplitude(self, matrix: bool = False, new_field: bool = False):
        """Gets the amplitude of the field.

        Args:
            matrix (bool): if True numpy.matrix is returned
            new_field (bool): if True it returns a new Scalar_field_XY

        Returns:
            if New_field is True: Scalar_field_X
            if matrix is True: numpy.np.array
        """
        amplitude = abs(self.u)

        if matrix is True:
            return amplitude

        if new_field is True:
            u_salida = Scalar_field_XY(self.x, self.y, self.wavelength)
            u_salida.u = amplitude
            return u_salida

        else:
            self.u = amplitude


    @check_none('u', raise_exception=bool_raise_exception)
    def get_phase(self, matrix: bool = False, new_field: bool = False):
        """Gets the phase of the field.

        Args:
            matrix(bool): if True numpy.matrix is returned
            new_field(bool): if True it returns a new Scalar_field_XY

        Returns:
            if New_field is True: Scalar_field_X.
            if Matrix is True: numpy.np.array.
        """
        phase = np.exp(1j * np.angle(self.u))

        if matrix is True:
            return phase

        if new_field is True:
            u_salida = Scalar_field_XY(self.x, self.y, self.wavelength)
            u_salida.u = phase
            return u_salida

        else:
            self.u = phase

    @check_none('u', raise_exception=bool_raise_exception)
    def remove_phase(self, sign: bool = False, matrix: bool = False, new_field: bool = False):
        """Removes the phase of the field. Amplitude is kept.

        Args:
            sign (bool): If True, sign is kept, else, it is removed
            matrix (bool): if True numpy.matrix is returned
            new_field (bool): if True it returns a new Scalar_field_XY

        Returns:
            if New_field is True: Scalar_field_X.
            if Matrix is True: numpy.np.array.
        """

        amplitude = np.abs(self.u)
        phase = np.angle(self.u)

        if sign is False:
            only_amplitude = amplitude
        elif sign is True:
            only_amplitude = np.sign(phase) * amplitude

        if matrix is True:
            return only_amplitude

        if new_field is True:
            u_salida = Scalar_field_XY(self.x, self.y, self.wavelength)
            u_salida.u = only_amplitude
            return u_salida

        else:
            self.u = only_amplitude


    @check_none('u', raise_exception=bool_raise_exception)
    def binarize(self,
                 kind: str = "amplitude",
                 bin_level: float | None = None,
                 level0: float | None = None,
                 level1: float | None = None,
                 new_field: bool = False,
                 matrix: bool = False):
        """Changes the number of points in field, mantaining the area.

        Args:
            kind (str): 'amplitude' or 'phase'
            bin_level (float): value of cut. If None, the cut is in the mean value
            level0 (float): minimum value. If None, minimum value of field
            level1 (float): maximum value. If None, maximum value of field
            new_field (bool): if True returns new field
            matrix (bool): if True it returs a matrix

        Returns:
            Scalar_field_XY: if new_field is True returns Scalar_field_XY

        TODO: Check and pass to utils
        """

        amplitude = self.get_amplitude(matrix=True, new_field=False)
        phase = self.get_phase(matrix=True, new_field=False)

        if kind == 'amplitude':
            amplitude_binarized = amplitude
            maximum = amplitude.max()
            minimum = amplitude.min()
            if bin_level is None:
                bin_level = (maximum + minimum)/2
            if level0 is None:
                level0 = minimum
            if level1 is None:
                level1 = maximum

            amplitude_binarized[amplitude <= bin_level] = level0
            amplitude_binarized[amplitude > bin_level] = level1
            u_binarized = amplitude_binarized * phase

        if kind == 'phase':
            phase_binarized = phase
            maximum = phase.max()
            minimum = phase.min()
            if bin_level is None:
                bin_level = (maximum + minimum)/2
            if level0 is None:
                level0 = minimum
            if level1 is None:
                level1 = maximum

            phase_binarized[phase <= bin_level] = level0
            phase_binarized[phase > bin_level] = level1
            u_binarized = amplitude * phase_binarized

        if new_field is False and matrix is False:
            self.u = u_binarized
            return self.u

        if new_field is False and matrix is True:
            return u_binarized

        if new_field is True:
            cn = Scalar_field_XY(self.x, self.y, self.wavelength)
            cn.u = u_binarized
            return cn

    @check_none('u', raise_exception=bool_raise_exception)
    def discretize(self,
                   kind: str = 'amplitude',
                   num_levels: int = 2,
                   factor: float = 1,
                   phaseInicial: float = 0,
                   new_field: bool = True,
                   matrix: bool = False):
        """Discretize in a number of levels equal to num_levels.

        Args:
            kind (str): "amplitude" o "phase"
            num_levels (int): number of levels for the discretization
            factor (float): from the level, how area is binarized. if 1 everything is binarized,
            phaseInicial (float): *
            new_field (bool): if True returns new field
            matrix (bool): if True it returs a matrix

        Returns:
            Scalar_field_XY: if new_field is True returns Scalar_field_XY

        TODO: Check and pass to utils
        """

        if kind == 'amplitude':
            heights = np.linspace(0, 1, num_levels)
            posX = 256 / num_levels

            amplitude = self.get_amplitude(matrix=True, new_field=False)
            phase = self.get_phase(matrix=True, new_field=False)
            discretized_image = amplitude

            dist = factor * posX

            for i in range(num_levels):
                center = posX/2 + i * posX
                abajo = amplitude * 256 > center - dist/2
                arriba = amplitude * 256 <= center + dist/2
                Trues = abajo * arriba
                discretized_image[Trues] = center/256

            u_binarized = discretized_image * phase

        if kind == 'phase':
            ang = np.angle(self.get_phase(matrix=True, new_field=False)) + phaseInicial + np.pi
            ang = ang % (2 * np.pi)
            amplitude = self.get_amplitude(matrix=True, new_field=False)

            heights = np.linspace(0, 2 * np.pi, num_levels + 1)

            dist = factor * (heights[1] - heights[0])

            discretized_image = np.exp(1j * ang)

            for i in range(num_levels + 1):
                center = heights[i]
                abajo = ang > (center - dist/2)
                arriba = ang <= (center + dist/2)
                Trues = abajo * arriba
                discretized_image[Trues] = np.exp(1j * center)  # - np.pi

            Trues = ang > (center + dist/2)
            discretized_image[Trues] = np.exp(1j * heights[0])  # - np.pi

            phase = np.angle(discretized_image) / np.pi
            phase[phase == 1] = -1
            phase = phase - phase.min()
            discretized_image = np.exp(1j * np.pi * phase)

            u_binarized = amplitude * discretized_image

        if new_field is False and matrix is False:
            self.u = u_binarized
            return

        if new_field is True:
            cn = Scalar_field_XY(self.x, self.y, self.wavelength)
            cn.u = u_binarized
            return cn

        if matrix is True:
            return u_binarized


    @check_none('u', raise_exception=bool_raise_exception)
    def normalize(self, kind='amplitude', new_field: bool = False):
        """Normalizes the field so that intensity.max()=1.

        Args:
            kind (str): 'amplitude', or 'intensity'
            new_field (bool): If False the computation goes to self.u. If True a new instance is produced

        Returns
            u (numpy.array): normalized optical field
        """
        return normalize_field(self, kind, new_field)

    @check_none('x', 'y', 'n')
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
    
    
    @check_none('x', 'y', 'u', raise_exception=bool_raise_exception)
    def get_RS_minimum_z(self, n: float = 1., quality: float = 1., verbose=True):
        """Determines the minimum available distance for RS algorithm. If higher or lower quality parameters is required you can add as a parameter

            Args:
                n (float): refractive index of the surrounding medium.
                quality (int, optional): quality. Defaults to 1.
                verbose (bool, optional): prints info. Defaults to True.

            Returns:
                z_min (float): z_min for quality_factor>quality
            """
        
        range_x = self.x[-1] - self.x[0]
        range_y = self.y[-1] - self.y[0]
        num_x = len(self.x)
        num_y = len(self.y)

        dx = range_x / num_x
        dy = range_y / num_y
        dr_real = np.sqrt(dx**2 + dy**2)
        rmax = np.sqrt(range_x**2 + range_y**2)

        factor = (((quality * dr_real + rmax)**2 -
                   (self.wavelength / n)**2 - rmax**2)/2 * n /
                  self.wavelength)**2 - rmax**2

        if factor > 0:
            z_min = np.sqrt(factor)
            z_min = z_min/2
        else:
            z_min = 0

        if verbose:
            if z_min > 1000:
                print("z min = {:2.2f} mm".format(z_min / mm))
            else:
                print("z min = {:2.2f} um".format(z_min))

        return z_min


    @check_none('x', 'y', 'u', raise_exception=bool_raise_exception)
    def draw(self,
             kind: Draw_XY_Options = 'intensity',
             logarithm: float = 0.,
             normalize: bool = False,
             title: str = "",
             filename: str = '',
             cut_value: float | None = None,
             has_colorbar: str = '',
             colormap_kind: str = '',
             reduce_matrix: str = 'standard',
             percentage_intensity: float | None = None,
             **kwargs):
        """Draws  XY field.

        Args:
            kind (str): type of drawing: 'amplitude', 'intensity', 'phase', ' 'field', 'real_field', 'refractive_index'
            logarithm (float): If >0, intensity is scaled in logarithm
            normalize (str):  False, 'maximum', 'area', 'intensity'
            title (str): title for the drawing
            filename (str): if not '' stores drawing in file,
            cut_value (float): if provided, maximum value to show
            has_colorbar (bool): if True draws the colorbar
            percentage_intensity (None or number): If None it takes from CONF_DRAWING['percentage_intensity'], else uses this value
            reduce_matrix (str): 'standard'
        """

        if reduce_matrix in ([], None, ''):
            pass
        else:
            self.reduce_matrix = reduce_matrix

        if kind == 'intensity':
            id_fig, IDax, IDimage = self.__draw_intensity__(
                logarithm, normalize, title, cut_value, colormap_kind,
                **kwargs)
        elif kind == 'amplitude':
            id_fig, IDax, IDimage = self.__draw_amplitude__(
                logarithm, normalize, title, cut_value, colormap_kind,
                **kwargs)
        elif kind == 'phase':
            id_fig, IDax, IDimage = self.__draw_phase__(
                title, colormap_kind, percentage_intensity, **kwargs)
        elif kind == 'field':
            id_fig = self.__draw_field__(logarithm, normalize, title,
                                         cut_value, colormap_kind,
                                         percentage_intensity, **kwargs)
            IDax = None
            IDimage = None
        elif kind == 'real_field':
            id_fig, IDax, IDimage = self.__draw_real_field__(
                logarithm, normalize, title, cut_value, **kwargs)
            
        elif kind == 'refractive_index':
            id_fig, IDax, IDimage = self.__draw_refractive_index__(title, cut_value, **kwargs)
        else:
            print("Accepted values:", Draw_XY_Options)

        if has_colorbar in ('horizontal', 'vertical'):
            plt.colorbar(orientation=has_colorbar, shrink=0.5)

        plt.tight_layout()
        if filename != '':
            plt.savefig(filename,
                        dpi=100,
                        bbox_inches='tight',
                        pad_inches=0.05)

        return id_fig, IDax, IDimage


    @check_none('x', 'y', 'u', raise_exception=bool_raise_exception)
    def __draw_intensity__(self,
                           logarithm: float = 0.,
                           normalize='maximum',
                           title: str = "",
                           cut_value: float | None = None,
                           colormap_kind: str = '',
                           **kwargs):
        """Draws intensity  XY field.

        Args:
            logarithm (float): If >0, intensity is scaled in logarithm
            normalize (str):  False, 'maximum', 'area', 'intensity'
            title (str): title for the drawing
            cut_value (float): if provided, maximum value to show
        """
        
        amplitude, intensity, phase = field_parameters(self.u,        has_amplitude_sign=True)
        if colormap_kind in ['', None, []]:
            colormap_kind = self.CONF_DRAWING["color_intensity"]
        intensity = normalize_draw(intensity, logarithm, normalize, cut_value)
        id_fig, IDax, IDimage = draw2D(intensity,
                                       self.x,
                                       self.y,
                                       xlabel=r"$x  (\mu m)$",
                                       ylabel=r"$y  (\mu m)$",
                                       title=title,
                                       color=colormap_kind,
                                       reduce_matrix=self.reduce_matrix,
                                       **kwargs)
        plt.tight_layout()

        # if self.type == 'Scalar_mask_XY':
        #     plt.clim(0, 1)

        return id_fig, IDax, IDimage


    def __draw_amplitude__(self,
                          logarithm: float = 0.,
                          normalize='maximum',
                          title='intensity',
                          cut_value=1,
                          colormap_kind: str = '',
                          **kwargs):
        """Draws amplitude  XY field.

        Args:
            logarithm (float): If >0, intensity is scaled in logarithm
            normalize (str):  False, 'maximum', 'area', 'intensity'
            title (str): title for the drawing
            cut_value (float): if provided, maximum value to show
        """
                
        amplitude, intensity, phase = field_parameters(self.u, has_amplitude_sign=True)
        amplitude = normalize_draw(amplitude, logarithm, normalize, cut_value)
        max_amplitude = np.abs(amplitude).max()
        if colormap_kind in ['', None, []]:
            colormap_kind = self.CONF_DRAWING["color_amplitude"]
        id_fig, IDax, IDimage = draw2D(amplitude,
                                       self.x,
                                       self.y,
                                       xlabel=r"$x  (\mu m)$",
                                       ylabel=r"$y  (\mu m)$",
                                       title=title,
                                       color=colormap_kind,
                                       reduce_matrix=self.reduce_matrix,
                                       **kwargs)
        plt.clim(-max_amplitude, max_amplitude)

        return id_fig, IDax, IDimage


    def __draw_phase__(self,
                       title: str = r'phase/np.pi',
                       colormap_kind: str = '',
                       percentage_intensity: float | None = None,
                       **kwargs):
        """Draws phase of  XY field

        Args:
            title (str): title for the drawing
        """
                
        amplitude, intensity, phase = field_parameters(self.u, has_amplitude_sign=True)
        phase[phase == 1] = -1
        phase = phase/degrees

        if percentage_intensity is None:
            percentage_intensity = percentage_intensity_config

        phase[intensity < percentage_intensity * (intensity.max())] = 0

        if colormap_kind in ['', None, []]:
            colormap_kind = self.CONF_DRAWING["color_phase"]

        id_fig, IDax, IDimage = draw2D(phase,
                                       self.x,
                                       self.y,
                                       xlabel=r"$x  (\mu m)$",
                                       ylabel=r"$y  (\mu m)$",
                                       title=title,
                                       color=colormap_kind,
                                       reduce_matrix=self.reduce_matrix,
                                       **kwargs)  # seismic gist_heat
        plt.clim(vmin=-180, vmax=180)

        return id_fig, IDax, IDimage


    def __draw_field__(self,
                       logarithm: float = 0.,
                       normalize: str = 'maximum',
                       title: str = "",
                       cut_value: float  = 0.,
                       colormap_kind: str = '',
                       percentage_intensity: float  = 0.,
                       **kwargs):
        """Draws field  XY field.

        Args:
            logarithm(bool): If >0, intensity is scaled in logarithm
            normalize(str):  False, 'maximum', 'area', 'intensity'
            title(str): title for the drawing
            cut_value(float): if provided, maximum value to show
        """
        
        amplitude, intensity, phase = field_parameters(self.u, has_amplitude_sign=True)
        intensity = reduce_matrix_size(self.reduce_matrix, self.x, self.y, intensity)
        phase = reduce_matrix_size(self.reduce_matrix, self.x, self.y, phase)

        if percentage_intensity is None:
            percentage_intensity = percentage_intensity_config

        phase[intensity < percentage_intensity * (intensity.max())] = 0

        xsize, ysize = rcParams['figure.figsize']

        # plt.figure(figsize=(2 * xsize, 2 * ysize))
        plt.figure()
        plt.suptitle(title)
        extension = [self.x[0], self.x[-1], self.y[0], self.y[-1]]

        intensity = normalize_draw(intensity, logarithm, normalize, cut_value)

        plt.subplot(1, 2, 1)

        h1 = plt.imshow(intensity,
                        interpolation='bilinear',
                        aspect='auto',
                        origin='lower',
                        extent=extension)
        plt.xlabel(r"$x  (\mu m)$")
        plt.ylabel(r"$y  (\mu m)$")
        plt.colorbar(orientation='horizontal', shrink=0.66, pad=0.1)

        plt.axis('scaled')
        plt.axis(extension)
        plt.title("intensity")
        h1.set_cmap(self.CONF_DRAWING["color_intensity"])
        # if self.type == 'Scalar_mask_XY':
        #     plt.clim(0, 1)

        plt.subplot(1, 2, 2)
        phase = phase/degrees

        # elimino la fase en la visualicion cuando no hay campo
        h2 = plt.imshow(phase,
                        interpolation='bilinear',
                        aspect='auto',
                        origin='lower',
                        extent=extension)
        plt.xlabel(r"$x  (\mu m)$")
        plt.ylabel(r"$y  (\mu m)$")
        plt.colorbar(orientation='horizontal', shrink=0.66, pad=0.1)

        plt.axis('scaled')
        plt.axis(extension)
        plt.title("phase")
        plt.clim(-180, 180)
        h2.set_cmap(self.CONF_DRAWING["color_phase"])  #
        plt.subplots_adjust(0.01, 0.01, 0.99, 0.95, 0.35, 0.35)
        return (h1, h2)


    def __draw_real_field__(self,
                            logarithm: float = 0.,
                            normalize: str = 'maximum',
                            cut_value: float | None = 1,
                            title: str = "",
                            colormap_kind: str = '',
                            percentage_intensity: float | None = None,
                            **kwargs):
        """Draws np.real field  XY field.

        Args:
            logarithm(bool): If >0, intensity is scaled in logarithm
            normalize(str):  False, 'maximum', 'area', 'intensity'
            title(str): title for the drawing
            cut_value(float): if provided, maximum value to show
        """
        if percentage_intensity is None:
            percentage_intensity = percentage_intensity_config

        rf = np.real(self.u)
        intensity = np.abs(self.u)**2
        rf[intensity < percentage_intensity * (intensity.max())] = 0

        if colormap_kind in ['', None, []]:
            colormap_kind = self.CONF_DRAWING["color_real"]

        id_fig, IDax, IDimage = draw2D(rf,
                                       self.x,
                                       self.y,
                                       xlabel=r"$x  (\mu m)$",
                                       ylabel=r"$y  (\mu m)$",
                                       title=title,
                                       color=colormap_kind,
                                       reduce_matrix=self.reduce_matrix,
                                       **kwargs)

        return id_fig, IDax, IDimage


    @check_none('x', 'y', 'n', raise_exception=bool_raise_exception)
    def __draw_refractive_index__(self,
                           title: str = "",
                           cut_value: float | None = None,
                           colormap_kind: str = '',
                           **kwargs):
        """Draws refractive index, normally coming from  XYZ field.

        Args:
            logarithm (float): If >0, intensity is scaled in logarithm
            normalize (str):  False, 'maximum', 'area', 'intensity'
            title (str): title for the drawing
            cut_value (float): if provided, maximum value to show
        """
        
        if colormap_kind in ['', None, []]:
            colormap_kind = self.CONF_DRAWING["color_n"]
        id_fig, IDax, IDimage = draw2D(np.abs(self.n),
                                       self.x,
                                       self.y,
                                       xlabel=r"$x  (\mu m)$",
                                       ylabel=r"$y  (\mu m)$",
                                       title=title,
                                       color=colormap_kind,
                                       reduce_matrix=self.reduce_matrix,
                                       **kwargs)
        plt.tight_layout()

        # if self.type == 'Scalar_mask_XY':
        #     plt.clim(0, 1)

        return id_fig, IDax, IDimage



    def video(self,
              kind: str,
              zs: NDArrayFloat,
              logarithm: float = 0.,
              normalize: bool = False,
              time_video: float = 10 * seconds,
              frames_reduction: int = 1,
              filename: str = 'video.avi',
              dpi: int = 100):
        """Makes a video

        Args:
            kind(str): 'intensity', 'phase', 'amplitude'
        """
                
        fig = plt.figure()
        ax = fig.add_subplot(111, autoscale_on=False)
        ax.grid()
        plt.xlim(self.x[0], self.x[-1])
        plt.ylim(self.y[0], self.y[-1])

        def animate(i):
            t2 = self.RS(z=zs[i], new_field=True)

            image = reduce_matrix_size(self.reduce_matrix, self.x, self.y, t2.u)

            I_drawing = prepare_drawing(image, kind, logarithm, normalize)
            ax.imshow(I_drawing)
            ax.set_title(r"$z = {:2.0f} \mu m$".format(zs[i]))
            return i

        ani = animation.FuncAnimation(fig, animate, list(range(0, len(zs), frames_reduction)),
                                      interval=25, blit=False)

        fps = int(len(zs) / (time_video * frames_reduction))

        ani.save(filename, fps=fps, dpi=dpi)



def kernelRS(X: NDArrayFloat, Y: NDArrayFloat, wavelength: float, z: float,
             n: float = 1., kind: str = 'z'):
    """Kernel for RS propagation. 

    Args:
        X(numpy.np.array): positions x
        Y(numpy.np.array): positions y
        wavelength(float): wavelength of incident fields
        z(float): distance for propagation
        n(float): refractive index of background
        kind(str): 'z', 'x', '0': for simplifying vector propagation

    Returns:
        complex np.array: kernel
    """
    
    k = 2 * np.pi * n / wavelength
    R = np.sqrt(X**2 + Y**2 + z**2)
    if kind == 'z':
        return 1 / (2 * np.pi) * np.exp(1.j * k * R) * z / R**2 * (1 / R - 1.j * k)
    elif kind == 'x':
        return 1 / (2 * np.pi) * np.exp(1.j * k * R) * X / R**2 * (1 / R - 1.j * k)
    elif kind == 'y':
        return 1 / (2 * np.pi) * np.exp(1.j * k * R) * Y / R**2 * (1 / R - 1.j * k)
    elif kind == '0':
        return 1 / (2 * np.pi) * np.exp(1.j * k * R) / R * (1 / R - 1.j * k)


def kernelRSinverse(X: NDArrayFloat, Y: NDArrayFloat, wavelength: float, z: float,
                    n: float = 1., kind: str = 'z'):
    """Kernel for inverse RS propagation

    Args:
        X(numpy.np.array): positions x
        Y(numpy.np.array): positions y
        wavelength(float): wavelength of incident fields
        z(float): distance for propagation
        n(float): refractive index of background
        kind(str): 'z', 'x', '0': for simplifying vector propagation

    Returns:
        complex np.array: kernel
    """

    k = 2 * np.pi * n / wavelength
    R = np.sqrt(X**2 + Y**2 + z**2)
    if kind == 'z':
        return 1 / (2 * np.pi) * np.exp(-1.j * k * R) * z / R**2 * (1 / R + 1.j * k)
    elif kind == 'x':
        return 1 / (2 * np.pi) * np.exp(-1.j * k * R) * X / R**2 * (1 / R + 1.j * k)
    elif kind == 'y':
        return 1 / (2 * np.pi) * np.exp(-1.j * k * R) * Y / R**2 * (1 / R + 1.j * k)
    elif kind == '0':
        return 1 / (2 * np.pi) * np.exp(-1.j * k * R) / R * (1 / R + 1.j * k)


def kernelFresnel(X: NDArrayFloat, Y: NDArrayFloat, wavelength: float, z: float, n: float = 1.):
    """
    Kernel for Fresnel propagation.

    Args:
        X(numpy.np.array): positions x
        Y(numpy.np.array): positions y
        wavelength(float): wavelength of incident fields
        z(float): distance for propagation
        n(float): refractive index of background

    Returns:
        complex np.array: kernel
    """

    k = 2 * np.pi * n / wavelength
    return np.exp(1.j * k * (z + (X**2 + Y**2) /
                             (2 * z))) / (1.j * wavelength * z)


def PWD_kernel(u: NDArrayComplex, n: float, k0: float, k_perp2: NDArrayComplex, dz: float):
    """
    Step for scalar(TE) Plane wave decomposition(PWD) algorithm.

    Args:
        u(np.array): field
        n(np.array): refractive index
        k0(float): wavenumber
        k_perp(np.array): transversal k
        dz(float): increment in distances

    Returns:
        (numpy.np.array): Field at at distance dz from the incident field

    References:
        1. Schmidt, S. et al. Wave - optical modeling beyond the thin - element - approximation. Opt. Express 24, 30188 (2016).

    """

    absorption = 0.00

    Ek = fftshift(fft2(u))
    H = np.exp(1j * dz * csqrt(n**2 * k0**2 - k_perp2) - absorption)

    result = (ifft2(fftshift(H * Ek)))
    return result


def WPM_schmidt_kernel(u: NDArrayComplex, n: float, k0: float, k_perp2: NDArrayComplex, dz: float):
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
        Imz = (n == n_m)
        u_final = u_final + Imz * u_temp

    return u_final


def get_RS_minimum_z(range_x: float, range_y: float, num_x: int, num_y: int,
                     wavelength: float, n: float = 1., quality: float = 1, verbose: bool = True):
    """_summary_

    Args:
        range_x (float): range_x
        range_y (float): range_y
        num_x (int): num_x
        num_y (int): num_y
        z (float): z
        wavelength (float): wavelength
        n (float): refractive index
        quality (int, optional): quality. Defaults to 1.
        verbose (bool, optional): prints info. Defaults to True.

    Returns:
        z_min (float): z_min for quality_factor>quality
    """

    dx = range_x / num_x
    dy = range_y / num_y
    dr_real = np.sqrt(dx**2 + dy**2)
    rmax = np.sqrt(range_x**2 + range_y**2)

    factor = (
        ((quality * dr_real + rmax)**2 -
         (wavelength / n)**2 - rmax**2)/2 * n / wavelength)**2 - rmax**2

    if factor > 0:
        zmin = np.sqrt(factor)
        zmin = zmin/2
    else:
        zmin = 0

    if verbose:
        print("z min = {:2.2f}".format(zmin))

    return zmin


def quality_factor(range_x: float, range_y: float, num_x: int, num_y: int,
                   z: float, wavelength: float, n: float = 1., verbose: bool = False):
    """Determine the quality factor for RS algorithm

    Args:
        range_x (float): range_x
        range_y (float): range_y
        num_x (int): num_x
        num_y (int): num_y
        z (float): z
        wavelength (float): wavelength
        n (float): refractive index
        verbose (bool, optional): prints info. Defaults to True.


    Returns:
        _type_: _description_
    """

    dx = range_x / num_x
    dy = range_y / num_y
    dr_real = np.sqrt(dx**2 + dy**2)
    rmax = np.sqrt(range_x**2 + range_y**2)

    dr_ideal = np.sqrt((wavelength / n)**2 + rmax**2 + 2 *
                       (wavelength / n) * np.sqrt(rmax**2 + z**2)) - rmax
    quality = dr_ideal / dr_real
    quality = quality * 2

    if verbose:
        print("Quality factor = {:2.2f}".format(quality))

    return quality