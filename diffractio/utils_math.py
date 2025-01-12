#!/usr/bin/env python3
# ----------------------------------------------------------------------
# Name:        utils_math.py
# Purpose:     Utility functions for mathematical operations
#
# Author:      Luis Miguel Sanchez Brea
#
# Created:     2024
# Licence:     GPLv3
# ----------------------------------------------------------------------


""" Common functions to classes """

# flake8: noqa


from copy import deepcopy
from math import factorial
import numpy as np

import scipy.ndimage as ndimage
from scipy.signal import fftconvolve
from numpy.fft import fft, ifft
from scipy.ndimage import rank_filter

from .utils_typing import npt, Any, NDArray,  NDArrayFloat, NDArrayComplex
from .__init__ import mm


def nextpow2(x: float):
    """Exponent of next higher power of 2. It returns the exponents for the smallest powers of two that satisfy $2^p≥A$ for each element in A. 
    By convention, nextpow2(0) returns zero.

    Args:
        x (float): value

    Returns:
        integer: Exponent of next higher power of 2
    """
    y = np.ceil(np.log2(x))
    if type(x) is np.ndarray:
        y[y == -np.inf] = 0
        return y
    else:
        if y == -np.inf:
            y = 0
        return int(y)


def Bluestein_dft_x(x, f1, f2, fs, mout):
    """Bluestein dft

    Args:
        x (_type_): _description_
        f1 (_type_): _description_
        f2 (_type_): _description_
        fs (_type_): _description_
        mout (_type_): _description_

    Reference:
        - Hu, Y., Wang, Z., Wang, X., Ji, S., Zhang, C., Li, J., Zhu, W., Wu, D.,  Chu, J. (2020). "Efficient full-path optical calculation of scalar and vector diffraction using the Bluestein method". Light: Science and Applications, 9(1). https://doi.org/10.1038/s41377-020-00362-z
    """
    # print("mout = {}".format(mout))

    m = len(x)
    # print("m (len(x)) = {}".format(m))

    f11 = f1 + (mout * fs + f2 - f1) / (2 * mout)
    f22 = f2 + (mout * fs + f2 - f1) / (2 * mout)
    a = np.exp(1j * 2 * np.pi * f11 / fs)
    w = np.exp(-1j * 2 * np.pi * (f22 - f11) / (mout * fs))
    h = np.arange(-m + 1, max(mout, m))
    mp = m + mout - 1
    h = w**((h**2)/2)
    ft = fft(1 / h[0:mp + 1], 2**nextpow2(mp))
    b = a**(-(np.arange(0, m))) * h[np.arange(m - 1, 2 * m - 1)]
    tmp = b.T
    b = fft(x * tmp, 2**nextpow2(mp), axis=0)

    b = ifft(b * ft.T, axis=0)
    # b = b[m:mp + 1].T * h[m - 1:mp]
    # Nuevo:
    # print("b = {}".format(b))
    if mout > 1:
        b = b[m:mp + 1].T * h[m - 1:mp]
    else:
        b = b[0] * h[0]
    l = np.linspace(0, mout - 1, mout)
    l = l / mout * (f22 - f11) + f11
    # print("b = {}".format(b))
    # print("l = {}".format(l))

    Mshift = -m/2
    Mshift = np.exp(-1j * 2 * np.pi * l * (Mshift + 1/2) / fs)
    # print("Mshift = {}".format(Mshift))

    b = b * Mshift

    return b


def Bluestein_dft_xy(x, f1, f2, fs, mout):
    """Bluestein dft

    Args:
        x (_type_): _description_
        f1 (_type_): _description_
        f2 (_type_): _description_
        fs (_type_): _description_
        mout (_type_): _description_
    """
    verbose = False

    m, n = x.shape
    f11 = f1 + (mout * fs + f2 - f1) / (2 * mout)
    f22 = f2 + (mout * fs + f2 - f1) / (2 * mout)
    a = np.exp(1j * 2 * np.pi * f11 / fs)
    w = np.exp(-1j * 2 * np.pi * (f22 - f11) / (mout * fs))
    h = np.arange(-m + 1, max(mout, m))
    mp = m + mout - 1
    h = w**((h**2)/2)
    ft = fft(1 / h[0:mp + 1], 2**nextpow2(mp))
    b = a**(-(np.arange(0, m))) * h[np.arange(m - 1, 2 * m - 1)]
    tmp = np.tile(b, (n, 1)).T
    b = fft(x * tmp, 2**nextpow2(mp), axis=0)
    b = ifft(b * np.tile(ft, (n, 1)).T, axis=0)

    if verbose:
        print("b = {}".format(b))

    if mout > 1:
        b = b[m:mp + 1, 0:n].T * np.tile(h[m - 1:mp], (n, 1))
    else:
        b = b[0] * h[0]

    l = np.linspace(0, mout - 1, mout)
    l = l / mout * (f22 - f11) + f11

    if verbose:
        print("b = {}".format(b))
        print("l = {}".format(l))

    Mshift = -m/2
    Mshift = np.tile(np.exp(-1j * 2 * np.pi * l * (Mshift + 1/2) / fs), (n, 1))
    b = b * Mshift

    return b


def find_local_extrema(kind: str,
                       y: NDArrayFloat,
                       x: NDArrayFloat,
                       pixels_interpolation: float = 0.):
    """Determine local minima in a numpy np.array.

    Args:
        kind (str): 'maxima', 'minima'
        y (numpy.ndarray): variable with local minima.
        x (numpy.ndarray): x position
        pixels_interpolation (float):
    Returns:
        (numpy.ndarray): i positions of local minima.

    Todo:
        Add a filter to remove noise.
    """

    if kind == 'minima':
        y_erode = rank_filter(y, -0, size=3)
        Trues = y_erode == y
    elif kind == 'maxima':
        y_dilate = rank_filter(y, -1, size=3)
        Trues = y_dilate == y
    else:
        print("bad parameter in find_local_extrema: only 'maxima or 'minima'")

    i_pos_integer = np.where(Trues == True)[0]
    i_pos_integer = i_pos_integer[0:-1]
    x_minima = x[i_pos_integer]
    y_minima = y[i_pos_integer]

    if pixels_interpolation == 0:
        x_minima_frac = x_minima
        y_minima_frac = y_minima
    else:
        x_minima_frac = np.zeros_like(x_minima, dtype=float)
        y_minima_frac = np.zeros_like(x_minima, dtype=float)
        for i_j, j in enumerate(i_pos_integer):
            js = np.array(
                np.arange(j - pixels_interpolation,
                          j + pixels_interpolation + 1))
            p_j = np.polyfit(x[js], y[js], 2)
            y_minima_interp = np.poly1d(p_j)
            x_minima_frac[i_j] = -p_j[1] / (2 * p_j[0])
            y_minima_frac[i_j] = y_minima_interp(x_minima_frac[i_j])

    return x_minima_frac, y_minima_frac, i_pos_integer


def reduce_to_1(class_diffractio):
    """All the values greater than 1 pass to 1. This is used for Scalar_masks when we add two masks.
    Args:
        class (class): Scalar_field_X, XY ,....

    """
    class_diffractio.u[np.abs(class_diffractio.u > 1)] = 1

    return class_diffractio

def distance(p1: NDArray | float,p2: NDArray | float, verbose: bool = False):
    """Distance between to floats or numpy arrays.

    Args:
        p1 (NDArray | float): Point 1
        p2 (NDArray | float): Point 2
        verbose (bool, optional): prints distance. Defaults to False.

    Returns:
        float: distance between the two points
    """

    if isinstance(p2, (int, float)):
        d = np.abs((p2-p1))
    else:
        d = np.sqrt(np.sum((p2-p1)**2, axis=1))

    if verbose:
        print(f'distance between p1 and p2: {d}')

    return d

def distance_backup(x1: NDArrayFloat, x2: NDArrayFloat):
    """Compute distance between two vectors.

    Args:
        x1 (numpy.np.array): vector 1
        x2 (numpy.np.array): vector 2

    Returns:
        (float): distance between vectors.
    """
    if len(x1) != len(x2):
        raise Exception('distance: arrays with different number of elements')
    else:
        return np.linalg.norm(x2 - x1)

def nearest(vector: NDArray, number: float | int, verbose: bool=False):
    """find the nearest value in a vector to a given number

    Args:
        vector (NDArray): Array with values
        number (float | int): value to compare with vector
        verbose (bool, optional): prints something. Defaults to True.

    Returns:
        i_mins: index of the nearest value in numbers
        values: nearest values in numbers
        distances: distances between vector and nearest values in numbers
    """
    
    ds = distance(vector, number)

    index = ds.argmin()
    value = vector[index]
    dist_min = ds[index]

    if verbose:
        print(f'index: {index}, value: {value}, dist_min: {dist_min}')

    return index, value, dist_min


def nearest_backup(vector: NDArrayFloat, number: float):
    """Computes the nearest element in vector to number.

    Args:
        vector (numpy.np.array): np.array with numbers
        number (float):  number to determine position

    Returns:
        (int): index - index of vector which is closest to number.
        (float): value  - value of vector[index].
        (float): distance - difference between number and chosen element.
    """

    indexes = np.abs(vector - number).argmin()
    values = vector.flat[indexes]
    distances = values - number

    return indexes, values, distances

def nearest2(vector: NDArray, numbers: NDArray, verbose: bool=False):
    """find the nearest value of numbers to a given vector

    Args:
        vector (NDArray): N dimensional vector to compare with numbers
        numbers (NDArray): N dimensional array with values

    Returns:
        i_mins: index of the nearest value in numbers
        values: nearest values in numbers
        distances: distances between vector and nearest values in numbers
    """

    i_mins = np.zeros(len(numbers))
    values = np.zeros(len(numbers))
    distances = np.zeros(len(numbers))
    
    i_mins = []
    values = []
    distances = []

    for i, number in enumerate(numbers):
        # i_mins[i], values[i], distances[i] = nearest(vector, number, verbose=False)
        i_min, value, dist_min = nearest(vector, number, verbose=False)
        i_mins.append(i_min)
        values.append(value)
        distances.append(dist_min)

    i_mins = np.array(i_mins)
    values = np.array(values)
    distances = np.array(distances)
           


    if verbose:
        print(f'i_mins: {i_mins}')    
        print(f'values: \n {values}')    
        print(f'distances: {distances}')    

    return i_mins, values, distances

def nearest2_backup(vector: NDArrayFloat, numbers: NDArrayFloat):
    """Computes the nearest element in vector to numbers.

    Args:
        vector (numpy.np.array): np.array with numbers
        number (numpy.np.array):  numbers to determine position

    Returns:
        (numpy.np.array): index - indexes of vector which is closest to number.
        (numpy.np.array): value  - values of vector[indexes].
        (numpy.np.array): distance - difference between numbers and chosen elements.
    """

    indexes = np.abs(np.subtract.outer(vector, numbers)).argmin(0)
    values = vector[indexes]
    distances = values - numbers

    return indexes, values, distances


def find_extrema(array2D: NDArrayFloat, x: NDArrayFloat, y: NDArrayFloat, kind: float | str = 'max',
                 verbose: bool = False):
    """In a 2D-np.array, formed by vectors x, and y, the maxima or minima are found

    Args:
        array2D (np.array 2D): 2D np.array with variable
        x (np.array 1D): 1D np.array with x axis
        y (np.array 1D): 1D np.array with y axis
        kind (str): 'min' or 'max': detects minima or maxima
        verbose (bool): If True prints data.

    Returns:
        indexes (int,int): indexes of the position
        xy_ext (float, float): position of maximum
        extrema (float): value of maximum
    """

    if kind == 'max':
        result = np.where(array2D == np.a_max(array2D))
    elif kind == 'min':
        result = np.where(array2D == np.min(array2D))

    listOfCordinates = list(zip(result[1], result[0]))

    num_extrema = len(listOfCordinates)

    indexes = np.zeros((num_extrema, 2), dtype=int)
    xy_ext = np.zeros((num_extrema, 2))
    extrema = np.zeros((num_extrema))

    for i, cord in enumerate(listOfCordinates):
        indexes[i, :] = cord[0], cord[1]
        xy_ext[i, 0] = x[cord[0]]
        xy_ext[i, 1] = y[cord[1]]
        extrema[i] = array2D[cord[1], cord[0]]

    if verbose is True:
        for cord in listOfCordinates:
            print(cord, x[cord[0]], y[cord[1]], array2D[cord[1], cord[0]])

    return indexes, xy_ext, extrema


def get_amplitude(u: NDArrayComplex, sign: bool = False):
    """Gets the amplitude of the field.

    Args:
        u (numpy.np.array): Field.
        sign (bool): If True, sign is kept, else, it is removed

    Returns:
        (numpy.np.array): numpy.np.array
    """

    amplitude = np.abs(u)

    if sign is True:
        phase = np.angle(u)
        amplitude = np.sign(phase) * amplitude

    return amplitude


def get_phase(u: NDArrayComplex):
    """Gets the phase of the field.

    Args:
        u (numpy.np.array): Field.

    Returns:
        (numpy.np.array): numpy.np.array
    """

    phase = np.exp(1j * np.angle(u))

    return phase


def amplitude2phase(u: NDArrayComplex):
    r"""Passes the amplitude of a complex field to phase. Previous phase is removed. :math:`u = A e^{i \phi}  -> e^(i abs(A))`

    Args:
        u (numpy.np.array, dtype=complex): complex field

    Returns:
        (numpy.np.array): only-phase complex vector.
    """

    amplitude = np.abs(u)
    u_phase = np.exp(1.j * amplitude)

    return u_phase


def phase2amplitude(u: NDArrayComplex):
    """Passes the phase of a complex field to amplitude.

    Args:
        u (numpy.np.array, dtype=complex): complex field

    Returns:
        (numpy.np.array): amplitude without phase complex vector.
    """

    phase = np.angle(u)
    u_amplitud = phase

    return u_amplitud


def normalize(v: NDArray, order: int = 2):
    """Normalize vectors with different L norm (standard is 2).

    Args:
        v (numpy.np.array): vector to normalize
        order (int): order for norm

    Returns:
        (numpy.np.array): normalized vector.
    """

    norm = np.linalg.norm(v, ord=order)

    if norm == 0:
        raise ValueError('normalize: norm = 0.')

    return v / norm


def binarize(vector: NDArrayFloat, min_value: float = 0., max_value: float = 1.):
    """Binarizes vector between two levels, min and max. The central value is (min_value+max_value)/2

    Args:
        vector: (numpy.np.array) np.array with values to binarize
        min_value (float): minimum value for binarization
        max_value (float): maximum value for binarization

    Returns:
        (numpy.np.array): binarized vector.
    """

    central_value = (min_value + max_value)/2

    vector2 = deepcopy(vector)
    vector2[vector2 <= central_value] = min_value
    vector2[vector2 > central_value] = max_value

    return vector2


def discretize(u: NDArrayComplex,
               kind: str = 'amplitude',
               num_levels: int = 2,
               factor: float = 1.,
               phase0: float = 0.,
               new_field: bool = True,
               matrix: bool = False):
    """Discretize in a number of levels equal to num_levels.

    Args:
        u (np.array, dtype = comples): field
        kind (str): "amplitude" o "phase"
        num_levels (int): number of levels for the discretization
        factor (float): from the level, how area is binarized. if 1 everything is binarized,
        phase0 (float): *
        new_field (bool): if True returns new field
        matrix (bool): if True it returs a matrix

    Returns:
        scalar_fields_XY: if new_field is True returns scalar_fields_XY
    """

    if kind == 'amplitude':
        heights = np.linspace(0, 1, num_levels)
        posX = 256 / num_levels

        amplitude = get_amplitude(u)
        phase = get_phase(u)
        discretized_image = amplitude

        dist = factor * posX

        for i in range(num_levels):
            centro = posX/2 + i * posX
            abajo = amplitude * 256 > centro - dist/2
            arriba = amplitude * 256 <= centro + dist/2
            Trues = abajo * arriba
            discretized_image[Trues] = centro/256

        fieldDiscretizado = discretized_image * phase

    if kind == 'phase':
        ang = np.angle(get_phase(u)) + phase0 + np.pi
        ang = ang % (2 * np.pi)
        amplitude = get_amplitude(u)

        heights = np.linspace(0, 2 * np.pi, num_levels + 1)

        dist = factor * (heights[1] - heights[0])

        discretized_image = np.exp(1j * (ang))

        for i in range(num_levels + 1):
            centro = heights[i]
            abajo = (ang) > (centro - dist/2)
            arriba = (ang) <= (centro + dist/2)
            Trues = abajo * arriba
            discretized_image[Trues] = np.exp(1j * centro)  # - np.pi

        Trues = (ang) > (centro + dist/2)
        discretized_image[Trues] = np.exp(1j * heights[0])  # - np.pi

        # esto no haría falta, pero es para tener tantos levels
        # como decimos, no n+1 (-np.pi,np.pi)
        phase = np.angle(discretized_image) / np.pi
        phase[phase == 1] = -1
        phase = phase - phase.min()  # esto lo he puesto a última hora
        discretized_image = np.exp(1j * np.pi * phase)

        discretized_field = amplitude * discretized_image

        return discretized_field


def delta_kronecker(a: float, b: float):
    """Delta kronecker

    Args:
        a (float): number
        b (float): number

    Returns:
        1 if a==b and 0 if a<>b
    """

    if a == b:
        return 1
    else:
        return 0


def vector_product(A: NDArrayFloat, B: NDArrayFloat):
    """Returns the vector product between two vectors.

    Args:
        A (numpy.np.array): 3x1 vector np.array.
        B (numpy.np.array): 3x1 vector np.array.

    Returns:
        (numpy.np.array): 3x1 vector product np.array
    """

    Cx = A[1] * B[2] - A[2] * B[1]
    Cy = A[2] * B[0] - A[0] * B[2]
    Cz = A[0] * B[1] - A[1] * B[0]

    return np.array((Cx, Cy, Cz))


def dot_product(A: NDArrayFloat, B: NDArrayFloat):
    """Returns the dot product between two vectors.

    Args:
        A (numpy.np.array): 3x1 vector np.array.
        B (numpy.np.array): 3x1 vector np.array.

    Returns:
        (complex): 3x1 dot product
    """

    return A[0] * B[0] + A[1] * B[1] + A[2] * B[2]


def divergence(E: NDArrayFloat, r: NDArrayFloat):
    """Returns the divergence of a field a given point (x0,y0,z0).

    Args:
        E (numpy.np.array): complex field
        r (numpy.np.array): 3x1 np.array with position r=(x,y,z).

    Returns:
        (float): Divergence of the field at (x0,y0,z0)
    """

    x0, y0, z0 = r

    dEx, dEy, dEz = np.gradient(E, x0[1] - x0[0], y0[1] - y0[0], z0[1] - z0[0])

    return dEx + dEy + dEz


def curl(E: NDArrayFloat, r: NDArrayFloat):
    """Returns the Curl of a field a given point (x0,y0,z0)

    Args:
        E (numpy.np.array): complex field
        r (numpy.np.array): 3x1 np.array with position r=(x,y,z).

    Returns:
        (numpy.np.array): Curl of the field at (x0,y0,z0)
    """

    x0, y0, z0 = r

    dEx, dEy, dEz = np.gradient(E, x0[1] - x0[0], y0[1] - y0[0], z0[1] - z0[0])
    curl_X = E[2] * dEy - E[1] * dEz
    curl_Y = E[0] * dEz - E[2] * dEx
    curl_Z = E[1] * dEx - E[0] * dEy

    return curl_X, curl_Y, curl_Z


def get_edges(x: NDArrayFloat, f: NDArrayFloat, kind_transition: str = 'amplitude',
              min_step: float = 0., verbose: bool = False, filename: str = ''):
    """We have a binary mask and we obtain locations of edges. Valid for litography engraving of gratings

    Args:
        x (NDArrayFloat): position x
        f (NDArrayFloat): Field. If real function, use 'amplitude' in kind_transition.
        kind_transition (str):'amplitude' 'phase' of the field where to get the transitions.
        min_step (float): minimum step for consider a transition
        verbose (bool): If True prints information about the process.
        filename (str): If not '', saves the data on files. filename is the file name.

    Returns:
        type_transition (numpy.np.array): np.array with +1, -1 with rasing or falling edges
        pos_transition (numpy.np.array): positions x of transitions
        raising (numpy.np.array): positions of raising
        falling (numpy.np.array): positions of falling
    """

    incr_x = x[1] - x[0]
    if kind_transition == 'amplitude':
        t = np.abs(f)
    elif kind_transition == 'phase':
        t = np.angle(f)
    diferencias = np.diff(t)
    t = np.concatenate((diferencias, np.array([0.])))

    raising = x[t > min_step] + .5 * incr_x
    falling = x[t < -min_step] + .5 * incr_x

    ones_raising = np.ones_like(raising)
    ones_falling = -np.ones_like(raising)

    pos_transitions = np.concatenate((raising, falling))
    type_transitions = np.concatenate((ones_raising, ones_falling))

    i_pos = np.argsort(pos_transitions)
    pos_transitions = pos_transitions[i_pos]
    type_transitions = type_transitions[i_pos]

    if verbose is True:
        print("position of transitions:")
        print("_______________________")
        print(np.array([pos_transitions, type_transitions]).T)
        print("\n\n")
        print("raising         falling:")
        print("_______________________")
        print(np.array([raising, falling]).T)

    if filename != '':
        np.savetxt("{}_pos_transitions.txt".format(filename),
                   pos_transitions,
                   fmt='%10.6f')
        np.savetxt("{}_type_transitions.txt".format(filename),
                   type_transitions,
                   fmt='%10.6f')
        np.savetxt("{}_raising.txt".format(filename), raising, fmt='%10.6f')
        np.savetxt("{}_falling.txt".format(filename), falling, fmt='%10.6f')

    return pos_transitions, type_transitions, raising, falling


def cut_function(x: NDArrayFloat, y: NDArrayFloat, length: float, x_center: float | None = None):
    """Takes values of function inside (x_center+length/2: x_center+length/2)


    Args:
        x (np.array): x of function
        y (np.array): y of function
        length (float): range of data to keep
        x_center (float or None, optional): position of center of Range. If None, the center is x_center.

    Returns:
        y cutted (np.array): values in range.
    """

    if x_center in ('', None, []):
        x_center = (x[0] + x[-1])/2

    incr = length/2
    left = x_center - incr
    right = x_center + incr

    i_min, _, _ = nearest(x, left)
    i_max, _, _ = nearest(x, right)

    y[0:i_min] = 0
    y[i_max::] = 0
    y[-1] = y[-2]

    return y


def fft_convolution2d(x: NDArrayFloat, y: NDArrayFloat):
    """ 2D convolution, using FFT

    Args:
        x (numpy.np.array): np.array 1 to convolve
        y (numpy.np.array): np.array 2 to convolve

    Returns:
        convolved function
    """
    return fftconvolve(x, y, mode='same')


def fft_convolution1d(x: NDArrayFloat, y: NDArrayFloat):
    """ 1D convolution, using FFT

    Args:
        x (numpy.np.array): np.array 1 to convolve
        y (numpy.np.array): np.array 2 to convolve

    Returns:
        convolved function
    """

    return fftconvolve(x, y, mode='same')


def fft_filter(x: NDArrayFloat, y: NDArrayFloat, normalize: bool = False):
    """ 1D convolution, using FFT

    Args:
        x (numpy.np.array): np.array 1 to convolve
        y (numpy.np.array): np.array 2 to convolve

    Returns:
        convolved function
    """

    y = y / y.sum()

    return fftconvolve(x, y, mode='same') / fftconvolve(
        x, np.ones_like(y) / sum(y), mode='same')


def fft_correlation1d(x: NDArrayFloat, y: NDArrayFloat):
    """ 1D correlation, using FFT (fftconvolve)

    Args:
        x (numpy.np.array): np.array 1 to convolve
        y (numpy.np.array): np.array 2 to convolve

    Returns:
        numpy.np.array: correlation function
    """
    return fftconvolve(x, y[::-1], mode='same')


def fft_correlation2d(x: NDArrayFloat, y: NDArrayFloat):
    """Args:
        x (numpy.np.array): np.array 1 to convolve
        y (numpy.np.array): np.array 2 to convolve

    Returns:
        numpy.np.array: 2d correlation function
    """

    return fftconvolve(x, y[::-1, ::-1], mode='same')


def rotate_image(x: NDArrayFloat, z: NDArrayFloat, img: NDArrayFloat, angle: float,
                 pivot_point: tuple[float, float]):
    """similar to rotate image, but not from the center but from the given

    Args:
        x (np.array): x of image
        z (np.array): z of image
        img (np.array): image to rotate
        angle (float): np.angle to rotate
        pivot_point (float, float): (z,x) position for rotation

    Returns:
        rotated image

    Reference:
        https://stackoverflow.com/questions/25458442/rotate-a-2d-image-around-specified-origin-in-python point

    """

    # first get (i,j) pixel of rotation
    ipivotz, _, _ = nearest(z, pivot_point[0])
    ipivotx, _, _ = nearest(x, pivot_point[1])

    ipivot = [ipivotx, ipivotz]

    # rotates
    padX = [img.shape[1] - ipivot[0], ipivot[0]]
    padZ = [img.shape[0] - ipivot[1], ipivot[1]]
    imgP = np.pad(img, [padZ, padX], 'constant')
    imgR = ndimage.rotate(imgP, angle, reshape=False)

    return imgR[padZ[0]:-padZ[1], padX[0]:-padX[1]]


def cart2pol(x: NDArrayFloat, y: NDArrayFloat):
    """ cartesian to polar coordinate transformation.

    Args:
        x (np.array): x coordinate
        y (np.aray): y coordinate

    Returns:
        numpy.np.array: rho
        numpy.np.array: phi
    """
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)

    return rho, phi


def pol2cart(rho: NDArrayFloat, phi: NDArrayFloat):
    """Polar to cartesian coordinate transformation

    Args:
        rho (np.array): rho coordinate
        rho (np.aray): rho coordinate

    Returns:
        numpy.np.array: x
        numpy.np.array: y
    """

    x = rho * np.cos(phi)
    y = rho * np.sin(phi)

    return (x, y)


def fZernike(X: NDArrayFloat, Y: NDArrayFloat, n: int, m: int, radius: float):
    """Zernike function for aberration computation.

    Args:
        X (np.array): X
        Y (np.array): Y
        n (int): _description_
        m (int): _description_
        radius (_type_, optional): _description_. Defaults to 5*mm.

    Returns:
        zernike polinomial: _description_

    Note:
        k>=l

        if k is even then l is even.
        if k is  odd then l is  odd.

        The first polinomial is the real part and the second de imaginary part.

        * n m aberration
        * 0 0 piston
        * 1 -1 vertical tilt
        * 1 1 1 horizontal tilt
        * 2 -2 oblique astigmatism
        * 2 0 defocus myopia if c>0 or defocus hypermetropia if c<0
        * 2 2 2 abnormal astigmatism if c>0 or normal astigmatism if c<0
        * 3 -3 oblique trefoil
        * 3 -1 as vertical coma, c>0 upper steepening, c<0 lower steepening
        * 3 1 as horizontal
        * 3 3 3 horizontal trefoil
        * 4 -4 oblique four-leaf clover
        * 4 -2 oblique secondary astigmatism
        * 4 0 spherical c>0 periphery more myopic than centre, c<0 periphery more hyperopic than centre
        * 4 2 secondary astigmatism for or against the ruler
        * 4 4 4 horizontal four-leaf clover


    Reference:

        R. Navarro, J. Arines, R. Rivera "Direct and inverse discrete Zernike transform" Opt. Express 17(26) 24269

    """

    R = np.sqrt(X**2 + Y**2) / (radius)
    THETA = np.arctan2(Y,X)

    N = np.sqrt((n + 1) * (2 - delta_kronecker(m, 0)))

    Z = np.zeros(R.shape, dtype=float)
    s_max = int(((n - np.abs(m))/2 + 1))
    for s in np.arange(0, s_max):
        Z = Z + (-1)**s * R**(n - 2 * s) * factorial(np.abs(n - s)) / (
            factorial(np.abs(s)) * factorial(np.abs(round(0.5 * (n + np.abs(m)) - s))) *
            factorial(np.abs(round(0.5 * (n - np.abs(m)) - s))))

    if m >= 0:
        fz1 = N * Z * np.cos(m * THETA)
    else:
        fz1 = N * Z * np.sin(np.abs(m) * THETA)

    fz1[R >= 1] = 0

    return fz1


def laguerre_polynomial_nk(x: NDArrayFloat, n: int, k: int):
    """Auxiliar laguerre polinomial of orders n and k. 
        Calculates the utilsized Laguerre polynomial L{n, alpha}
        This function computes the utilsized Laguerre polynomial L{n,alpha}.
        If no alpha is supplied, alpha is set to zero and this function
        calculates the "normal" Laguerre polynomial.

        Calculation is done recursively using matrix operations for very fast execution time.

        Args:
            - x (nd.array): position
            - n (int): nonnegative integer as degree level
            - alpha (float): >= -1 real number (input is optional)

        The output is formated as a polynomial vector of degree (n+1)
        corresponding to MatLab norms (that is the highest coefficient
        is the first element).

        Example:
            - polyval(LaguerreGen(n, alpha), x) evaluates L{n, alpha}(x)
            - roots(LaguerreGen(n, alpha)) calculates roots of L{n, alpha}

        Author: Matthias.Trampisch@rub.de
        Date: 16.08.2007
        Version 1.2

        References:
            Szeg: "Orthogonal Polynomials" 1958, formula (5.1.10)

        """

    f = factorial
    summation = np.zeros_like(x, dtype=float)
    for m in range(n + 1):
        summation = summation + (-1)**m * f(n + k) / (f(n - m) * f(k + m) *
                                                      f(m)) * x**m
    return summation


def get_k(x: NDArrayComplex, flavour: str = '-'):
    """Provides k vector from x vector. With flavour set to "-", the axis will be inverse-fftshifted,
        thus its DC part being the first index. 

    Args:
        x (np.array): x np.array
        flavour (str): '-' (for ifftshifted axis)

    Returns:
        kx (np.array): k vector

    Fixed by Bob-Swinkels
    """

    num_x = x.size
    integerFrom = int(np.floor((1-num_x)/2))
    integerTo = int(np.floor((num_x-1)/2))
    intRange = np.linspace(integerFrom, integerTo, num_x)  # ordered k axis, DC is at int(np.floor(num_x/2))
    if flavour == '-':
        intRange = np.fft.ifftshift(intRange)  # leading zero (DC) frequency
    dx = x[1] - x[0]
    dk = 2 * np.pi / (num_x * dx)
    return dk * intRange


def filter_edge_1D(x: NDArrayFloat, size: float = 1.1, exponent: float = 32):
    """function 1 at center and reduced at borders. For propagation algorithms

    Args:
        x (np.array): position
        size (float): related to relative position of x
        exponent (integer): related to shape of edges
    Returns:
        np.array: function for filtering
    """

    # num_x = len(x)
    x_center = (x[-1] + x[0])/2
    Dx = size * (x[-1] - x[0])
    return np.exp(-(2 * (x - x_center) / (Dx))**np.abs(exponent))


def filter_edge_2D(x: NDArrayFloat, y: NDArrayFloat, size: float = 1.1, exponent: float = 32):
    """function 1 at center and reduced at borders. For propagation algorithms

    Args:
        x (np.array): x position
        y (np.array): y position
        size (float): related to relative position of x and y
        exponent (integer): related to shape of edges
    Returns:
        np.array: function for filtering
    """

    x_center = (x[-1] + x[0])/2
    y_center = (y[-1] + y[0])/2
    Dx = size * (x[-1] - x[0])
    Dy = size * (y[-1] - y[0])

    X, Y = np.meshgrid(x, y)

    exp1 = np.exp(-(2 * (X - x_center) / (Dx))**np.abs(exponent))
    exp2 = np.exp(-(2 * (Y - y_center) / (Dy))**np.abs(exponent))

    return exp1 * exp2
