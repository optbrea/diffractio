#!/usr/bin/env python3

# ----------------------------------------------------------------------
# Name:        scalar_masks_XZ.py
# Purpose:     Defines the Scalar_mask_XZ class for working with XZ scalar masks
#
# Author:      Luis Miguel Sanchez Brea
#
# Created:     2024
# Licence:     GPLv3
# ----------------------------------------------------------------------


"""Generates Scalar_mask_XZ class for definingn masks. Its parent is Scalar_field_XZ.

The main atributes are:
    * self.x - x positions of the field
    * self.z - z positions of the field
    * self.u - field XZ
    * self.n - refractive index XZ
    * self.wavelength - wavelength of the incident field. The field is monochromatic


The magnitude is related to microns: `micron = 1.`


*Class for unidimensional scalar masks*

*Functions*
    * extrude_mask, mask_from_function, mask_from_array, object_by_surfaces
    * image
    * semi_plane, layer, square, slit, cylinder, semi_cylinder
    * wedge, prism, biprism
    * ronchi_grating, sine_grating
    * probe
    * aspheric_lens, lens
    * roughness
"""

# flake8: noqa


from copy import deepcopy

import matplotlib.image as mpimg
import numexpr as ne
import scipy.ndimage as ndimage
from scipy.interpolate import interp1d
from scipy.signal import fftconvolve


from .__init__ import degrees, np, plt, sp, um
from .config import bool_raise_exception
from .utils_typing import npt, Any, NDArray,  NDArrayFloat, NDArrayComplex
from .utils_math import nearest, nearest2
from .utils_optics import roughness_1D
from .utils_dxf import load_dxf
from .utils_common import check_none
from .scalar_fields_XZ import Scalar_field_XZ
from .scalar_masks_X import Scalar_mask_X



class Scalar_mask_XZ(Scalar_field_XZ):
    """Class for working with XZ scalar masks.

    Args:
        x (numpy.array): linear array with equidistant positions.
            The number of data is preferibly :math:`2^n` .
        z (numpy.array): linear array wit equidistant positions for z values
        wavelength (float): wavelength of the incident field
        n_background (float): refractive index of background
        info (str): String with info about the simulation

    Attributes:
        self.x (numpy.array): linear array with equidistant positions. The number of data is preferibly :math:`2^n`.
        self.z (numpy.array): linear array wit equidistant positions for z values
        self.wavelength (float): wavelength of the incident field.
        self.u0 (numpy.array): (x) size x - field at the last z position
        self.u (numpy.array): (x,z) complex field
        self.n_background (numpy.array): (x,z) refractive index
        self.info (str): String with info about the simulation
    """

    def __init__(self, x: NDArrayFloat | None = None,
                 z: NDArrayFloat | None = None, wavelength: float | None = None,
                 n_background: float = 1., info: str = ""):
        """inits a new experiment:
        x: numpy array with x locations
        z: numpy array with z locations
        wavelength: wavelength of light
        n_backgraound: refractive index of background
        info: text to describe the instance of the class"""
        super().__init__(x, z, wavelength, n_background, info)
        self.type = "Scalar_mask_XZ"


    @check_none('x','z',raise_exception=bool_raise_exception)
    def object_by_surfaces(
        self, rotation_point: tuple[float, float], refractive_index: complex | float | str,
        Fs: list, angle: float, v_globals: dict = {}, verbose: bool = False
    ):
        """Mask defined by n surfaces given in array Fs={f1, f2, ....}.
        h(x,z)=f1(x,z)*f2(x,z)*....*fn(x,z)


        Args:
            rotation_point (float, float): location of the mask
            refractive_index (float, str): can be a number or a function n(x,z)
            Fs (tuple): condtions as str that will be computed using eval
            angle (float): angle of rotation (radians)
            v_globals (dict): dict with global variables -> TODO perphaps it is not necessary
            verbose (bool): shows data if true
        """

        # Rotacion del square/square
        Xrot, Zrot = self.__rotate__(angle, rotation_point)

        v_locals = {"self": self, "degrees": degrees, "um": um, "np": np}

        v_locals["Xrot"] = Xrot
        v_locals["Zrot"] = Zrot

        conditions = []
        for fi in Fs:
            try:
                result_condition = ne.evaluate(fi, v_globals, v_locals)
            except:
                result_condition = eval(fi, v_globals, v_locals)

            conditions.append(result_condition)

        # Transmitancia de los puntos interiores
        ipasa = conditions[0]
        for cond in conditions:
            ipasa = ipasa & cond

        if verbose is True:
            print(("n = {}".format(refractive_index)))

        if isinstance(refractive_index, (int, float, complex)):
            self.n[ipasa] = refractive_index
            return ipasa
        else:
            v_locals = {"self": self, "np": np, "degrees": degrees, "um": um}
            tmp_refractive_index = refractive_index

            v_locals["X"] = Xrot
            v_locals["Z"] = Zrot

            refractive_index = eval(tmp_refractive_index, v_globals, v_locals)
            self.n[ipasa] = refractive_index[ipasa]
            return ipasa


    @check_none('x','z',raise_exception=bool_raise_exception)
    def extrude_mask(self, t, z0: float, z1: float, refractive_index: complex | float | str,
                     angle: float = 0*degrees, v_globals: dict = {}):
        """
        Converts a Scalar_mask_X in volumetric between z0 and z1 by growing between these two planes
        Args:
            t (Scalar_mask_X): an amplitude mask of type Scalar_mask_X.
            z0 (float): initial  position of mask
            z1 (float): final position of mask
            refractive_index (float, str): can be a number or a function n(x,z)
        """

        iz0, value, distance = nearest(vector=self.z, number=z0)
        iz1, value, distance = nearest(vector=self.z, number=z1)

        if isinstance(refractive_index, (int, float, complex)):
            n_is_number = True
            # refractive_index = refractive_index * np.ones((iz1 - iz0))
        else:
            n_is_number = False
            v_locals = {"self": self, "np": np, "degrees": degrees, "um": um}
            tmp_refractive_index = refractive_index

        for i, index in enumerate(range(iz0, iz1)):
            if n_is_number is False:
                v_locals["z"] = self.z[index]
                v_locals["x"] = self.x

                refractive_index = eval(
                    tmp_refractive_index, v_globals, v_locals)

            self.n = self.n.astype(complex)
            self.n[index, :] = refractive_index * (1 - t.u)
            self.n[index, :] = self.n[index, :] + self.n_background * t.u


    def mask_from_function(
            self, r0: tuple[float, float], refractive_index: complex | float | str, f1, f2, z_sides: tuple[float],
            angle: float, r_rot: tuple[float, float] | None = None, v_globals: dict = {}):
        """
        Phase mask defined between two surfaces f1 and f1: h(x,z)=f2(x,z)-f1(x,z)

        Args:
            r0 (float, float): location of the mask
            refractive_index (float, str): can be a number or a function n(x,z)
            f1 (str): function that delimits the first surface
            f2 (str): function that delimits the second surface
            z_sides (float, float): limiting upper and lower values in z,
            angle (float): angle of rotation (radians)
            r_rot (float, float): rotation point. If None then r_rot = r0.
            v_globals (dict): dict with global variables
        """

        v_locals = {"self": self, "np": np, "degrees": degrees, "um": um}
        
        if r_rot == None:
            r_rot = r0

        F2 = eval(f2, v_globals, v_locals)
        F1 = eval(f1, v_globals, v_locals)

        Xrot, Zrot = self.__rotate__(angle, r_rot)

        ipasa = (Xrot > z_sides[0]) & (
            Xrot < z_sides[1]) & (Zrot < F2) & (Zrot > F1)
        self.n[ipasa] = refractive_index
        return ipasa

    @check_none('x','z',raise_exception=bool_raise_exception)
    def mask_from_array(
        self,
        r0=(0*um, 0*um),
        refractive_index=1.5,
        array1: NDArrayFloat | None = None,
        array2: NDArrayFloat | None = None,
        x_sides: tuple[float, float] | None = None,
        angle: float = 0*degrees,
        v_globals: dict = {},
        interp_kind: str = "quadratic",
        has_draw: bool = False,
    ):
        """Mask defined between two surfaces given by arrays (x,z): h(x,z)=f2(x,z)-f1(x,z).
        For the definion of f1 and f2 from arrays is performed an interpolation

        Args:
            r0 (float, float): location of the mask
            refractive_index (float, str): can be a number or a function n(x,z)
            array1 (numpy.array): array (x,z) that delimits the first surface
            array2 (numpy.array): array (x,z) that delimits the second surface
            x_sides (float, float): limiting upper and lower values in x,
            angle (float): angle of rotation (radians): TODO -> not working
            v_globals (dict): dict with global variables -> TODO perphaps it is not necessary
            interp_kind: 'linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic'
        """

        x_c, z_c = r0

        f1_interp = interp1d(
            array1[0,:] + x_c, array1[1,:] + z_c, kind=interp_kind, bounds_error=False, fill_value=1, # array1[1,0] + z_c, assume_sorted=False,
        )

        f2_interp = interp1d(
            array2[0,:] + x_c, array2[1,:] + z_c, kind=interp_kind, bounds_error=False, fill_value=1, # array2[1,0] + z_c, assume_sorted=False,
        )

        F1 = f1_interp(self.x)
        F2 = f2_interp(self.x)

        if has_draw is True:
            plt.figure()
            plt.plot(self.x, F1, "b")
            plt.plot(self.x, F2, "r")
            

        Xrot, Zrot = self.__rotate__(angle, r0)

        i_z1, _, _ = nearest2(self.z, F1)
        i_z2, _, _ = nearest2(self.z, F2)
        ipasa1 = np.zeros_like(self.n, dtype=bool)
        for i, xi in enumerate(self.x):
            ipasa1[i_z1[i]: i_z2[i], i] = True
            
        if x_sides is None:
            self.n[ipasa1] = refractive_index
            return ipasa1

        else:
            ipasa2 = Xrot < x_sides[1]
            ipasa3 = Xrot > x_sides[0]

            self.n[ipasa1 * ipasa2 * ipasa3] = refractive_index
            return ipasa1 * ipasa2 * ipasa3



    # @check_none('x','z',raise_exception=bool_raise_exception)
    # def mask_from_array_proposal(
    #     self,
    #     r0: tuple[float, float] = (0*um, 0*um),
    #     refractive_index_substrate: float | float = 1.5,
    #     refractive_index_mask: float | float = None,
    #     array1: NDArrayFloat | float = None,
    #     array2: NDArrayFloat | float = None,
    #     x_sides: tuple[float, float] = None,
    #     angle: float = 0*degrees,
    #     v_globals: dict = {},
    #     interp_kind: str = "quadratic",
    #     has_draw: bool = False,
    # ):
    #     """Mask defined between two surfaces given by arrays (x,z): h(x,z)=f2(x,z)-f1(x,z).
    #     For the definion of f1 and f2 from arrays is performed an interpolation

    #     Args:
    #         r0 (float, float): location of the mask
    #         refractive_index_mask (float, str): can be a number or a function n(x,z)
    #         refractive_index_substrate (float, str): can be a number or a function n(x,z)

    #         array1 (numpy.array): array (x,z) that delimits the first surface
    #         array2 (numpy.array): array (x,z) that delimits the second surface
    #         x_sides (float, float): limiting upper and lower values in x,
    #         angle (float): angle of rotation (radians): TODO -> not working
    #         v_globals (dict): dict with global variables -> TODO perphaps it is not necessary
    #         interp_kind: 'linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic'
    #     """

    #     x0, z0 = r0

    #     f1_interp = interp1d(
    #         array1[:, 0] + x0,
    #         array1[:, 1] + z0,
    #         kind=interp_kind,
    #         bounds_error=False,
    #         fill_value=array1[0, 1] + z0,
    #         assume_sorted=True,
    #     )

    #     f2_interp = interp1d(
    #         array2[:, 0] + x0,
    #         array2[:, 1] + z0,
    #         kind=interp_kind,
    #         bounds_error=False,
    #         fill_value=array2[0, 1] + z0,
    #         assume_sorted=True,
    #     )

    #     F1 = f1_interp(self.x)
    #     F2 = f2_interp(self.x)

    #     if has_draw is True:
    #         plt.figure()
    #         plt.plot(self.x, F1)
    #         plt.plot(self.x, F2, "r")

    #     Xrot, Zrot = self.__rotate__(angle, r0)

    #     i_z1, _, _ = nearest2(self.z, F1)
    #     i_z2, _, _ = nearest2(self.z, F2)
    #     ipasa = np.zeros_like(self.n, dtype=bool)

    #     for i, xi in enumerate(self.x):
    #         minor, mayor = min(i_z1[i], i_z2[i]), max(i_z1[i], i_z2[i])
    #         ipasa[i, minor:mayor] = True

    #     if refractive_index_mask not in (None, "", []):
    #         ipasa_substrate = np.zeros_like(self.u)
    #         z_subst_0 = np.max(F1)
    #         z_subst_1 = np.min(F2)

    #         i_z1, _, _ = nearest(self.z, z_subst_0)
    #         i_z2, _, _ = nearest(self.z, z_subst_1)
    #         ipasa_substrate[:, i_z1:i_z2] = refractive_index_substrate

    #     if x_sides is None:
    #         self.n[ipasa] = refractive_index_mask
    #         if refractive_index_mask not in (None, "", []):
    #             ipasa_substrate[:, i_z1:i_z2] = refractive_index_substrate
    #             self.n[ipasa_substrate] = refractive_index_substrate

    #         return ipasa

    #     else:
    #         ipasa2 = Xrot < x_sides[1]
    #         ipasa3 = Xrot > x_sides[0]

    #         self.n[ipasa * ipasa2 * ipasa3] = refractive_index_mask
    #         self.n[ipasa_substrate] = refractive_index_substrate

    #         return ipasa * ipasa2 * ipasa3



    @check_none('x','z',raise_exception=bool_raise_exception)
    def insert_array_masks(self, txz, refractive_index: float, space: tuple[float], margin: tuple[float] | float = 0, angle: float = 0*degrees):
        """Generates a matrix of shapes given in txz.

        Args:
            txz (Scalar_mask_XZ): Mask of the desired figure to be drawn
            space (float, float) or (float): spaces between figures.
            margin (float, float) or (float): extra space outside the mask
            angle (float): Angle to rotate the matrix of circles

        Returns:
            (int): number of points in the mask
        """

        if isinstance(space, (int, float)):
            delta_x, delta_z = (space, space)
        else:
            delta_x, delta_z = space

        if isinstance(margin, (float, int)):
            margin_x, margin_z = (margin, margin)
        else:
            margin_x, margin_z = margin

        assert delta_x > 0 and delta_z > 0

        nj = np.zeros_like(self.X)

        X = margin_x + np.arange(self.x.min(), self.x.max() + delta_x, delta_x)
        Z = margin_z + np.arange(self.z.min(), self.z.max() + delta_z, delta_z)
        
        mask = txz.n - txz.n_background
        
        
        for i, x_i in enumerate(X):
            i_xcercano, _, _ = nearest(self.x, x_i)
            for j, z_j in enumerate(Z):
                j_zcercano, _, _ = nearest(self.z, z_j)
                if x_i < self.x.max() and x_i > self.x.min(
                ) and z_j < self.z.max() and z_j > self.z.min():
                    nj[i_xcercano, j_zcercano] = 1

        n_new = fftconvolve(nj, mask, mode='same')
        # u[u > 1] = refractive_index
        self.n = n_new + self.n_background
        return self

    #@check_none('x','z','u',raise_exception=bool_raise_exception)
    def repeat_structure(self,                 num_repetitions: tuple[int,int],                 position: str = 'center',                 new_field: bool = True):
        """Repeat the structure (n x m) times.

        Args:
            num_repetitions (int, int): Number of repetitions of the mask
            position (string or number,number): 'center', 'previous' or initial position. Initial x
            new_field (bool): If True, a new mask is produced, else, the mask is modified.

        """

        n_new = np.tile(self.n, (num_repetitions[1], num_repetitions[0]))

        x_min = self.x[0]
        x_max = self.x[-1]

        z_min = self.z[0]
        z_max = self.z[-1]

        x_new = np.linspace(num_repetitions[0] * x_min, num_repetitions[0] * x_max, num_repetitions[0] * len(self.z))
        z_new = np.linspace(num_repetitions[1] * z_min, num_repetitions[1] * z_max, num_repetitions[1] * len(self.z))

        center_x = (x_new[-1] + x_new[0])/2
        center_z = (z_new[-1] + z_new[0])/2

        if position == 'center':
            x_new = x_new - center_x
            z_new = z_new - center_z

        elif position == 'previous':
            x_new = x_new - x_new[0] + x0[0]
            z_new = z_new - z_new[0] + z0[0]

        elif isinstance(position, np.array):
            x_new = x_new - x_new[0] + position[0]
            z_new = z_new - z_new[0] + position[1]

        t_new = Scalar_mask_XZ(x=x_new, z=z_new, wavelength=self.wavelength)
        t_new.n = n_new

        return t_new

    
    @check_none('x','z',raise_exception=bool_raise_exception)
    def add_surfaces(
        self, fx, x_sides: tuple[float, float],
        refractive_index: complex | float | str, min_incr: float = 0.1, angle: float = 0*degrees):
        """A topography fx is added to one of the faces of object u (self.n).

        Args:
            u (Scalar_mask_XZ): topography
            fx (numpy.array, numpy.array):  [x1, fx1], [x2, fx2] array with topography to add
            x_sides (float, float): positions of edges
            refractive_index (float, str): refractive index: number of string
            min_incr (float): minimum variation of refractive index to detect edge.
            angle (float (float, float)): angle and optative rotation angle.
        """

        z0 = self.z
        x0 = self.x
        len_z = len(z0)
        len_x = len(x0)

        # plt.figure()
        # plt.imshow(np.abs(self.n).transpose(), aspect="auto")

        diff1a = np.diff(np.abs(self.n), axis=0)
        diff1a = np.append(diff1a, np.zeros((1,len_x)), axis=0)

        # print(diff1a.shape, len_x)
        # plt.figure()
        # plt.plot(diff1a)

        iz_l,ix_l = (diff1a > min_incr).nonzero()
        iz_r,ix_r = (diff1a < -min_incr).nonzero()
        
        # plt.figure()
        # plt.plot(ix_l, iz_l,'r.')
        # plt.plot(ix_r, iz_r,'b.')

        x_lens_l = x0[ix_l]
        h_lens_l = z0[iz_l]

        x_lens_r = x0[ix_r]
        h_lens_r = z0[iz_r]

        fx1, fx2 = fx

        if fx1 is not None:
            x_1, h_1 = fx1  # first surface
            h_1_new = np.interp(x_lens_l, x_1, h_1)
            h_lens_l = h_lens_l + h_1_new
        if fx2 is not None:
            x_2, h_2 = fx2  # second surface
            h_2_new = np.interp(x_lens_r, x_2, h_2)
            h_lens_r = h_lens_r + h_2_new

        len_z1 = len(x_lens_l)
        fx1_n = np.concatenate((x_lens_l, h_lens_l)).reshape(2, len_z1).T

        len_z2 = len(x_lens_r)
        fx2_n = np.concatenate((x_lens_r, h_lens_r)).reshape(2, len_z2).T
            
        # plt.figure()
        # plt.plot(fx1_n[:,0], fx1_n[:,1],'g.')
        # plt.plot(fx2_n[:,0], fx2_n[:,1],'k.')

        previous_profile = self.borders
        self.clear_refractive_index()
        
        self.mask_from_array(
            r0=(0*um, 0*um),
            refractive_index=refractive_index,
            array1=fx1_n,
            array2=fx2_n,
            x_sides=x_sides,
            angle=0*degrees,
            interp_kind="linear",
            has_draw=False
        )

        self.surface_detection()
        new_profile = self.borders

        return previous_profile, new_profile


    @check_none('n',raise_exception=bool_raise_exception)
    def discretize_refractive_index(
        self, num_layers: tuple[int, int] | None = None, n_layers: NDArrayComplex | complex = None,
        new_field: bool = False
    ):
        """Takes a refractive index an discretize it according refractive indexes.

        Args:
            num_layers ((int,int), optional): number of layers (for real and imaginary parts), without counting background. Defaults to None.
                By default, both parameters are None, but one of then must be filled. If both are present, num_layers is considered
            n_layers (np.array, optional): array with refractive indexes to discretize. Defaults to None.
            new_field (bool): If True, it generates a new field.

        Returns:
            (np.array): refractive indexes selected.
        """

        def _discretize_(refractive_index: NDArrayComplex, n_layers: int):
            """internal function to discretize the refractive index

            Args:
                refractive_index (NDArrayComplex): refractive index
                n_layers (int): number of layers

            Returns:
                NDArrayComplex: discretized refractive index
            """
            n_new = np.zeros_like(refractive_index, dtype=float)

            i_n, _, _ = nearest2(n_layers, refractive_index)
            i_n = i_n.reshape(refractive_index.shape)

            for i, n in enumerate(n_layers):
                n_new[i_n == i] = n_layers[i]

            return n_new

        if num_layers is not None:
            if isinstance(num_layers, int):
                num_layers = (num_layers, 1)

        n_real = np.around(self.n.real, 4)
        kappa = np.around(self.n.imag, 4)

        if num_layers[0] is not None:
            if num_layers[0] > 1:
                repeated_values = np.unique(n_real)
                repeated_values = np.delete(
                    repeated_values, np.where(
                        repeated_values == self.n_background)
                )

                n_min, n_max = repeated_values.min(), repeated_values.max()
                n_layers = np.linspace(n_min, n_max, num_layers[0])
                n_layers = np.append(n_layers, self.n_background)
                n_layers = np.unique(n_layers)
                n_new = _discretize_(n_real, n_layers)
            else:
                n_new = n_real
        else:
            n_new = n_real

        if num_layers[1] is not None:
            if num_layers[1] > 1:
                repeated_values = np.unique(kappa)

                k_min, k_max = repeated_values.min(), repeated_values.max()
                k_layers = np.linspace(k_min, k_max, num_layers[1])
                k_layers = np.unique(k_layers)
                k_new = _discretize_(kappa, k_layers)
            else:
                k_new = np.zeros_like(kappa)
                k_new[kappa > kappa.max()/2] = kappa.max()
        else:
            k_new = kappa

        if new_field is True:
            t_new = self.duplicate()
            t_new.u = np.zeros_like(t_new.u)
            t_new.n = n_new + 1j * k_new
            return t_new
        else:
            self.n = n_new + 1j * k_new



    def image(self, filename: str, n_max: float, n_min: float, angle: float = 0*degrees,
              invert: bool = False):
        """Converts an image file in an xz-refractive index matrix.
        If the image is gray-scale the refractive index is gradual betwee n_min and n_max.
        If the image is color, we get the first Red frame

        Args:
            filename (str): filename of the image
            n_max (float): maximum refractive index
            n_min (float): minimum refractive index
            angle (float): angle to rotate the image in radians
            invert (bool): if True the image is inverted

        TODO: Now it is only possible that image size is equal to XZ, change using interpolation
            Rotation position
        """

        image3D = mpimg.imread(filename)
        if len(image3D.shape) > 2:
            image = image3D[:, :, 0]
        else:
            image = image3D

        # angle is in degrees
        image = ndimage.rotate(image, angle * 180 / np.pi, reshape=False)
        image = np.array(image)
        image = (image - image.min()) / (image.max() - image.min())

        if invert is False:
            image = image.max() - image
        self.n = n_min + image * (n_max - n_min)


    def dxf(self, filename_dxf: str, n_max: float, n_min: float, num_pixels: tuple[int, int] | None = None,
            extent: tuple[float] | None = None, units: str = 'um', invert: bool = False, verbose: bool = False):
        """Loads a dxf file. Internally it has the extension of the drawing, so it is not required to generate x,y spaces. It is possible with extent, but then the file is scaled. Warning: Dxf files are usually in mm. and diffractio works in um. To generate .u, a temporal .png file is generated. 

        Args:
            filename_dxf (str): DXF filename .dxf
            num_pixels (tuple[int, int] | None, optional): If . Defaults to None.
            extent (_type_, optional): _description_. Defaults to None.
            units (str, optional): _description_. Defaults to 'mm'.
            invert (bool, optional): _description_. Defaults to False.
            filename_png (str, optional): _description_. Defaults to 'new.png'.
            has_draw (bool, optional): _description_. Defaults to True.
            verbose (bool, optional): _description_. Defaults to True.
        """

        if self.x is not None:
            num_pixels = len(self.z), len(self.x)

        image_new, p_min, p_max, msp = load_dxf(filename_dxf, num_pixels, verbose)
        image_new = np.flipud(image_new)
        image_new = np.transpose(image_new)

        if units == 'mm':
            p_min = p_min*1000
            p_max = p_max*1000
        elif units == 'inches':
            p_min = p_min*25400
            p_max = p_max*25400

        if self.x is None:

            if extent is None:
                self.z = np.linspace(p_min[0], p_max[0], num_pixels[0])
                self.x = np.linspace(p_min[1], p_max[1], num_pixels[1])
                self.X, self.Z = np.meshgrid(self.x, self.z)
                self.n = self.n_background*np.ones_like(self.X)
                self.u = np.zeros_like(self.X, dtype=complex)
            else:
                self.z = np.linspace(extent[2], extent[3], num_pixels[0])
                self.x = np.linspace(extent[0], extent[1], num_pixels[1])
                self.X, self.Z = np.meshgrid(self.x, self.z)
                self.n = self.n_background*np.ones_like(self.X)
                self.u = np.zeros_like(self.X, dtype=complex)

        if invert is True:
            image_new = 1-image_new

        self.n = self.n + image_new * (n_max - n_min)
        # TODO: cuidado con n_min y n_background ¿es lo mismo?


    @check_none('x','z','X',raise_exception=bool_raise_exception)
    def dots(self, positions: tuple[float, float], refractive_index: float = 1.):
        """Generates 1 or several point masks at positions r0

        Args:
            positions (float, float) or (np.array, np.array): (x,z) point or points where mask is 1
            refractive_index (float): refractive index
        """
        x0, z0 = positions
        n = np.zeros_like(self.X)

        if type(positions[0]) in (int, float):
            i_x0, _, _ = nearest(self.x, x0)
            i_z0, _, _ = nearest(self.z, z0)
            n[i_x0, i_z0] = refractive_index
        else:
            i_x0s, _, _ = nearest2(self.x, x0)
            i_z0s, _, _ = nearest2(self.z, z0)
            for i_x0, i_z0 in zip(i_x0s, i_z0s):
                n[i_x0, i_z0] = refractive_index

        self.n = n
        return self


    def semi_plane(self, r0: tuple[float, float], refractive_index: complex | float | str,
                   angle: float = 0*degrees, rotation_point: tuple[float, float] | None = None):
        """Inserts a semi-cylinder in background (x>x0). If something else previous, it is removed.

        Args:
            r0=(x0,z0) (float,float): Location of the same plane.
            refractive_index (float, str): refractive index.
            angle (float): angle of rotation of the semi-plane, in radians
            rotation_point (float, float). Rotation point
        """

        x0, z0 = r0
        if rotation_point is None:
            rotation_point = r0  # DUDA

        cond1 = "Zrot>{}".format(z0)

        Fs = [cond1]

        ipasa = self.object_by_surfaces(
            rotation_point, refractive_index, Fs, angle, v_globals={}
        )
        return ipasa


    def layer(self, r0: tuple[float, float], depth: float, refractive_index: complex | float | str,
              angle: float = 0*degrees, rotation_point: tuple[float, float] | None = None):
        """Insert a layer. If it is something else previous, it is removed.

        Args:
        r0 (float, float): (x0,z0) Location of the same plane, for example (0*um, 20*um)
        depth (float): depth of the layer
        refractive_index (float, str): refractive index , for example: 1.5 + 1.0j
        angle (float): angle of rotation of the semi-plane, in radians
        rotation_point (float, float). Rotation point
        """

        x0, z0 = r0
        if rotation_point is None:
            rotation_point = r0

        cond1 = "Zrot>{}".format(z0)
        cond2 = "Zrot<{}".format(z0 + depth)

        Fs = [cond1, cond2]

        ipasa = self.object_by_surfaces(
            rotation_point, refractive_index, Fs, angle, v_globals={}
        )
        return ipasa



    def square(self, r0: tuple[float, float], size: tuple[float, float],
                  refractive_index: complex | float | str, angle: float = 0*degrees,
                  rotation_point: tuple[float, float] | None = None):
        """Insert a square in background. Anything previous, is removed.

        Args:
            r0 (float, float): (x0,z0) Location of the square, for example (0*um, 20*um)
            size (float, float): x,z size of the square
            refractive_index (float, str): refractive index , for example: 1.5 + 1.0j
            angle (float): angle of rotation of the semi-plane, in radians
            rotation_point (float, float). Rotation point
        """

        x0, z0 = r0
        if rotation_point is None:
            rotation_point = r0

        if isinstance(size, (float, int, complex)):
            sizex, sizez = size, size
        else:
            sizex, sizez = size

        if isinstance(angle, (float, int, complex)):
            rotation_point = r0
        else:
            angle, rotation_point = angle

        # Definition of square/square
        xmin = x0 - sizex/2
        xmax = x0 + sizex/2
        zmin = z0 - sizez/2
        zmax = z0 + sizez/2

        cond1 = "Xrot<{}".format(xmax)
        cond2 = "Xrot>{}".format(xmin)
        cond3 = "Zrot<{}".format(zmax)
        cond4 = "Zrot>{}".format(zmin)

        Fs = [cond1, cond2, cond3, cond4]

        ipasa = self.object_by_surfaces(
            rotation_point, refractive_index, Fs, angle, v_globals={"np": np}
        )

        return ipasa


    @check_none('n',raise_exception=bool_raise_exception)
    def slit(self, r0: tuple[float, float], aperture: float, depth: float,
             refractive_index: complex | float | str, refractive_index_center: complex | float | str = "",
             angle: float = 0, rotation_point: tuple[float, float] | None = None):
        """Insert a slit in background.

        Args:
            r0 (float, float): (x0,z0) Location of the square, for example (0*um, 20*um)
            aperture (float): length of the opened part of the slit
            depth (float): depth of the slit
            refractive_index (float, str): refractive index , for example: 1.5 + 1.0j
            refractive_index_center (float, str?): refractive index of center
                if refractive_index_center='', [], 0 then we copy what it was previously at aperture
            angle (float): angle of rotation of the semi-plane, in radians
            rotation_point (float, float). Rotation point
        """

        x0, z0 = r0
        if rotation_point is None:
            rotation_point = r0

        n_back = deepcopy(self.n)

        cond1 = "Zrot>{}".format(z0)
        cond2 = "Zrot<{}".format(z0 + depth)
        cond3 = "Xrot<{}".format(x0 + aperture/2)
        cond4 = "Xrot>{}".format(x0 - aperture/2)

        Fs1 = [cond1, cond2]

        ipasa_slit = self.object_by_surfaces(
            r0, refractive_index, Fs1, angle, v_globals={}
        )

        Fs2 = [cond1, cond2, cond3, cond4]
        if refractive_index_center not in ("", [], 0):
            ipasa = self.object_by_surfaces(
                r0, refractive_index_center, Fs2, angle, v_globals={}
            )
        elif refractive_index_center in ("", [], 0):
            ipasa = self.object_by_surfaces(r0, 1, Fs2, angle, v_globals={})
            self.n[ipasa] = n_back[ipasa]
        return ipasa_slit != ipasa

    def cylinder(self, r0: tuple[float, float], radius: tuple[float, float],
               refractive_index: complex | float | str, angle: float = 0*degrees,
               rotation_point: tuple[float, float] | None = None):
        """Insert a cylinder in background.

        Args:
            r0 (float, float): (x0,z0) Location of the cylinder, for example (0*um, 20*um)
            radius (float, float): radius x,y of the cylinder (ellipsoid)
            refractive_index (float, str): refractive index , for example: 1.5 + 1.0j
            angle (float): angle of rotation of the semi-plane, in radians
            rotation_point (float, float). Rotation point
        """

        x0, z0 = r0
        if rotation_point is None:
            rotation_point = r0

        if isinstance(radius, (float, int, complex)):
            radiusx, radiusz = (radius, radius)
        else:
            radiusx, radiusz = radius

        cond = "(Xrot - {})**2 / {}**2 + (Zrot- {})**2 / {}**2 < 1".format(
            x0, radiusx, z0, radiusz
        )

        Fs = [cond]

        ipasa = self.object_by_surfaces(
            rotation_point, refractive_index, Fs, angle, v_globals={}
        )
        return ipasa


    def semi_cylinder(self, r0: tuple[float, float], radius: tuple[float, float],
                    refractive_index: complex | float | str, angle: float = 0*degrees,
                    rotation_point: tuple[float, float] | None = None):
        """Insert a semi_cylinder in background.

        Args:
            r0 (float, float): (x0,z0) Location of the square, for example (0*um, 20*um)
            radius (float, float): radius x,y of the cylinder (ellipsoid)
            refractive_index (float, str): refractive index , for example: 1.5 + 1.0j
            angle (float): angle of rotation of the semi-plane, in radians
            rotation_point (float, float). Rotation point
        """

        x0, z0 = r0
        if rotation_point is None:
            rotation_point = r0

        if isinstance(radius, (float, int, complex)):
            radiusx, radiusz = (radius, radius)
        else:
            radiusx, radiusz = radius

        cond1 = "Zrot>{}".format(z0)
        cond2 = "(Xrot - {})**2 / {}**2 + (Zrot- {})**2 / {}**2 < 1".format(
            x0, radiusx, z0, radiusz
        )

        Fs = [cond1, cond2]

        ipasa = self.object_by_surfaces(
            rotation_point, refractive_index, Fs, angle, v_globals={}
        )
        return ipasa


    def aspheric_surface_z(self, r0: tuple[float, float], refractive_index: complex | float | str,
                           cx: float, Qx: float, a2: float, a3: float, a4: float,
                           side: str, angle: float = 0*degrees):
        """Define an aspheric surface

        Args:
            r0 (float, float): (x0,z0) position of apex
            refractive_index (float, str): refractive index
            cx (float): curvature
            Qx (float): Conic constant
            side (str): 'left', 'right'

        Returns:
            numpy.array   : Bool array with positions inside the surface
        """

        x0, z0 = r0

        if isinstance(angle, (float, int, complex)):
            rotation_point = r0
        else:
            angle, rotation_point = angle

        if side == "right":
            sign = ">"
        elif side == "left":
            sign = "<"
        else:
            print("possible error in aspheric")

        params = dict(x0=x0, z0=z0, cx=cx, Qx=Qx,
                      a2=a2, a3=a3, a4=a4, sign=sign)

        cond = "Zrot{sign}{z0}+{cx}*(Xrot-{x0})**2/(1+np.sqrt(1-(1+{Qx})*{cx}**2*(Xrot-{x0})**2+{a2}*(Xrot-{x0})**4+{a3}*(Xrot-{x0})**6)+{a4}*(Xrot-{x0})**8)".format(
            **params
        )

        Fs = [cond]
        v_globals = {"self": self, "np": np, "degrees": degrees}

        ipasa = self.object_by_surfaces(
            rotation_point, refractive_index, Fs, angle, v_globals=v_globals
        )
        return ipasa


    def aspheric_lens(
            self,
            r0: tuple[float, float],
            angle: float,
            refractive_index: complex | float | str,
            cx: tuple[float, float],
            thickness: tuple[float, float],
            size: float,
            Qx: tuple[float, float]= (0, 0),
            a2: tuple[float, float] = (0, 0),
            a3: tuple[float, float] = (0, 0),
            a4: tuple[float, float] = (0, 0),
            a5: tuple[float, float] = (0, 0),
            a6: tuple[float, float] = (0, 0),
            a7=(0, 0)):
        """Define an aspheric surface as defined in Gomez-Pedrero.

        Args:
            r0 (float, float): position x,z of lens
            angle (float): rotation angle of lens + r0_rot
            cx (float, float): curvature radii
            Qx (float, float): Conic constant
            depth  (float, float): distance of the apex
            size (float): diameter of lens

        Returns:
            numpy.array   : Bool array with positions inside the surface
        """
        x0, z0 = r0
        if isinstance(angle, (float, int, complex)):
            rotation_point = r0
        else:
            angle, rotation_point = angle

        cx1, cx2 = cx
        Qx1, Qx2 = Qx
        a21, a22 = a2
        a31, a32 = a3
        a41, a42 = a4
        a51, a52 = a5
        a61, a62 = a6
        a71, a72 = a7
        side1, side2 = "left", "right"

        if side1 == "right":
            sign1 = "<"
        else:
            sign1 = ">"

        if side2 == "right":
            sign2 = "<"
        else:
            sign2 = ">"

        params = dict(
            cx1=cx1,
            Qx1=Qx1,
            cx2=cx2,
            Qx2=Qx2,
            x0=x0,
            a21=a21,
            a22=a22,
            a31=a31,
            a32=a32,
            a41=a41,
            a42=a42,
            a51=a51,
            a52=a52,
            a61=a61,
            a62=a62,
            a71=a71,
            a72=a72,
            d1=z0,
            d2=z0 + thickness,
            sign1=sign1,
            sign2=sign2,
        )

        cond1 = "Zrot{sign1}{d1}+{cx1}*(Xrot-{x0})**2/(1+np.sqrt(1-(1+{Qx1})*{cx1}**2*(Xrot-{x0})**2))+{a21}*(Xrot-{x0})**4+{a31}*(Xrot-{x0})**6+{a41}*(Xrot-{x0})**8+{a51}*(Xrot-{x0})**10+{a61}*(Xrot-{x0})**12+{a71}*(Xrot-{x0})**14".format(
            **params
        )

        cond2 = "Zrot{sign2}{d2}+{cx2}*(Xrot-{x0})**2/(1+np.sqrt(1+(1+{Qx2})*{cx2}**2*(Xrot-{x0})**2))+{a22}*(Xrot-{x0})**4+{a32}*(Xrot-{x0})**6+{a42}*(Xrot-{x0})**8+{a52}*(Xrot-{x0})**10+{a62}*(Xrot-{x0})**12+{a72}*(Xrot-{x0})**14".format(
            **params
        )

        cond3 = "(Xrot-{})<{}".format(x0, size/2)
        cond4 = "(Xrot-{})>{}".format(x0, -size/2)

        cond5 = "Zrot > {}".format(z0 - thickness)
        cond6 = "Zrot < {}".format(z0 + thickness)

        Fs = [cond1, cond2, cond3, cond4]
        v_globals = {"self": self, "np": np, "degrees": degrees}

        ipasa = self.object_by_surfaces(
            rotation_point, refractive_index, Fs, angle, v_globals=v_globals
        )
        
        if cx[0] != 0:
            r1 = 1/cx[0]
        else:
            r1 = 1e12
        
        if cx[1] != 0:
            r2 = 1/cx[1]
        else:
            r2 = 1e12
                  
        radii = (r1,r2)
        
        # https://en.wikipedia.org/wiki/Focal_length
        focal = ((refractive_index-1)*(1/radii[0]-1/radii[1] + (refractive_index-1)*thickness/(refractive_index*radii[0]*radii[1])))**(-1)


        return focal, ipasa


    def lens(self, r0: tuple[float, float], size: float, radii: tuple[float, float], thickness: float,
            refractive_index: float, angle: float = 0 * degrees, mask: tuple  | None= (50 * um, 1 + 2.05j)):
        """
        Lens defined by two radii of curvature and thickness.
        
        Args:
            r0 (tuple[float, float]): position of the initial point of the lens.
            size (float): _size of the lens, at x dimension
            radii (tuple[float, float]): radii of curvature of the two surfaces of the lens.
            thickness (float): thickness of the lens at the central axis.
            refractive_index (float): refractive index of the lens.
            angle (float, optional): angle of the lens. Defaults to 0*degrees.
            mask (tuple | None, optional): If not None, (thicknes, refractive index) of the pupil. Defaults to (50 * um, 1 + 2.05j).

        Reference:  
            https://en.wikipedia.org/wiki/Focal_length

        Returns:
            focal: focal distance of the lens (theoretical)
        """
        
        cx = (1/radii[0], 1/radii[1])
            
        focal, ipasa = self.aspheric_lens(r0, angle, refractive_index, cx, thickness, size)

        if mask is not None:
            mask_thickness = mask[0]
            mask_n = mask[1]
            
            self.slit(r0=(0, r0[0]+thickness/2-mask_thickness/2), aperture=size, depth=mask_thickness, refractive_index=mask_n)

                
        return focal, ipasa


    def wedge(
            self, r0: tuple[float, float], length, refractive_index: complex | float | str, angle_wedge: float,
            angle: float = 0*degrees, rotation_point: tuple[float, float] | None = None):
        """Insert a wedge pointing towards the light beam

        Args:
            r0 (float, float): (x0,z0) Location of the square, for example (0*um, 20*um)
            length (float): length of the long part (z direction)
            refractive_index (float, str): refractive index , for example: 1.5 + 1.0j
            angle_wedge (float), angle of the wedge in radians
            angle (float): angle of rotation of the semi-plane, in radians
            rotation_point (float, float). Rotation point
        """

        x0, z0 = r0
        if rotation_point is None:
            rotation_point = r0

        cond1 = "Xrot>{}".format(x0)
        cond2 = "Zrot<({}+{})".format(z0, length)
        cond3 = "(Xrot-{})<{}*(Zrot-{})".format(x0, np.tan(angle_wedge), z0)
        Fs = [cond1, cond2, cond3]

        ipasa = self.object_by_surfaces(
            rotation_point, refractive_index, Fs, angle, v_globals={}
        )
        return ipasa


    def prism(self, r0: tuple[float, float], length: float, refractive_index: complex | float | str,
            angle_prism: float, angle: float = 0*degrees, rotation_point: tuple[float, float] | None = None):
        """Similar to wedge but the use is different. Also the angle is usually different. One of the sides is paralel to x=x0.

        Args:
            r0 (float, float): (x0,z0) Location of the square, for example (0*um, 20*um)
            length (float): length of the long part (z direction)
            refractive_index (float, str): refractive index , for example: 1.5 + 1.0j
            angle_prism (float), angle of the prism in radians
            angle (float): angle of rotation of the semi-plane, in radians
            rotation_point (float, float). Rotation point
        """

        x0, z0 = r0
        if rotation_point is None:
            rotation_point = r0

        cond1 = "Xrot>{}".format(x0)
        cond2 = "Zrot-({})>{}*(Xrot-{})".format(z0,                     np.tan(angle_prism/2), x0)
        cond3 = "Zrot-({})<{}*(Xrot-{})".format(
            z0 + length, np.tan(np.pi - angle_prism/2), x0
        )

        Fs = [cond1, cond2, cond3]

        ipasa = self.object_by_surfaces(
            rotation_point, refractive_index, Fs, angle, v_globals={}
        )
        return ipasa


    def biprism(self, r0: tuple[float, float], length: float, height: float,
                refractive_index: complex | float | str, angle: float = 0*degrees):
        """Fresnel biprism.

        Args:
            r0 (float, float): (x0,z0) Location of the square, for example (0*um, 20*um)
            length (float): length of the long part (z direction)
            height (float): height of biprism
            refractive_index (float, str): refractive index , for example: 1.5 + 1.0j
            angle (float): angle of rotation of the semi-plane, in radians
        """

        x0, z0 = r0
        if isinstance(angle, (float, int, complex)):
            rotation_point = r0
        else:
            angle, rotation_point = angle

        vuelta = 1
        if vuelta == 1:
            cond1 = "Zrot>{}".format(z0)
            cond2 = "Zrot-({})<{}*(Xrot-{})".format(
                z0 + height, -2 * height / length, x0
            )
            cond3 = "Zrot-({})<{}*(Xrot-{})".format(
                z0 + height, 2 * height / length, x0
            )
        else:
            cond1 = "Zrot<{}".format(z0)
            cond2 = "Zrot-({})>{}*(Xrot-{})".format(
                z0 - height, -2 * height / length, x0
            )
            cond3 = "Zrot-({})>{}*(Xrot-{})".format(
                z0 - height, +2 * height / length, x0
            )

        Fs = [cond1, cond2, cond3]

        ipasa = self.object_by_surfaces(
            rotation_point, refractive_index, Fs, angle, v_globals={}
        )
        return ipasa


    def ronchi_grating(
            self,
            r0: tuple[float, float],
            period: float,
            fill_factor: float,
            length: float,
            height: float,
            Dx: float,
            refractive_index: complex | float | str,
            heigth_substrate: float,
            refractive_index_substrate: float,
            angle: float = 0*degrees):
        """Insert a ronchi grating in background.

        Args:
            r0 (float, float): (x0,z0) Location of the square, for example (0*um, 20*um)
            period (float): period of the grating
            fill_factor (float): [0,1], fill factor of the grating
            length (float): length of the grating
            height (float): height of the grating
            Dx (float): displacement of grating with respect x=0
            refractive_index (float, str): refractive index , for example: 1.5 + 1.0j
            heigth_substrate (float): height of the substrate
            refractive_index_substrate (float, str): refractive index of substrate,  1.5 + 1.0j
            angle (float): angle of rotation of the semi-plane, in radians
        """

        x0, z0 = r0

        Xrot, Zrot = self.__rotate__(angle, r0)

        t0 = Scalar_mask_X(x=self.x, wavelength=self.wavelength)
        t0.ronchi_grating(x0=Dx, period=period, fill_factor=fill_factor)

        self.extrude_mask(
            t=t0,
            z0=z0 + heigth_substrate/2,
            z1=z0 + heigth_substrate/2 + height,
            refractive_index=refractive_index,
            angle=angle)

        if heigth_substrate > 0:
            self.square(
                r0, (length, heigth_substrate), refractive_index_substrate, angle
            )
        self.slit(
            r0=(x0, z0 + heigth_substrate/2),
            aperture=length,
            depth=height,
            refractive_index=self.n_background,
            refractive_index_center="",
            angle=angle)

    def sine_grating(
            self,
            period: float,
            heigth_sine: float,
            heigth_substrate: float,
            r0: tuple[float, float],
            length: float,
            Dx: float,
            refractive_index: complex | float | str,
            angle: float = 0*degrees):
        """Insert a sine grating in background.

        Args:
            period (float): period of the grating
            fill_factor (float): [0,1], fill factor of the grating
            heigth_sine (float): height of the grating
            heigth_substrate (float): height of the substrate
            r0 (float, float): (x0,z0) Location of the square, for example (0*um, 20*um)
            length (float): length of the grating
            Dx (float): displacement of grating with respect x=0
            refractive_index (float, str): refractive index , for example: 1.5 + 1.0j
            angle (float): angle of rotation of the semi-plane, in radians
        """

        x0, z0 = r0
        Xrot, Zrot = self.__rotate__(angle, r0)

        c1 = Zrot < z0 + heigth_substrate - heigth_sine/2 + heigth_sine/2 * np.cos(
            2 * np.pi * (Xrot - x0 - Dx) / period
        )
        c2 = Zrot > z0
        conditionZ = c1 * c2  # no es sin, es square
        conditionX = (Xrot > x0 - length/2) * (Xrot < x0 + length/2)
        ipasa = conditionZ * conditionX
        self.n[ipasa] = refractive_index
        return ipasa


    def probe(self, r0: tuple[float, float], base: float, length: float,
              refractive_index: complex | float | str, angle: float = 0*degrees):
        """Probe with a sinusoidal shape.

        Args:
            r0 (float, float): (x0,z0) position of the center of base, for example (0*um, 20*um)
            base (float): base of the probe
            length (float): length of the graprobeing
            Dx (float): displacement of grating with respect x=0
            refractive_index (float, str): refractive index , for example: 1.5 + 1.0j
            angle (float): angle of rotation of the semi-plane, in radians
        """

        x0, z0 = r0
        if isinstance(angle, (float, int, complex)):
            rotation_point = r0
        else:
            angle, rotation_point = angle

        cond1 = "Zrot<{}+{}/2*np.cos(2*np.pi*Xrot/{})".format(
            length - z0, length, base)
        cond2 = "Xrot<{}".format(x0 + base/2)
        cond3 = "Xrot>{}".format(x0 - base/2)
        cond4 = "Zrot>{}".format(z0)
        Fs = [cond1, cond2, cond3, cond4]
        v_globals = {"self": self, "np": np, "degrees": degrees, "um": um}
        ipasa = self.object_by_surfaces(
            rotation_point, refractive_index, Fs, angle, v_globals=v_globals
        )
        return ipasa

    @check_none('x','z',raise_exception=bool_raise_exception)
    def rough_sheet(self, r0: tuple[float, float], size: float, t: float, s: float,
                    refractive_index: complex | float | str, angle: float = 0*degrees,
                    rotation_point: tuple[float, float] | None = None):
        """Sheet with one of the surface rough.

        Args:
            r0 (float, float):(x0,z0) Location of cylinder, for example (0*um, 20*um)
            size (float, float): (sizex, sizez) size of the sheet
            s (float): std roughness
            t (float): correlation length of roughness
            refractive_index (float, str): refractive index
            angle (float): angle
            rotation_point (float, float): rotation point

        Returns:
            (numpy.array): ipasa, indexes [iz,ix] of surface

        References:
            According to Ogilvy p.224
        """

        x0, z0 = r0
        if rotation_point is None:
            rotation_point = r0

        if isinstance(size, (float, int, complex)):
            sizex, sizez = size, size
        else:
            sizex, sizez = size

        k = 2 * np.pi / self.wavelength
        xmin = x0 - sizex/2
        xmax = x0 + sizex/2

        # I do not want to touch the previous

        n_back = deepcopy(self.n)

        h_corr = roughness_1D(self.x, t, s)

        fx = h_corr / (k * (refractive_index - 1))  # heights

        cond1 = "Zrot>{}".format(z0)
        cond2 = "Xrot<{}".format(xmax)
        cond3 = "Xrot>{}".format(xmin)

        Fs = [cond1, cond2, cond3]

        ipasa = self.object_by_surfaces(
            rotation_point, refractive_index, Fs, 0, v_globals={}
        )

        i_z, _, _ = nearest2(self.z, z0 + sizez - fx)
        i_final = len(self.z)
        for i in range(len(self.x)):
            self.n[i_z[i]: i_final, i] = n_back[i_z[i]: i_final, i]

        if angle != 0:
            self.rotate_field(angle, rotation_point,   n_background=self.n_background)
        return ipasa