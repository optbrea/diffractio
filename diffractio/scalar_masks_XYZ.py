# !/usr/bin/env python3

# ----------------------------------------------------------------------
# Name:        scalar_masks_XYZ.py
# Purpose:     Define Scalar_mask_XYZ class for creating 3D scalar masks
#
# Author:      Luis Miguel Sanchez Brea
#
# Created:     2024
# Licence:     GPLv3
# ----------------------------------------------------------------------


"""
This module generates Scalar_mask_XYZ class for definingn masks. Its parent is scalar_fields_XYZ.

The main atributes are:
    * self.x - x positions of the field
    * self.y - y positions of the field
    * self.z - z positions of the field
    * self.u - field XYZ
    * self.n - refractive index XYZ
    * self.wavelength - wavelength of the incident field. The field is monochromatic

The magnitude is related to microns: `micron = 1.`

*Class for unidimensional scalar masks*

*Functions*
    * object_by_surfaces
    * sphere
    * square
    * cylinder
"""

# flake8: noqa


from .__init__ import degrees, np, um
from .config import bool_raise_exception
from .utils_typing import npt, Any, NDArray,  NDArrayFloat, NDArrayComplex
from .utils_common import check_none
from .utils_math import nearest

from .scalar_fields_XYZ import Scalar_field_XYZ
from .scalar_masks_XY import Scalar_mask_XY
from .scalar_masks_XZ import Scalar_mask_XZ
from .utils_drawing3D import load_stl, voxelize_volume_diffractio

class Scalar_mask_XYZ(Scalar_field_XYZ):

    def __init__(self, x: NDArrayFloat | None = None, y: NDArrayFloat | None = None,
                 z: NDArrayFloat | None = None, wavelength: float | None = None,
                 n_background: float = 1., info: str = ""):
        super().__init__(x, y, z, wavelength, n_background, info)
        self.type = 'Scalar_mask_XYZ'



    @check_none('X', 'Y', 'Z', 'n', raise_exception=bool_raise_exception)
    def mask_from_function(
            self, r0: tuple[float, float, float], refractive_index: complex | float | str, fs: tuple[str], rotation: dict | None, v_globals: dict = {}):
        """
        Phase mask defined between two surfaces f1 and f1: h(x,z)=f2(x,z)-f1(x,z)

        Args:
            r0 (float, float): location of the mask
            refractive_index (float, str): can be a number or a function n(x,z)
            fs (tuple[str]): functions that delimits the volume
            rotation (dict): dictionary with the rotation parameters
            v_globals (dict): dict with global variables
        """


        if rotation is not None:
            if rotation['kind']== 'point' and rotation['point'] is None:
                rotation['point'] = r0

            Xrot, Yrot, Zrot = self.__XYZ_rotate__(rotation)
        else:
            Xrot, Yrot, Zrot = self.X, self.Y, self.Z
             
        v_locals = {"self": self, "np": np, "degrees": degrees, "um": um, "Xrot": Xrot, "Yrot": Yrot, "Zrot": Zrot}
    
        F = []
        for i, fi in enumerate(fs):
            Fi = eval(fi, v_globals, v_locals)
            F.append(Fi)

        ipasa = np.ones_like(self.X, dtype=bool)
        for i, Fi in enumerate(F):
            ipasa = np.bitwise_and(ipasa.astype(bool),Fi.astype(bool))

        self.n[ipasa] = refractive_index
        return ipasa

    @check_none('X', 'Y', 'Z', raise_exception=bool_raise_exception)
    def object_by_surfaces(self,
                           r0: tuple[float, float,float],
                           refractive_index: float,
                           Fs,
                           rotation: dict | None = None,
                           v_globals={}):
        """  Mask defined by n surfaces given in array Fs={f1, f2,    h(x,y,z)=f1(x,y,z)*f2(x,y,z)*....*fn(x,y,z)

        Args:
            r0 (float, float, float): location of the mask
            refractive_index (float, str): can be a number or a function n(x, y,z)
            Fs (tuple): condtions as str that will be computed using eval
            array1 (numpy.array): array (x,y,z) that delimits the second surface
            v_globals (dict): dict with global variables
        """

        x0, y0, z0 = r0

        
        if rotation is not None:
            if rotation['kind']== 'point' and rotation['point'] is None:
                rotation['point'] = r0

            Xrot, Yrot, Zrot = self.__XYZ_rotate__(rotation)
        else:
            Xrot, Yrot, Zrot = self.X, self.Y, self.Z

        v_locals = {'self': self, 'np': np, 'degrees': degrees, 'um': um, 
                    'Xrot': Xrot, 'Yrot': Yrot, 'Zrot': Zrot, 
                    'x0': x0, 'y0': y0, 'z0': z0}


        conditions = []
        for fi in Fs:
            result_condition = eval(fi, v_globals, v_locals)
            conditions.append(result_condition)

        ipasa = conditions[0]
        for cond in conditions:
            ipasa = ipasa & cond

        self.n[ipasa] = refractive_index
        return ipasa



    def extrude_mask_XY(self, txy: Scalar_mask_XY, refractive_index: float | complex | None, z0: float | None = None, z1: float | None = None, keep_rest = True,
                        v_globals: dict = {}):
        """
        Converts a Scalar_mask_X in volumetric between z0 and z1 by growing between these two planes.
        
        Args:
            t (Scalar_mask_X): an amplitude mask of type Scalar_mask_X.
            refractive_index (float, str): can be a number or a function n(x,z). If none It just substitutes
            z0 (float): initial position of mask
            z1 (float): final position of mask
        """

        if z0 == None:
            iz0 = 0
        else: 
            iz0, _, _ = nearest(vector=self.z, number=z0)
            
        if z1 == None:
            iz1 = len(self.z)
        else:
            iz1, _, _ = nearest(vector=self.z, number=z1)
            
            
        num_layers = iz1-iz0

        
        layer = txy.u
        layer = layer.astype(complex)

        #zone= np.tile(layer,(1,1,num_layers)).reshape(len(self.y), len(self.x), num_layers)
        # print(zone.shape)

        # print(self.n.shape)
        # self.n[:,:,iz0:iz1] =zone
        # self.n = self.n.astype(complex)

        for index in range(iz0, iz1):
            i_mask = np.abs(txy.u)>0
            i_background = np.logical_not(i_mask)
            layer[i_mask]=refractive_index
            if keep_rest is False:
                layer[i_background]=txy.n_background
            else:
                layer[i_background]=self.n[i_background,index]
            self.n[:,:,index]=layer

        self.n = self.n.astype(complex)



    def extrude_mask_XZ(self, txz: Scalar_mask_XZ, y0: float | None, y1: float  | None, 
                        refractive_index: float | None, n_new: float | None = None, 
                        v_globals: dict = {}):
        """
        Converts a Scalar_mask_X in volumetric between z0 and z1 by growing between these two planes
        Args:
            t (Scalar_mask_X): an amplitude mask of type Scalar_mask_X.
            y0 (float): initial  position of mask
            z1 (float): final position of mask
            refractive_index (float, None): If None takes the value of txz.n directly. TODO: can be a number or a function n(x,z)
        """

        if y0 == None:
            iy0 = 0
        else: 
            iy0, _, _ = nearest(vector=self.y, number=y0)
            
        if y1 == None:
            iy1 = len(self.y)
        else:
            iy1, _, _ = nearest(vector=self.y, number=y1)
                

        i_mask = np.abs(txz.n)>txz.n_background
        i_background = np.logical_not(i_mask)


        layer = txz.n
        
        if refractive_index is not None:
            if n_new is not None:
                layer[i_mask]=n_new
            else:
                layer[i_mask]=refractive_index
                
        layer[i_background]=self.n_background


        self.n[iy0:iy1,:,:] = np.tile(layer.transpose(),(iy1-iy0,1,1))
        self.n = self.n.astype(complex)


    #@check_none('X', 'Y', 'Z', raise_exception=bool_raise_exception)
    def sphere(self, r0: tuple[float, float, float], radius: tuple[float],
                refractive_index: float, rotation: dict | None = None) -> bool:
        """ Insert a cylinder in background. If something previous, is removed.

        Args:
            r0 (float, float, float): (x0, y0,z0) Location of the square, for example (0*um, 0*um, 0*um)
            radius (float,float): x,y, size of the circular part of cylinder
            length (float): length of cylidner
            refractive_index (float, str): refractive index , for example: 1.5 + 1.0j
            rotation (dict): kind: 'axis' or 'point'
                            if 'axis': angle (float) and axis (tuple[float,float,float])
                            if 'point': angle (tuple[float,float,float]) and point (tuple[float,float,float])
 
        """

        x0, y0, z0 = r0

        if isinstance(radius, (float, int, complex)):
            radius = (radius, radius, radius)

        radiusx, radiusy, radiusz = radius
        
        if rotation is not None:
            if rotation['kind']== 'point' and rotation['point'] is None:
                rotation['point'] = r0

            Xrot, Yrot, Zrot = self.__XYZ_rotate__(rotation)
        else:
            Xrot, Yrot, Zrot = self.X, self.Y, self.Z
            
            

        ipasa = (Xrot - x0)**2 / radiusx**2 + (
            Yrot - y0)**2 / radiusy**2 + (Zrot - z0)**2 / radiusz**2 < 1

        self.n[ipasa] = refractive_index

        return ipasa


    @check_none('X', 'Y', 'Z', raise_exception=bool_raise_exception)
    def cube(self,
               r0: tuple[float, float, float],
               size: tuple[float, float, float],
               refractive_index: float,
               rotation: dict | None = None) -> bool:
        """ Insert a square in background. If something previous, is removed.

        Args:
            r0 (float, float, float): (x0, y0,z0) Location of the square, for example (0*um, 0*um, 0*um)
            size (float, float, float): x,y,z size of the square
            refractive_index (float, str): refractive index , for example: 1.5 + 1.0j
            rotation (dict): kind: 'axis' or 'point'
                            if 'axis': angle (float) and axis (tuple[float,float,float])
                            if 'point': angle (tuple[float,float,float]) and point (tuple[float,float,float])

        """
        x0, y0, z0 = r0

        if isinstance(r0, (float, int, complex)):
            r0 = (r0[0], r0[0], r0[0])
        if len(size) == 1:
            size = (size[0], size[0], size[0])

        size_x, size_y, size_z = size
        
        if rotation is not None:
            if rotation['kind']== 'point' and rotation['point'] is None:
                rotation['point'] = r0

            Xrot, Yrot, Zrot = self.__XYZ_rotate__(rotation)
        else:
            Xrot, Yrot, Zrot = self.X, self.Y, self.Z


        ipasax1 = Xrot >= x0 - size_x/2
        ipasax2 = Xrot <= x0 + size_x/2
        ipasay1 = Yrot >= y0 - size_y/2
        ipasay2 = Yrot <= y0 + size_y/2
        ipasaz1 = Zrot >= z0 - size_z/2
        ipasaz2 = Zrot <= z0 + size_z/2
        ipasa = ipasax1 * ipasax2 * ipasay1 * ipasay2 * ipasaz1 * ipasaz2
        self.n[ipasa] = refractive_index

        return ipasa


    @check_none('X', 'Y', 'Z', raise_exception=bool_raise_exception)
    def cylinder(self, r0: tuple[float], radius: tuple[float], length: float,
                 refractive_index: float, rotation: dict | None = None):
        """ Insert a cylinder in background. If something previous, is removed.

        Args:
            r0 (float, float, float): (x0, y0,z0) Location of the square, for example (0*um, 0*um, 0*um)
            radius (float,float): x,y, size of the circular part of cylinder
            length (float): length of cylidner
            refractive_index (float, str): refractive index , for example: 1.5 + 1.0j
            rotation (dict): kind: 'axis' or 'point'
                            if 'axis': angle (float) and axis (tuple[float,float,float])
                            if 'point': angle (tuple[float,float,float]) and point (tuple[float,float,float])

        """
        
        if isinstance(radius, (float, int, complex)):
            radius = (radius, radius)

        x0, y0, z0 = r0
        radiusx, radiusy = radius
        
        if rotation is not None:
            if rotation['kind']== 'point' and rotation['point'] is None:
                rotation['point'] = r0

            Xrot, Yrot, Zrot = self.__XYZ_rotate__(rotation)
        else:
            Xrot, Yrot, Zrot = self.X, self.Y, self.Z


        ipasar = (Xrot - x0)**2 / radiusx**2 + (Yrot - y0)**2 / radiusy**2 <= 1
        ipasaz1 = Zrot >= z0 - length/2
        ipasaz2 = Zrot <= z0 + length/2
        ipasa = ipasar * ipasaz1 * ipasaz2

        self.n[ipasa] = refractive_index

        return ipasa


    def aspheric_lens(
            self,
            r0: tuple[float, float, float],
            refractive_index: complex | float | str,
            thickness: tuple[float, float],
            cx: tuple[float, float],
            diameter: float | None = None,
            mask: tuple  | None= None,
            Qx: tuple[float, float]= (0, 0),
            a: tuple[tuple[float,float]] | None = None,
            rotation: dict | None = None,):
        """Define an aspheric surface as defined in Gomez-Pedrero.

        Args:
            r0 (float, float): position x,z of lens
            refractive_index (float, str): refractive index , for example: 1.5 + 1.0j
            thickness  (float, float): distance of the apex
            diamater (float |None): diameter of the lens
            mask (tuple): mask to apply to the surface (thickness, refractive_index)
            cx (float, float): curvature radii
            Qx (float, float): Conic constant
            a7  (float, float): Aspheric coefficients a2, a3, a4, a5, a6, a7. Each (float, float)
            rotation (dict): kind: 'axis' or 'point'
                if 'axis': angle (float) and axis (tuple[float,float,float])
                if 'point': angle (tuple[float,float,float]) and point (tuple[float,float,float])

        Example: 
            rotation = dict(kind = 'axis', point=(0,0,0), axis=(1,0,0), angle=5*degrees)
            

        Returns:
            numpy.array   : Bool array with positions inside the surface
        """
        x0, y0, z0 = r0

        radius = diameter/2
        
        
        if a is None:
            a2 = (0, 0)
            a3 = (0, 0)
            a4 = (0, 0)
            a5 = (0, 0)
            a6 = (0, 0)
            a7 = (0, 0)
        else:  
            a2, a3, a4, a5, a6, a7 = a
            

        if rotation is not None:
            if rotation['kind']== 'point' and rotation['point'] is None:
                rotation['point'] = r0
            Xrot, Yrot, Zrot = self.__XYZ_rotate__(rotation)
        else:
            Xrot, Yrot, Zrot = self.X, self.Y, self.Z
            
            
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

        R =f'np.sqrt((Xrot-{x0})**2 + (Yrot-{y0})**2)'


        params = dict(
            cx1=cx1,
            Qx1=Qx1,
            cx2=cx2,
            Qx2=Qx2,
            x0=x0,
            y0=y0,
            z0=z0,
            radius=radius,
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


        cond1 = "Zrot{sign1}{d1}+{cx1}*("+R+")**2/(1+np.sqrt(1-(1+{Qx1})*{cx1}**2*("+R+")**2))+{a21}*("+R+")**4+{a31}*("+R+")**6+{a41}*("+R+")**8+{a51}*("+R+")**10+{a61}*("+R+")**12+{a71}*("+R+")**14"

        cond1 = cond1.format(**params)

        cond2 = "Zrot{sign2}{d2}+{cx2}*("+R+")**2/(1+np.sqrt(1+(1+{Qx2})*{cx2}**2*("+R+")**2))+{a22}*("+R+")**4+{a32}*("+R+")**6+{a42}*("+R+")**8+{a52}*("+R+")**10+{a62}*("+R+")**12+{a72}*("+R+")**14"

        cond2 = cond2.format(**params)


        cond3 = "("+R+")**2<{radius}**2"
        cond3 = cond3.format(**params)


        Fs = [cond1, cond2, cond3]
        v_globals = {"self": self, "np": np, "degrees": degrees}

        ipasa = self.object_by_surfaces(
            r0, refractive_index, Fs, rotation, v_globals=v_globals
        )

        if mask is not None:
            pupil_xy = Scalar_mask_XY(x=self.x, y=self.y,  wavelength=self.wavelength)
            pupil_xy.circle(r0=(0,0), radius=(400*um, 400*um))
            pupil_xy.u = 1-pupil_xy.u
            self.extrude_mask_XY(txy=pupil_xy, z0=z0, z1=z0+thickness, 
                refractive_index=2+2.j)

        
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


    def lens(self, r0: tuple[float, float], 
             diameter: float,
             radii: tuple[float, float],
             thickness: float,
             refractive_index: float, 
             mask: tuple  | None= (50 * um, 1 + 2.05j), 
             rotation: dict | None = None):
        """
        Lens defined by two radii of curvature and thickness.
        
        Args:
            r0 (tuple[float, float]): position of the initial point of the lens.
            size (float): _size of the lens, at x dimension
            radii (tuple[float, float]): radii of curvature of the two surfaces of the lens.
            thickness (float): thickness of the lens at the central axis.
            refractive_index (float): refractive index of the lens.
            angles (float, optional): angles of the lens. Defaults to 0*degrees.
            mask (tuple | None, optional): If not None, (thicknes, refractive index) of the pupil. Defaults to (50 * um, 1 + 2.05j).

        Reference:  
            https://en.wikipedia.org/wiki/Focal_length


        Example: 
            rotation = dict(kind = 'axis', point=(0,0,0), axis=(1,0,0), angle=5*degrees)
            


        Returns:
            focal: focal distance of the lens (theoretical)
        """
        
        cx = (1/radii[0], 1/radii[1])
        Qx = (0,0)
            
        focal, ipasa = self.aspheric_lens(r0=r0, refractive_index=refractive_index, thickness=thickness, cx=cx,
                                           diameter=diameter, mask= mask, Qx=Qx, a=None, rotation=rotation)

        
        return focal, ipasa



    @check_none('x', 'y', 'z', raise_exception=bool_raise_exception)
    def stl(self, filename: str, refractive_index: float, Dx: float | None = None, Dy: float | None = None, 
            Dz: float | None = None, has_draw: bool = False, verbose: bool = False):
        """
        stl file

        Include a stl part 

        Args:
            filename (str): _description_
            refractive_index (float): _description_
            Dx (float | None, optional): _description_. Defaults to None.
            Dy (float | None, optional): _description_. Defaults to None.
            Dz (float | None, optional): _description_. Defaults to None.
            has_draw (bool, optional): _description_. Defaults to False.
            verbose (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        
        mesh = load_stl(filename, has_draw=has_draw)
        
        bounds = mesh.bounds

        if Dx is None:
            x = self.x
        else:
            x = np.linspace(bounds[0]-Dx[0], bounds[1]+Dx[1], len(self.x))

        if Dy is None:
            y = self.y
        else:
            y = np.linspace(bounds[2]-Dy[0], bounds[3]+Dy[1], len(self.y))

        if Dz is None:
            z = self.z
        else:
            z = np.linspace(bounds[4]-Dz[0], bounds[5]+Dz[1], len(self.z))

        self.x = x
        self.y = y
        self.z = z
        
        self.X, self.Y, self.Z = np.meshgrid(x, y, z)
        self, voi= voxelize_volume_diffractio(self, mesh, refractive_index = refractive_index)

        return voi, mesh, bounds


