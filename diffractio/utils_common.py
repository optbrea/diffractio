# !/usr/bin/env python3


# ----------------------------------------------------------------------
# Name:        utils_common.py
# Purpose:     Common utility functions for various operations
#
# Author:      Luis Miguel Sanchez Brea
#
# Created:     2017
# Licence:     GPLv3
# ----------------------------------------------------------------------

""" Common functions to classes """
# flake8: noqa

import datetime
import multiprocessing

import sys
import numpy as np
import psutil
from scipy.io import loadmat, savemat
from scipy.ndimage import center_of_mass

from .config import (bool_raise_exception, Options_add, Options_sub, Options_rmul,
                     get_scalar_options, get_vector_options)



def check_none(*variables, raise_exception = bool_raise_exception):
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            for variable in variables:
                if getattr(self, variable) is None:
                    if raise_exception:
                        raise ValueError(f"{variable} is None")
                    else:
                        print(f"{variable} is None")
                    return  # Return immediately, do not execute the method
            return func(self, *args, **kwargs)
        return wrapper
    return decorator


@check_none('Ex', 'Ey', 'Ez', raise_exception=bool_raise_exception)
def get_vector(cls, kind: get_vector_options, mode='modulus', **kwargs):
    """Takes the vector field and divide in Scalar_field_X.

    Args:
        kind (str): 'fields', 'intensity', 'intensities', 'phases', 'poynting_vector', 'poynting_vector_averaged', 
        'poynting_total', 'energy_density', 'irradiance', 'stokes', 'params_ellipse', 

    Returns:
        Vector_field_X: (Ex, Ey, Ez),
    """

    if kind == "E":
        return cls.Ex, cls.Ey, cls.Ez
    
    if kind == "H":
        return cls.Hx, cls.Hy, cls.Hz

    if kind == "EH" or kind == 'fields':
        return (cls.Ex, cls.Ey, cls.Ez), (cls.Hx, cls.Hy, cls.Hz)

    elif kind == "intensity":
        intensity = np.abs(cls.Ex) ** 2 + np.abs(cls.Ey) ** 2 + np.abs(cls.Ez) ** 2
        return intensity

    elif kind == "intensities":
        intensity_x = np.abs(cls.Ex) ** 2
        intensity_y = np.abs(cls.Ey) ** 2
        intensity_z = np.abs(cls.Ez) ** 2
        return intensity_x, intensity_y, intensity_z

    elif kind == "phases":
        phase_x = np.angle(cls.Ex)
        phase_y = np.angle(cls.Ey)
        phase_z = np.angle(cls.Ez)
        return phase_x, phase_y, phase_z

    elif kind == 'poynting_vector':
        Sx = np.real(cls.Ey * cls.Hz - cls.Ez * cls.Hy)
        Sy = np.real(cls.Ez * cls.Hx - cls.Ex * cls.Hz)
        Sz = np.real(cls.Ex * cls.Hy - cls.Ey * cls.Hx)
        return Sx, Sy, Sz

    elif kind == 'poynting_vector_averaged':
        Sx = np.real(cls.Ey * cls.Hz.conjugate() - cls.Ez * cls.Hy.conjugate()).squeeze()
        Sy = np.real(cls.Ez * cls.Hx.conjugate() - cls.Ex * cls.Hz.conjugate()).squeeze()
        Sz = np.real(cls.Ex * cls.Hy.conjugate() - cls.Ey * cls.Hx.conjugate()).squeeze()
        return Sx, Sy, Sz

    elif kind == 'poynting_total':
        Sx, Sy, Sz = cls.get('poynting_vector_averaged')

        S_total = np.sqrt(np.abs(Sx)**2 + np.abs(Sy)**2 + np.abs(Sz)**2)
        return S_total

    elif kind == 'energy_density':
        permeability = 4*np.pi*1e-7
        epsilon = cls.n**2 * permeability
        U = epsilon * np.real(np.abs(cls.Ex)**2 + np.abs(cls.Ey)**2 + np.abs(cls.Ez)**2) + permeability * (np.abs(cls.Hx)**2 + np.abs(cls.Hy)**2 + np.abs(cls.Hz)**2)
        return U
    
    elif kind == 'energy_density2':
        permeability = 4*np.pi*1e-7
        c_light = 299792458*1e6
        epsilon = cls.n**2 / (permeability*c_light**2)
        U = epsilon * np.real(np.abs(cls.Ex)**2 + np.abs(cls.Ey)**2 + np.abs(cls.Ez)**2) + permeability * (np.abs(cls.Hx)**2 + np.abs(cls.Hy)**2 + np.abs(cls.Hz)**2)
        return 0.5*U*1e8

    elif kind == 'irradiance':
        permeability = 4*np.pi*1e-7
        epsilon = cls.n**2 * permeability

        Sx, Sy, Sz = cls.get('poynting_vector_averaged')
        
        if mode == 'modulus':
            irradiance = np.sqrt(Sx**2 + Sy**2 + Sz**2)
            
        elif mode == 'Sz':
            irradiance = Sz
            
        elif isinstance(mode, (list, tuple, np.ndarray)):
            mode = np.array(mode)
            mode = mode/np.linalg.norm(mode)
            irradiance = mode[0] * Sx + mode[1] * Sy + mode[2] * Sz
        return irradiance

    elif kind == 'stokes':
        # S0, S1, S2, S3
        S0 = np.abs(cls.Ex) ** 2 + np.abs(cls.Ey) ** 2
        S1 = np.abs(cls.Ex) ** 2 - np.abs(cls.Ey) ** 2
        S2 = 2 * np.real(cls.Ex * np.conjugate(cls.Ey))
        S3 = 2 * np.imag(cls.Ex * np.conjugate(cls.Ey))
        return S0, S1, S2, S3

    elif kind == "params_ellipse":
        # A, B, theta, h
        S0, S1, S2, S3 = cls.get('stokes')
        Ip = np.sqrt(S1**2 + S2**2 + S3**2)
        L = S1 + 1.0j * S2

        A = np.real(np.sqrt(0.5 * (Ip + np.abs(L))))
        B = np.real(np.sqrt(0.5 * (Ip - np.abs(L))))
        theta = 0.5 * np.angle(L)
        h = np.sign(S3)
        return A, B, theta, h

    else:
        print("The parameter '{}'' in .get(kind='') is wrong. Use one of this: {}".format(kind, get_vector_options))



def get_scalar(cls, kind: get_scalar_options):
    """Get parameters from Scalar field

    Args:
        kind (str): 'intensity', 'phase', 'field'

    Returns:
        matrices with required values
    """

    if kind == "intensity":
        intensity = np.abs(cls.u) ** 2
        return intensity

    elif kind == "phase":
        phase = np.angle(cls.u)
        return phase

    if kind == "field":
        intensity = np.abs(cls.u) ** 2
        phase = np.angle(cls.u)
        return intensity, phase


def oversampling(cls, factor_rate: int | tuple):# -> Any:
    """Function to oversampling the field

    Args:
        factor_rate (int | tuple, optional): factor rate. Defaults to 2.
    """


    if cls.type in ('Scalar_mask_X', 'Scalar_field_X', 'Scalar_source_X') :

        cls.x = np.linspace(cls.x[0], cls.x[-1], factor_rate*len(cls.x))
        cls.u =  cls.u.repeat(factor_rate[0],axis=0)


    if cls.type in ('Scalar_field_Z') :

        cls.z = np.linspace(cls.z[0], cls.z[-1], factor_rate*len(cls.x))
        cls.u =  cls.u.repeat(factor_rate[0],axis=0)


    elif cls.type in ('Scalar_mask_XY', 'Scalar_field_XY', 'Scalar_source_XY') :
        
        if isinstance(factor_rate, int):
            factor_rate = (factor_rate, factor_rate)    

        cls.x = np.linspace(cls.x[0], cls.x[-1], factor_rate[0]*len(cls.x))
        cls.y = np.linspace(cls.y[0], cls.y[-1], factor_rate[1]*len(cls.y))
        cls.X, cls.Y = np.meshgrid(cls.x, cls.y)

        new_matrix =  cls.u.repeat(factor_rate[0],axis=0)
        cls.u =  new_matrix.repeat(factor_rate[1],axis=1)

    elif cls.type in ('Scalar_mask_XYZ', 'Scalar_field_XYZ') :

        if isinstance(factor_rate, int):
            factor_rate = (factor_rate, factor_rate, factor_rate)

        cls.x = np.linspace(cls.x[0], cls.x[-1], factor_rate[0]*len(cls.x))
        cls.y = np.linspace(cls.y[0], cls.y[-1], factor_rate[1]*len(cls.y))
        cls.z = np.linspace(cls.z[0], cls.z[-1], factor_rate[2]*len(cls.z))
        cls.X, cls.Y, cls.Z = np.meshgrid(cls.x, cls.y, cls.z)

        new_matrix =  cls.u.repeat(factor_rate[0],axis=0)
        new_matrix =  new_matrix(factor_rate[1],axis=1)
        cls.u =  new_matrix.repeat(factor_rate[2],axis=2)

        new_matrix =  cls.n.repeat(factor_rate[0],axis=0)
        new_matrix =  new_matrix(factor_rate[1],axis=1)
        cls.n =  new_matrix.repeat(factor_rate[2],axis=2)

    elif cls.type in ('Scalar_mask_XZ', 'Scalar_field_XZ') :
        
        if isinstance(factor_rate, int):
            factor_rate = (factor_rate, factor_rate)

        cls.x = np.linspace(cls.x[0], cls.x[-1], factor_rate[0]*len(cls.x))
        cls.z = np.linspace(cls.z[0], cls.z[-1], factor_rate[1]*len(cls.z))
        cls.X, cls.Z = np.meshgrid(cls.x, cls.z)

        new_matrix =  cls.u.repeat(factor_rate[0],axis=0)
        cls.u =  new_matrix.repeat(factor_rate[1],axis=1)

        new_matrix =  cls.n.repeat(factor_rate[0],axis=0)
        cls.n =  new_matrix.repeat(factor_rate[1],axis=1)

    return cls


def add(self, other, kind: Options_add  = 'source'):
    """adds two fields. For example two light sources or two masks. The fields are added as complex numbers and then normalized so that the maximum amplitude is 1.
    
    Args:
        other (Other field): 2nd field to add
        kind (str): instruction how to add the fields: ['source', 'mask', 'refractive_index', 'phases', 'no_overlap', 'distances'].
            - 'source': adds the fields as they are
            - 'mask': adds the fields as complex numbers and then normalizes so that the maximum amplitude is 1.
            - 'phases': adds the phases and then normalizes so that the maximum amplitude is 1.
            - 'np_overlap': adds the fields as they are. If the sum of the amplitudes is greater than 1, an error is produced
            - 'distances': adds the fields as they are. If the fields overlap, the field with the smallest distance is kept.

    Returns:
        sum of the two fields.
    """
    
    from diffractio.scalar_sources_X import Scalar_source_X
    from diffractio.scalar_sources_XY import Scalar_source_XY
    from diffractio.scalar_masks_X import Scalar_mask_X
    from diffractio.scalar_masks_XY import Scalar_mask_XY
    from diffractio.scalar_fields_XZ import Scalar_field_XZ
    from diffractio.scalar_fields_XY import Scalar_field_XY
    from diffractio.scalar_masks_XZ import Scalar_mask_XZ
    from diffractio.scalar_fields_Z import Scalar_field_Z
    from diffractio.scalar_fields_XYZ import Scalar_field_XYZ
    from diffractio.scalar_masks_XYZ import Scalar_mask_XYZ

    if isinstance(self, Scalar_mask_XY):
        t = Scalar_mask_XY(self.x, self.y, self.wavelength)
    elif isinstance(self, Scalar_source_XY):
        t = Scalar_source_XY(self.x, self.y, self.wavelength)
    elif isinstance(self, Scalar_mask_X):
        t = Scalar_mask_X(self.x, self.wavelength)
    elif isinstance(self, Scalar_source_X):
        t = Scalar_source_X(self.x,  self.wavelength)
    elif isinstance(self, Scalar_mask_XZ):
        t = Scalar_mask_XZ(self.x, self.z, self.wavelength)
    elif isinstance(self, Scalar_field_XZ):
        t = Scalar_field_XZ(self.x, self.z, self.wavelength)
    elif isinstance(self, Scalar_field_Z):
        t = Scalar_field_Z(self.z, self.wavelength)
    elif isinstance(self, Scalar_field_XYZ):
        t = Scalar_field_XYZ(self.x, self.y, self.z, self.wavelength)
    elif isinstance(self, Scalar_mask_XYZ):
        t = Scalar_mask_XYZ(self.x, self.y, self.z, self.wavelength)
    elif isinstance(self, Scalar_field_XY):
        t = Scalar_field_XY(self.x, self.y, self.wavelength)

    if kind == 'source':
        if isinstance(other, tuple):
            t.u = self.u
            for o in other:
                t.u += o.u
        else:        
            t.u = self.u + other.u
    
    elif kind == 'mask':
        t1 = np.abs(self.u)
        f1 = np.angle(self.u)
        if isinstance(other, tuple):

            t.u = self.u
            for o in other:
                t2 = np.abs(o.u)
                f2 = np.angle(o.u)
                t.u += o.u
                i_change = t1+t2>1
                t.u[i_change]=(np.exp(1j*f1[i_change])+np.exp(1j*f2[i_change])).astype(np.complex128)
                t.u[i_change]= t.u[i_change]/np.abs( t.u[i_change])
        else:
            t2 = np.abs(other.u)
            f2 = np.angle(other.u)

            t.u = self.u + other.u
            i_change = t1+t2>1
            t.u[i_change]=np.exp(1j*f1[i_change])+np.exp(1j*f2[i_change])
            t.u[i_change]= t.u[i_change]/np.abs(t.u[i_change])
    
    elif kind == 'phases':
        t1 = np.abs(self.u)
        f1 = np.angle(self.u)
        
        if isinstance(other, tuple):
            t.u = self.u
            for o in other:
                t2 = np.abs(o.u)
                f2 = np.angle(o.u)
                t.u += o.u
                t.u = t1 * np.exp(1j * (f1 + f2))
        else:
            t2 = np.abs(other.u)
            f2 = np.angle(other.u)
        
            ts = t1 + t2
            ts[ts > 0] = 1.
            t.u = ts * np.exp(1j * (f1 + f2))

    elif kind == 'no_overlap':
        
        if isinstance(other, tuple):
            t.u = self.u
            for i, o in enumerate(other):
                i_pos1 = np.abs(t.u)>0
                i_pos2 = np.abs(o.u)>0
                print((i_pos1*i_pos2).sum())
                if (i_pos1 & i_pos2).any():
                    raise ValueError('The field {i} overlap with a previous one')
                t.u += o.u
        else:
            t1 = np.abs(self.u)
            t2 = np.abs(other.u)
            i_pos1 = t1>0
            i_pos2 = t2>0
            if (i_pos1 & i_pos2).any():
                raise ValueError('The two fields overlap')
    

    elif kind == 'refractive_index':
        if isinstance(other, tuple):
            t.n = self.n
            for o in other:
                t.n += o.n - o.n_background
        else:        
            t.n = self.n + other.n - other.n_background
        
        
    elif kind == 'distances':
        #todo: with simultaneous control of distances, not with for loop.
        if isinstance(other, tuple):
            pass
            
            """t.u = self.u

            for o in other:
                t.u = t.u + o.u
                com3 = center_of_mass(np.abs(t.u)>0)
                com4 = center_of_mass(np.abs(o.u)>0)
                
                t_max =  np.abs(t.u)>0
                o_max =  np.abs(o.u)>0
                overlap = t_max * o_max
                
                x3_center = t.x[int(com3[0])]
                x4_center = o.x[int(com4[0])]
                
                dist_t = np.abs(t.x-x3_center) * (np.abs(t.u)>0)
                dist_o = np.abs(o.x-x4_center) * (np.abs(o.u)>0)

                t.u = t.u+o.u
                i_menor = dist_t<dist_o
                i_mayor = dist_t>=dist_o

                t.u[i_menor*overlap] = t.u[i_menor*overlap]
                t.u[i_mayor*overlap] = o.u[i_mayor*overlap]"""
        else:  #only for 1D    
            print("not valid for 2D")
            t.u = self.u + other.u
            com3 = center_of_mass(np.abs(self.u)>0)
            com4 = center_of_mass(np.abs(other.u)>0)
            
            self_max =  np.abs(self.u)>0
            other_max =  np.abs(other.u)>0
            overlap = self_max * other_max
            
            x3_center = self.x[int(com3[0])]
            x4_center = other.x[int(com4[0])]
            
            dist_self = np.abs(self.x-x3_center) * (np.abs(self.u)>0)
            dist_other = np.abs(other.x-x4_center) * (np.abs(other.u)>0)

            t.u = self.u+other.u
            i_menor = dist_self<dist_other
            i_mayor = dist_self>=dist_other

            t.u[i_menor*overlap] = self.u[i_menor*overlap]
            t.u[i_mayor*overlap] = other.u[i_mayor*overlap]
        
    return t


def sub(self, other, kind: Options_sub  = 'source'):
    """substracts two fields. For example two light sources or two masks. The fields are added as complex numbers and then normalized so that the maximum amplitude is 1.
    
    Args:
        other (Other field): 2nd field to add
        kind (str): instruction how to add the fields: ['source', 'mask', 'phases', 'no_overlap', 'refractive_index'].
            - 'source': adds the fields as they are
            - 'mask': adds the fields as complex numbers and then normalizes so that the maximum amplitude is 1.
            - 'phases': adds the phases and then normalizes so that the maximum amplitude is 1.
            - 'np_overlap': adds the fields as they are. If the sum of the amplitudes is greater than 1, an error is produced

    Returns:
        Substraction of the two fields.
    """
    
    from diffractio.scalar_sources_X import Scalar_source_X
    from diffractio.scalar_sources_XY import Scalar_source_XY
    from diffractio.scalar_masks_X import Scalar_mask_X
    from diffractio.scalar_masks_XY import Scalar_mask_XY
    from diffractio.scalar_fields_XZ import Scalar_field_XZ
    from diffractio.scalar_masks_XZ import Scalar_mask_XZ
    from diffractio.scalar_fields_Z import Scalar_field_Z
    from diffractio.scalar_fields_XYZ import Scalar_field_XYZ
    from diffractio.scalar_masks_XYZ import Scalar_mask_XYZ
    

    if isinstance(self, Scalar_mask_XY):
        t = Scalar_mask_XY(self.x, self.y, self.wavelength)
    elif isinstance(self, Scalar_source_XY):
        t = Scalar_source_XY(self.x, self.y, self.wavelength)
    elif isinstance(self, Scalar_mask_X):
        t = Scalar_mask_X(self.x, self.wavelength)
    elif isinstance(self, Scalar_source_X):
        t = Scalar_source_X(self.x,  self.wavelength)
    elif isinstance(self, Scalar_field_XZ):
        t = Scalar_field_XZ(self.x, self.z, self.wavelength)
    elif isinstance(self, Scalar_field_Z):
        t = Scalar_field_Z(self.z, self.wavelength)
    elif isinstance(self, Scalar_mask_XY):
        t = Scalar_mask_XY(self.x, self.y, self.wavelength)
    elif isinstance(self, Scalar_mask_XZ):
        t = Scalar_mask_XZ(self.x, self.z, self.wavelength)
    elif isinstance(self, Scalar_field_XYZ):
        t = Scalar_field_XYZ(self.x, self.y, self.z, self.wavelength)
    elif isinstance(self, Scalar_mask_XYZ):
        t = Scalar_mask_XYZ(self.x, self.y, self.z, self.wavelength)  

    if kind == 'source':
        if isinstance(other, tuple):
            t.u = self.u
            for o in other:
                t.u -= o.u
        else:        
            t.u = self.u - other.u
    
    elif kind == 'mask':
        t1 = np.abs(self.u)
        f1 = np.angle(self.u)
        if isinstance(other, tuple):
            t.u = self.u
            for o in other:
                t2 = np.abs(o.u)
                f2 = np.angle(o.u)
                t.u = t.u - o.u
                i_change = t1-t2<0
                t.u[i_change]=(np.exp(1j*f1[i_change])-np.exp(1j*f2[i_change])).astype(np.complex128)
                t.u[i_change]= t.u[i_change]/np.abs( t.u[i_change])
        else:
            t2 = np.abs(other.n)
            f2 = np.angle(other.n)

            t.n = self.n - other.n
            i_change = t1-t2<0
            t.n[i_change]=np.exp(1j*f1[i_change])-np.exp(1j*f2[i_change])
            t.n[i_change]= t.n[i_change]/np.abs(t.n[i_change])
    
    elif kind == 'phases':
        t1 = np.abs(self.u)
        f1 = np.angle(self.u)
        
        if isinstance(other, tuple):
            t.u = self.u
            for o in other:
                t2 = np.abs(o.u)
                f2 = np.angle(o.u)
                t.u -= o.u
                t.u = t1 * np.exp(1j * (f1 - f2))
        else:
            t2 = np.abs(other.u)
            f2 = np.angle(other.u)
        
            ts = t1 - t2
            ts[ts < 0] = 0.
            t.u = ts * np.exp(1j * (f1 - f2))

    elif kind == 'no_overlap':
        
        if isinstance(other, tuple):
            t.u = self.u
            for i, o in enumerate(other):
                i_pos1 = np.abs(t.u)>0
                i_pos2 = np.abs(o.u)>0
                if (i_pos1 & i_pos2).any():
                    raise ValueError('The field {i} overlap with a previous one')
                t.u -= o.u
        else:
            t1 = np.abs(self.u)
            t2 = np.abs(other.u)
            i_pos1 = t1>0
            i_pos2 = t2>0
            if (i_pos1 & i_pos2).any():
                raise ValueError('The two fields overlap')

    elif kind == 'refractive_index':
        if isinstance(other, tuple):
            t.n = self.n
            for o in other:
                t.n -= (o.n - o.n_background)
        else:        
            t.n = self.n - (other.n - other.n_background)
            
    return t



def rmul(cls, number: float | complex | int, kind: Options_rmul  = 'intensity'):
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
    
    from diffractio.scalar_sources_X import Scalar_source_X
    from diffractio.scalar_sources_XY import Scalar_source_XY
    from diffractio.scalar_masks_X import Scalar_mask_X
    from diffractio.scalar_masks_XY import Scalar_mask_XY
    from diffractio.scalar_fields_XZ import Scalar_field_XZ
    from diffractio.scalar_fields_XY import Scalar_field_XY
    from diffractio.scalar_masks_XZ import Scalar_mask_XZ
    from diffractio.scalar_fields_Z import Scalar_field_Z
    from diffractio.scalar_fields_XYZ import Scalar_field_XYZ
    from diffractio.scalar_masks_XYZ import Scalar_mask_XYZ

    if isinstance(cls, Scalar_mask_XY):
        t = Scalar_mask_XY(cls.x, cls.y, cls.wavelength)
    elif isinstance(cls, Scalar_source_XY):
        t = Scalar_source_XY(cls.x, cls.y, cls.wavelength)
    elif isinstance(cls, Scalar_mask_X):
        t = Scalar_mask_X(cls.x, cls.wavelength)
    elif isinstance(cls, Scalar_source_X):
        t = Scalar_source_X(cls.x,  cls.wavelength)
    elif isinstance(cls, Scalar_mask_XZ):
        t = Scalar_mask_XZ(cls.x, cls.z, cls.wavelength)
    elif isinstance(cls, Scalar_field_XZ):
        t = Scalar_field_XZ(cls.x, cls.z, cls.wavelength)
    elif isinstance(cls, Scalar_field_Z):
        t = Scalar_field_Z(cls.z, cls.wavelength)
    elif isinstance(cls, Scalar_field_XYZ):
        t = Scalar_field_XYZ(cls.x, cls.y, cls.z, cls.wavelength)
    elif isinstance(cls, Scalar_mask_XYZ):
        t = Scalar_mask_XYZ(cls.x, cls.y, cls.z, cls.wavelength)
    elif isinstance(cls, Scalar_field_XY):
        t = Scalar_field_XY(cls.x, cls.y, cls.wavelength)

    if kind == 'intensity':
        t.u = cls.u * np.sqrt(number)
    
    elif kind == 'amplitude':
        t.u = cls.u * number

    elif kind == 'phase':
        ampltiude = np.abs(cls.u)
        phase = np.angle(cls.u)
        t.u = ampltiude * np.exp(1j * number * phase)
        
    return t


def computer_parameters(verbose: bool = False) -> tuple[int, float, float, float]:
    """Determine several computer Args:
        - number of processors
        - available memory
        - total memory
        - max frequency

    Args:
        verbose (bool): If True prints data

    Returns:
        num_max_processors (int): number of processors
        info_memory (int) : Gbytes
        memory_available (int): % available memory
        freq_max (int): Maximum frequency (GHz)
    """

    freq_max = psutil.cpu_freq()
    info_memory = psutil.virtual_memory()[0] / 1024**3
    memory_available = psutil.virtual_memory(
    ).available * 100 / psutil.virtual_memory().total

    num_max_processors = multiprocessing.cpu_count()

    if verbose:
        print("number of processors: {}".format(num_max_processors))
        print("total memory        : {:1.1f} Gb".format(info_memory))
        print("available memory    : {:1.0f} %".format(memory_available))
        print("max frequency       : {:1.0f} GHz".format(freq_max[2]))

    return num_max_processors, info_memory, memory_available, freq_max[2]


def clear_all():
    """clear all variables"""
    all = [var for var in globals() if var[0] != "_"]
    for var in all:
        del globals()[var]


def several_propagations(source, masks, distances: tuple[float]):
    '''performs RS propagation through several masks

    Args:
        source (Scalar_source_XY): illumination
        masks (tuple): list with several (Scalar_masks_XY)
        distances (tuple): list with seera distances


    Returns:
        Scalar_field_XY: u0 field at the last plane given by distances
        Scalar_field_XY: u1 field just at the plane of the last mask
    '''

    u0 = source

    for mask, distance in zip(masks, distances):
        u1 = u0 * mask
        u0 = u1.RS(z=distance)

    return u0, u1  # en el último plano y justo despues


def get_date():
    """gets current date and hour.

    Returns:
        (str): date in text
    """
    now = datetime.datetime.now()
    date = now.strftime("%Y-%m-%d_%H_%M_%S")
    return date


def save_data_common(cls,
                     filename: str,
                     add_name: str = '',
                     description: str = '',
                     verbose: bool = False) -> str:
    """Common save data function to be used in all the modules.
    The methods included are: npz, matlab

    Args:
        filename(str): filename
        add_name = (str): sufix to the name, if 'date' includes a date
        description(str): text to be stored in the dictionary to save.
        verbose(bool): If verbose prints filename.

    Returns:
        (str): filename. If False, file could not be saved.
    """

    now = datetime.datetime.now()
    date = now.strftime("%Y-%m-%d_%H_%M_%S")

    if add_name == 'date':
        add_name = "_" + date
    extension = filename.split('.')[-1]
    file = filename.split('.')[0]
    final_filename = file + add_name + '.' + extension

    if verbose:
        print(final_filename)

    cls.__dict__['date'] = date
    cls.__dict__['description'] = description

    if extension == 'npz':
        np.savez_compressed(file=final_filename, dict=cls.__dict__)

    elif extension == 'mat':
        savemat(final_filename, cls.__dict__)

    return final_filename


def load_data_common(cls, filename: str, verbose: bool = False):
    """Common load data function to be used in all the modules.
        The methods included are: npz, matlab

    Args:
        cls(class): class X, XY, XZ, XYZ, etc..
        filename(str): filename
        verbose(bool): If True prints data
    """

    def print_data_dict(dict0: dict):
        for k, v in dict0.items():
            print("{:12} = {}".format(k, v))
        print("\nnumber of data = {}".format(len(dict0['x'])))

    extension = filename.split('.')[-1]

    try:
        if extension in ('npy', 'npz'):
            npzfile = np.load(filename, allow_pickle=True)
            dict0 = npzfile['dict'].tolist()

        elif extension == 'mat':
            dict0 = loadmat(file_name=filename, mdict=cls.__dict__)

        else:
            print("extension not supported")

        if verbose is True:
            print(dict0.keys())

        if dict0 is not None:
            if isinstance(dict0, dict):
                cls.__dict__ = dict0
            else:
                raise Exception('no dictionary in load_data')

        return dict0

    except IOError:
        print('could not open {}'.format(filename))
        return None

    # with h5py.File('file.h5', 'r', libver='latest') as f:
    #     x_read = f['dict']['X'][:]  # [:] syntax extracts numpy array into memory
    #     y_read = f['dict']['Y'][:]


def print_axis_info(cls, axis: str):
    """Prints info about axis

    Args:
        cls(class): class of the modulus.
        axis(): axis x, y, z... etc.
    """

    x0 = eval("cls.{}[0]".format(axis))
    x1 = eval("cls.{}[-1]".format(axis))
    length = x1 - x0
    Dx = eval("cls.{}[1]-cls.{}[0]".format(axis, axis))
    axis_info = dict(axis=axis, min=x0, max=x1, length=length, Dx=Dx)
    print("   axis={axis}: min={min}, max={max}, length={length}, Dx={Dx}".
          format(**axis_info))


def date_in_name(filename: str) -> str:
    """introduces a date in the filename.

    Args:
        filename(str): filename

    Returns:
        (str): filename with current date
    """
    divided = filename.split(".")
    extension = divided[-1]
    rest = divided[0:-1]
    initial_name = ".".join(rest)
    now = datetime.datetime.now()
    date = now.strftime("%Y-%m-%d_%H_%M_%S_%f")
    filename_2 = "{}_{}.{}".format(initial_name, date, extension)
    return filename_2



def get_instance_size_MB(cls, verbose: bool = False) -> float:
    size = 0
    for key, value in cls.__dict__.items():
        size += sys.getsizeof(key)
        if isinstance(value, np.ndarray):
            size += value.nbytes
        else:
            size += sys.getsizeof(value)
        
    if verbose:
        if size < 1024:
            print("size = {} bytes".format(size))
        elif size < 1024**2:
            print("size = {:2.2f} Kbytes".format(size/1024))
        elif size < 1024**3:
            print("size = {:2.2f} Mbytes".format(size/1024**2))
        elif size < 1024**4:
            print("size = {:2.2f} Gbytes".format(size/1024**3))
    return size/1024**2