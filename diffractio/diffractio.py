# !/usr/bin/env python3

# ----------------------------------------------------------------------
# Name:        Diffractio.py
# Purpose:     Class for controlling the other modules.
#
# Author:      Luis Miguel Sanchez Brea
#
# Created:     2024
# Licence:     GPLv3
# ----------------------------------------------------------------------


"""
Init module to control the rest of the modules. 
    

"""
# flake8: noqa


from typing import Literal, overload

from .config import Options_Diffractio_kind, Options_Diffractio_frame

from diffractio.scalar_fields_X import Scalar_field_X
from diffractio.scalar_fields_XY import Scalar_field_XY
from diffractio.scalar_fields_XYZ import Scalar_field_XYZ
from diffractio.scalar_fields_XZ import Scalar_field_XZ
from diffractio.scalar_fields_Z import Scalar_field_Z

from diffractio.scalar_masks_X import Scalar_mask_X
from diffractio.scalar_masks_XY import Scalar_mask_XY
from diffractio.scalar_masks_XYZ import Scalar_mask_XYZ
from diffractio.scalar_masks_XZ import Scalar_mask_XZ

from diffractio.scalar_sources_X import Scalar_source_X
from diffractio.scalar_sources_XY import Scalar_source_XY

from diffractio.vector_fields_X import Vector_field_X
from diffractio.vector_fields_XY import Vector_field_XY
from diffractio.vector_fields_XYZ import Vector_field_XYZ
from diffractio.vector_fields_XZ import Vector_field_XZ
from diffractio.vector_fields_Z import Vector_field_Z

from diffractio.vector_masks_XY import Vector_mask_XY
from diffractio.vector_sources_XY import Vector_source_XY

from diffractio.utils_typing import NDArrayFloat

class Diffractio():
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

    @overload
    def __new__(
        cls, kind: Literal['scalar'], frame: Literal['source'],
        x: NDArrayFloat, y: None = ..., z: None = ...,
        wavelength: float = ..., n_background: float = ..., info: str = ...,
    ) -> Scalar_source_X: ...

    @overload
    def __new__(
        cls, kind: Literal['scalar'], frame: Literal['mask'],
        x: NDArrayFloat, y: None = ..., z: None = ...,
        wavelength: float = ..., n_background: float = ..., info: str = ...,
    ) -> Scalar_mask_X: ...

    @overload
    def __new__(
        cls, kind: Literal['scalar'], frame: Literal['field'],
        x: NDArrayFloat, y: None = ..., z: None = ...,
        wavelength: float = ..., n_background: float = ..., info: str = ...,
    ) -> Scalar_field_X: ...

    @overload
    def __new__(
        cls, kind: Literal['scalar'], frame: Literal['source'],
        x: NDArrayFloat, y: NDArrayFloat, z: None = ...,
        wavelength: float = ..., n_background: float = ..., info: str = ...,
    ) -> Scalar_source_XY: ...

    @overload
    def __new__(
        cls, kind: Literal['scalar'], frame: Literal['mask'],
        x: NDArrayFloat, y: NDArrayFloat, z: None = ...,
        wavelength: float = ..., n_background: float = ..., info: str = ...,
    ) -> Scalar_mask_XY: ...

    @overload
    def __new__(
        cls, kind: Literal['scalar'], frame: Literal['field'],
        x: NDArrayFloat, y: NDArrayFloat, z: None = ...,
        wavelength: float = ..., n_background: float = ..., info: str = ...,
    ) -> Scalar_field_XY: ...

    @overload
    def __new__(
        cls, kind: Literal['scalar'], frame: Literal['mask'],
        x: NDArrayFloat, y: None = ..., z: NDArrayFloat = ...,
        wavelength: float = ..., n_background: float = ..., info: str = ...,
    ) -> Scalar_mask_XZ: ...

    @overload
    def __new__(
        cls, kind: Literal['scalar'], frame: Literal['field'],
        x: NDArrayFloat, y: None = ..., z: NDArrayFloat = ...,
        wavelength: float = ..., n_background: float = ..., info: str = ...,
    ) -> Scalar_field_XZ: ...

    @overload
    def __new__(
        cls, kind: Literal['scalar'], frame: Literal['mask'],
        x: NDArrayFloat, y: NDArrayFloat, z: NDArrayFloat,
        wavelength: float = ..., n_background: float = ..., info: str = ...,
    ) -> Scalar_mask_XYZ: ...

    @overload
    def __new__(
        cls, kind: Literal['scalar'], frame: Literal['field'],
        x: NDArrayFloat, y: NDArrayFloat, z: NDArrayFloat,
        wavelength: float = ..., n_background: float = ..., info: str = ...,
    ) -> Scalar_field_XYZ: ...

    @overload
    def __new__(
        cls, kind: Literal['scalar'], frame: Literal['field'],
        x: None = ..., y: None = ..., z: NDArrayFloat = ...,
        wavelength: float = ..., n_background: float = ..., info: str = ...,
    ) -> Scalar_field_Z: ...

    @overload
    def __new__(
        cls, kind: Literal['vector'], frame: Literal['field'],
        x: NDArrayFloat, y: None = ..., z: None = ...,
        wavelength: float = ..., n_background: float = ..., info: str = ...,
    ) -> Vector_field_X: ...

    @overload
    def __new__(
        cls, kind: Literal['vector'], frame: Literal['mask'],
        x: NDArrayFloat, y: NDArrayFloat, z: None = ...,
        wavelength: float = ..., n_background: float = ..., info: str = ...,
    ) -> Vector_mask_XY: ...

    @overload
    def __new__(
        cls, kind: Literal['vector'], frame: Literal['source'],
        x: NDArrayFloat, y: NDArrayFloat, z: None = ...,
        wavelength: float = ..., n_background: float = ..., info: str = ...,
    ) -> Vector_source_XY: ...

    @overload
    def __new__(
        cls, kind: Literal['vector'], frame: Literal['field'],
        x: NDArrayFloat, y: NDArrayFloat, z: None = ...,
        wavelength: float = ..., n_background: float = ..., info: str = ...,
    ) -> Vector_field_XY: ...

    @overload
    def __new__(
        cls, kind: Literal['vector'], frame: Literal['field'],
        x: NDArrayFloat, y: None = ..., z: NDArrayFloat = ...,
        wavelength: float = ..., n_background: float = ..., info: str = ...,
    ) -> Vector_field_XZ: ...

    @overload
    def __new__(
        cls, kind: Literal['vector'], frame: Literal['field'],
        x: NDArrayFloat, y: NDArrayFloat, z: NDArrayFloat,
        wavelength: float = ..., n_background: float = ..., info: str = ...,
    ) -> Vector_field_XYZ: ...

    @overload
    def __new__(
        cls, kind: Literal['vector'], frame: Literal['field'],
        x: None = ..., y: None = ..., z: NDArrayFloat = ...,
        wavelength: float = ..., n_background: float = ..., info: str = ...,
    ) -> Vector_field_Z: ...

    @overload
    def __new__(
        cls, kind: Options_Diffractio_kind, frame: Options_Diffractio_frame,
        x: NDArrayFloat | None = ..., y: NDArrayFloat | None = ...,
        z: NDArrayFloat | None = ..., wavelength: float = ...,
        n_background: float = ..., info: str = ...,
    ) -> "Diffractio": ...

    def __new__(cls, *_args: object, **_kwargs: object) -> object:
        del _args, _kwargs
        return super().__new__(cls)

    def __init__(self, 
                 kind: Options_Diffractio_kind,
                 frame: Options_Diffractio_frame,
                 x: NDArrayFloat | None = None, 
                 y: NDArrayFloat | None = None, 
                 z: NDArrayFloat | None = None, 
                 wavelength: float  = 0,
                 n_background: float = 1., 
                 info: str = ""):
        
        if kind == 'scalar':
            if x is not None and y is None and z is None:
                if frame == 'source':
                    self.__class__ =  Scalar_source_X
                    self.__init__(x, wavelength, n_background, info)
                elif frame == 'mask':
                    self.__class__ =  Scalar_mask_X
                    self.__init__(x, wavelength, n_background, info)
                elif frame == 'field':
                    self.__class__ = Scalar_field_X
                    self.__init__(x, wavelength, n_background, info)
            elif x is not None and y is not None and z is None:
                if frame == 'source':
                    self.__class__ = Scalar_source_XY
                    self.__init__(x, y, wavelength, n_background, info)
                elif frame == 'mask':
                    self.__class__ = Scalar_mask_XY
                    self.__init__(x, y, wavelength, n_background, info)
                elif frame == 'field':
                    self.__class__ = Scalar_field_XY
                    self.__init__(x, y, wavelength, n_background, info)
            elif x is not None and y is None and z is not None:
                if frame == 'mask':
                    self.__class__ = Scalar_mask_XZ
                    self.__init__(x, z, wavelength, n_background, info)
                elif frame == 'field':
                    self.__class__ = Scalar_field_XZ
                    self.__init__(x, z, wavelength, n_background, info)
            elif x is not None and y is not None and z is not None:
                if frame == 'mask':
                    self.__class__ = Scalar_mask_XYZ
                    self.__init__(x, y, z, wavelength, n_background, info)
                elif frame == 'field':
                    self.__class__ = Scalar_field_XYZ
                    self.__init__(x, y, z, wavelength, n_background, info)
            elif x is None and y is None and z is not None:
                if frame == 'field':
                    self.__class__ = Scalar_field_Z
                    self.__init__(z, wavelength, n_background, info)
            else:
                raise ValueError('frame must be source, mask or field')
        elif kind == 'vector':
            if x is not None and y is None and z is None:
                if frame == 'field':
                    self.__class__ =  Vector_field_X    
                    self.__init__(x, wavelength, n_background, info)  
            elif x is not None and y is not None and z is None:
                if frame == 'field':
                    self.__class__ = Vector_field_XY  
                    self.__init__(x, y, wavelength, n_background, info)
                elif frame == 'mask':
                    self.__class__ = Vector_mask_XY
                    self.__init__(x, y, wavelength, n_background, info)  
                elif frame == 'source':
                    self.__class__ = Vector_source_XY
                    self.__init__(x, y, wavelength, n_background, info) 
            elif x is not None and y is None and z is not None:
                if frame == 'field':
                    self.__class__ = Vector_field_XZ
                    self.__init__(x, z, wavelength, n_background, info)
            elif x is not None and y is not None and z is not None:
                if frame == 'field':
                    self.__class__ = Vector_field_XYZ
                    self.__init__(x, y, z, wavelength, n_background, info)
            elif x is None and y is None and z is not None:
                if frame == 'field':
                    self.__class__ = Vector_field_Z
                    self.__init__(z, wavelength, n_background, info)
            else:
                raise ValueError('frame must be fields, source or mask')
        else:
            raise ValueError('kind must be scalar or vector')