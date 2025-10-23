# !/usr/bin/env python3

# ----------------------------------------------------------------------
# Name:        config.py
# Purpose:     Configuration settings for diffractio package
#
# Author:      Luis Miguel Sanchez Brea
#
# Created:     2024
# Licence:     GPLv3
# ----------------------------------------------------------------------


"""
Configuration file. Standard diffractio units are um: um = 1.
"""

from typing import Literal
from matplotlib import cm
import numpy as np

bool_raise_exception = True

# Configuration params for drawings
CONF_DRAWING = dict()
CONF_DRAWING['color_intensity'] = cm.gist_heat  # cm.gist_heat  #cm.hot
CONF_DRAWING['color_amplitude'] = cm.jet
CONF_DRAWING['color_amplitude_sign'] = cm.seismic
CONF_DRAWING['color_phase'] = cm.twilight  #twilight .twilight_shifted hsv
CONF_DRAWING['color_real'] = cm.seismic
CONF_DRAWING['color_stokes'] = cm.seismic
CONF_DRAWING['color_n'] = cm.Blues
CONF_DRAWING['color_directions'] = cm.twilight_shifted
CONF_DRAWING['color_arrows'] = cm.hsv
CONF_DRAWING['percentage_intensity'] = 0.0  #percentage of intensity not shown in phase
CONF_DRAWING['percentage_Ez'] = 0.0  #percentage of intensity to show Ez

number_types = (int, float, complex, np.int32, np.float64)
empty_types = ([], '', None)

# Kind parameters for drawing
Draw_refractive_index_Options = Literal['all', 'real', 'imag']
Draw_X_Options = Literal[ 'amplitude', 'intensity', 'field', 'phase', 'fill', 'fft']
Draw_Z_Options = Literal['amplitude', 'intensity', 'field', 'phase']
Draw_XZ_Options = Literal['amplitude', 'intensity', 'phase', 'real']
Draw_XY_Options = Literal['amplitude', 'intensity', 'phase', 'field', 'real_field', 'contour']
Draw_XYZ_Options = Literal['intensity', 'refractive_index']
Draw_Vector_X_Options = Literal[ 'intensity', 'intensities', 'intensities_rz', 'phases', 'fields', 'stokes']
Draw_Vector_XY_Options = Literal['intensity', 'intensities', 'intensities_rz', 'phases', 'fields', 'stokes', ' param_ellipse', 'ellipses']
Draw_Vector_XZ_Options = Literal[ 'intensity', 'intensities', 'intensities_rz', 'phases', 'fields', 'E', 'EH', 'E2H2', 'poynting_vector', 'poynting_vector_averaged', 'poynting_total', 'energy_density', 'irradiance', 'stokes', 'param_ellipses', 'directions', 'arrows']
Draw_pyvista_Options = Literal['volume', 'clip', 'slices', 'projections']
Draw_interactive_Options = Literal['intensity', 'amplitude', 'phase']

# Options for pupil in Scalar_XY
options_pupil = Literal['direct', 'inverse']

Save_mask_Options = Literal['amplitude', 'phase', 'intensity']

Options_Diffractio_kind = Literal['scalar', 'vector']
Options_Diffractio_frame = Literal['field', 'mask', 'source']

Options_add = Literal['source', 'mask', 'refractive_index', 'phases', 'no_overlap', 'distances']
Options_sub = Literal['source', 'mask', 'phases', 'no_overlap', 'refractive_index']
Options_rmul = Literal['intensity', 'amplitude', 'phase']

Options_squares_nxm = Literal['amplitude', 'intensity', 'gray_levels']
video_isovalue_Options = Literal['intensity', 'refractive_index']

get_scalar_options = Literal['intensity', 'phase', 'field']
get_vector_options = Literal['E', 'H', 'EH', 'fields', 'intensity', 'intensities', 'phases', 'poynting_vector', 
                             'poynting_vector_averaged',  'poynting_total',  'energy_density', 'irradiance', 
                             'stokes', 'params_ellipse', 'directions']

