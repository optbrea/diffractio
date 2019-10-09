# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import datetime
import os
import sys

from diffractio import degrees, mm, nm, np, plt, sp, um
from diffractio.utils_tests import save_figure_test
from diffractio.vector_paraxial_masks_XY import Vector_paraxial_mask_XY
from diffractio.vector_paraxial_sources_XY import Vector_paraxial_source_XY

path_base = "tests_results"
path_class = "Vector_paraxial_draw_XY"
now = datetime.datetime.now()
date = now.strftime("%Y-%m-%d_%H_%M_%S")
date = '0'

newpath = "{}_{}/{}/".format(path_base, date, path_class)

if not os.path.exists(newpath):
    os.makedirs(newpath)

v_lineal = (1, 0)
v_circular = (1, 1j) / np.sqrt(2)

length = 500 * um
num_data = 256
wavelength = 0.6328 * um

x0 = np.linspace(-length / 2, length / 2, num_data)
y0 = np.linspace(-length / 2, length / 2, num_data)

EM = Vector_paraxial_source_XY(x0, y0, wavelength)
EM.radial_wave(A=1, x_center=(0 * um, 0 * um), radius=length / 5)
EM.RS(z=30 * mm, new_field=False)

Ex, Ey = EM.get()

EM.reduce_matrix = ''


class Test_Vector_paraxial_fields_XY(object):
    def test_draw_intensity(self):

        func_name = sys._getframe().f_code.co_name
        filename = '{}{}.npz'.format(newpath, func_name)

        EM.draw(kind='intensity')
        save_figure_test(newpath, func_name, add_name='')
        assert True

    def test_draw_intensities(self):

        func_name = sys._getframe().f_code.co_name
        filename = '{}{}.npz'.format(newpath, func_name)

        EM.draw(kind='intensities')
        save_figure_test(newpath, func_name, add_name='')
        assert True

    def test_draw_phases(self):

        func_name = sys._getframe().f_code.co_name
        filename = '{}{}.npz'.format(newpath, func_name)

        EM.draw(kind='phases')
        save_figure_test(newpath, func_name, add_name='')
        assert True

    def test_draw_fields(self):

        func_name = sys._getframe().f_code.co_name
        filename = '{}{}.npz'.format(newpath, func_name)

        EM.draw(kind='fields')
        save_figure_test(newpath, func_name, add_name='')
        assert True

    def test_draw_stokes(self):

        func_name = sys._getframe().f_code.co_name
        filename = '{}{}.npz'.format(newpath, func_name)

        EM.draw(kind='stokes')
        save_figure_test(newpath, func_name, add_name='')
        assert True

    def test_draw_param_ellipse(self):

        func_name = sys._getframe().f_code.co_name
        filename = '{}{}.npz'.format(newpath, func_name)

        EM.draw(kind='param_ellipse')
        save_figure_test(newpath, func_name, add_name='')
        assert True