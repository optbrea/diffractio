# !/usr/bin/env python3

import datetime
import os
import sys  

from diffractio import mm, no_date, np, um
from diffractio.utils_tests import save_figure_test
from diffractio.vector_sources_XY import Vector_source_XY

path_base = "tests_results"
path_class = "Vector_draw_XY"

if no_date is True:
    date = '0'
else:
    now = datetime.datetime.now()
    date = now.strftime("%Y-%m-%d_%H")

newpath = "{}_{}/{}/".format(path_base, date, path_class)

if not os.path.exists(newpath):
    os.makedirs(newpath)

v_lineal = (1, 0)
v_circular = (1, 1j) / np.sqrt(2)

length = 500*um
num_data = 256
wavelength = 0.6328*um

x0 = np.linspace(-length/2, length/2, num_data)
y0 = np.linspace(-length/2, length/2, num_data)

EM = Vector_source_XY(x0, y0, wavelength)
EM.azimuthal_wave(u=1, r0=(0*um, 0*um), radius=(length / 5, length / 5))
EM.VRS(z=30*mm, new_field=False)

Ex, Ey, _ = EM.get(kind='E')

EM.reduce_matrix = ''


class Test_Vector_fields_XY():

    def test_draw_intensity(self):

        func_name = sys._getframe().f_code.co_name
        # filename = '{}{}.npz'.format(newpath, func_name)

        EM.draw(kind='intensity')
        save_figure_test(newpath, func_name, add_name='')
        assert True

    def test_draw_intensities(self):

        func_name = sys._getframe().f_code.co_name
        # filename = '{}{}.npz'.format(newpath, func_name)

        EM.draw(kind='intensities')
        save_figure_test(newpath, func_name, add_name='')
        assert True

    def test_draw_phases(self):

        func_name = sys._getframe().f_code.co_name
        # filename = '{}{}.npz'.format(newpath, func_name)

        EM.draw(kind='phases')
        save_figure_test(newpath, func_name, add_name='')
        assert True

    def test_draw_fields(self):

        func_name = sys._getframe().f_code.co_name
        # filename = '{}{}.npz'.format(newpath, func_name)

        EM.draw(kind='fields')
        save_figure_test(newpath, func_name, add_name='')
        assert True

    def test_draw_stokes(self):

        func_name = sys._getframe().f_code.co_name
        # filename = '{}{}.npz'.format(newpath, func_name)

        EM.draw(kind='stokes')
        save_figure_test(newpath, func_name, add_name='')
        assert True

    def test_draw_param_ellipse(self):

        func_name = sys._getframe().f_code.co_name
        # filename = '{}{}.npz'.format(newpath, func_name)

        EM.draw(kind='param_ellipses')
        save_figure_test(newpath, func_name, add_name='')
        assert True
