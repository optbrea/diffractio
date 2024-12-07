# !/usr/bin/env python3

"""
Test for scalar_sources_XYZ"""

import datetime
import os
import sys

from diffractio import degrees, no_date, np, um
from diffractio.scalar_masks_XY import Scalar_mask_XY
from diffractio.scalar_masks_XYZ import Scalar_mask_XYZ
from diffractio.utils_tests import comparison, save_figure_test

if no_date is True:
    date = '0'
else:
    now = datetime.datetime.now()
    date = now.strftime("%Y-%m-%d_%H")

path_base = "tests_results"
path_class = "scalar_masks_XYZ"

newpath = "{}_{}/{}/".format(path_base, date, path_class)

if not os.path.exists(newpath):
    os.makedirs(newpath)


class Test_Scalar_masks_XYZ():

    def test_sphere(self):
        func_name = sys._getframe().f_code.co_name
        # filename = '{}{}'.format(newpath, func_name)

        length = 100*um
        numdataX = 64
        numdataZ = 64

        longitud = 50*um

        x0 = np.linspace(-length/2, length/2, numdataX)
        y0 = np.linspace(-length/2, length/2, numdataX)
        z0 = np.linspace(-longitud, longitud, numdataZ)
        wavelength = 0.55*um

        t1 = Scalar_mask_XY(x=x0, y=y0, wavelength=wavelength)
        t1.circle(r0=(0*um, 0*um),
                  radius=(20*um, 20*um),
                  angle=0*degrees)

        uxyz = Scalar_mask_XYZ(x=x0,
                               y=y0,
                               z=z0,
                               wavelength=wavelength,
                               n_background=1.,
                               info='')
        uxyz.sphere(r0=(0*um, 0*um, 0*um),
                    radius=(10*um, 30*um, 50*um),
                    refractive_index=2,
                    rotation = None)
        uxyz.incident_field(t1)
        assert True
