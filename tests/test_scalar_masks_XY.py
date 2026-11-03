# !/usr/bin/env python3

"""Tests form Scalar_masks_XY"""

import datetime
import os
import sys

from diffractio import degrees, mm, no_date, np, plt, um
from diffractio.scalar_masks_XY import Scalar_mask_XY
from diffractio.utils_tests import comparison, save_figure_test

if no_date is True:
    date = '0'
else:
    now = datetime.datetime.now()
    date = now.strftime("%Y-%m-%d_%H")

path_base = "tests_results"
path_class = "scalar_masks_XY"

newpath = "{}_{}/{}/".format(path_base, date, path_class)

if not os.path.exists(newpath):
    os.makedirs(newpath)


class Test_Scalar_masks_XY():

    def test_add(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)

        num_data = 512
        length = 250*um
        x = np.linspace(-length/2, length/2, num_data)
        y = np.linspace(-length/2, length/2, num_data)
        wavelength = 0.6328*um

        t1 = Scalar_mask_XY(x, y, wavelength)
        t1.square(r0=(-50*um, 0*um),
                  size=(50*um, 50*um),
                  angle=0*degrees)
        t1.draw(title='test_square')

        t2 = Scalar_mask_XY(x, y, wavelength)
        t2.circle(r0=(50*um, 0*um),
                  radius=(25*um, 25*um),
                  angle=0*degrees)
        t2.draw(title='test_square')

        t3 = t2 + t1
        t3.draw(title='suma')

        t1.save_data(filename=filename + '.npz', add_name='_1')
        save_figure_test(newpath, func_name, add_name='_1')

        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)

        num_data = 512
        length = 250*um
        x = np.linspace(-length/2, length/2, num_data)
        y = np.linspace(-length/2, length/2, num_data)
        wavelength = 0.6328*um

        t1 = Scalar_mask_XY(x, y, wavelength)
        t1.square(r0=(-50*um, 0*um),
                  size=(150*um, 150*um),
                  angle=0*degrees)
        t1.draw(title='test_square')

        t2 = Scalar_mask_XY(x, y, wavelength)
        t2.circle(r0=(50*um, 0*um),
                  radius=(125*um, 125*um),
                  angle=0*degrees)
        t2.draw(title='test_square')

        t3 = t2 + t1
        t3.draw(title='add')

        t3.save_data(filename=filename + '.npz', add_name='_2')
        save_figure_test(newpath, func_name, add_name='_2')
        assert True

    def test_substract(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)

        num_data = 512
        length = 250*um
        x = np.linspace(-length/2, length/2, num_data)
        y = np.linspace(-length/2, length/2, num_data)
        wavelength = 0.6328*um

        t1 = Scalar_mask_XY(x, y, wavelength)
        t1.square(r0=(0*um, 0*um),
                  size=(100*um, 100*um),
                  angle=0*degrees)
        t1.draw(title='test_square')

        t2 = Scalar_mask_XY(x, y, wavelength)
        t2.circle(r0=(0*um, 0*um),
                  radius=(45*um, 45*um),
                  angle=0*degrees)
        t2.draw(title='test_circle')

        t3 = t2 - t1
        t3.draw(title='resta')

        t3.save_data(filename=filename + '.npz')
        save_figure_test(newpath, func_name)
        assert True

    def test_binarize(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)
        """
        tomamos una mascara de niveles(0,1) y la pasamos a ampitud-fase
        """

        num_data = 512
        length = 25*um
        x = np.linspace(-length/2, length/2, num_data)
        y = np.linspace(-length/2, length/2, num_data)
        wavelength = 0.6328*um

        t1 = Scalar_mask_XY(x, y, wavelength)
        t1.double_slit(x0=0,
                       size=5*um,
                       separation=15*um,
                       angle=0*degrees)
        t1.draw(kind='field')
        plt.title('(0,1)-sin binarizar')
        t1.save_data(filename=filename + '.npz', add_name='_wo_bin')
        save_figure_test(newpath, func_name, '_wo_bin')
        t1.binarize(kind='amplitude',
                    bin_level=None,
                    level0=0.25,
                    level1=.75,
                    new_field=False,
                    matrix=False)
        t1.draw(kind='field')
        plt.suptitle('binarizada')

        t1.save_data(filename=filename + '.npz', add_name='_wi_bin')
        save_figure_test(newpath, func_name, add_name='_wi_bin')
        assert True

    def test_slit(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)

        num_data = 512
        length = 250*um
        x = np.linspace(-length/2, length/2, num_data)
        y = np.linspace(-length/2, length/2, num_data)
        wavelength = 0.6328*um

        t1 = Scalar_mask_XY(x, y, wavelength)
        t1.slit(x0=0, size=100*um, angle=0*degrees)
        t1.draw(kind='intensity')

        t1.save_data(filename=filename + '.npz')
        save_figure_test(newpath, func_name)
        assert True

    def test_sinusoidal_slit(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)

        x = np.linspace(-500*um, 500*um, 512)
        y = np.linspace(-500*um, 500*um, 512)
        wavelength = 1*um

        t1 = Scalar_mask_XY(x, y, wavelength)
        t1.sinusoidal_slit(x0=0*um,
                           size=90,
                           amplitude=30*um,
                           phase=0*degrees,
                           angle=0*degrees,
                           period=100*um)
        t1.draw(kind='intensity')

        t1.save_data(filename=filename + '.npz')
        save_figure_test(newpath, func_name)
        assert True

    def test_slit_series(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)

        num_data = 256
        length = 1000*um
        x0 = np.linspace(-length/2, length/2, num_data)
        y0 = np.linspace(-length/2, length/2, num_data)
        wavelength = 0.6328*um

        a_coef1 = np.array([[0, 1, 2], [0, 25, 25]])
        a_coef2 = np.array([[0, 1, 2], [0, 25, 25]])

        t1 = Scalar_mask_XY(x=x0, y=y0, wavelength=wavelength)
        t1.slit_series(x0=0,
                       width=200,
                       period1=100,
                       period2=100,
                       Dy=(0, 0),
                       a_coef1=a_coef1,
                       a_coef2=a_coef2,
                       angle=0*degrees)
        t1.draw()

        t1.save_data(filename=filename + '.npz')
        save_figure_test(newpath, func_name)
        assert True

    def test_hiperellipse(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)

        x = np.linspace(-100*um, 100*um, 512)
        y = np.linspace(-100*um, 100*um, 512)
        wavelength = 1*um

        t1 = Scalar_mask_XY(x, y, wavelength)

        t1.super_ellipse(r0=(0*um, 0*um),
                         radius=(20*um, 40*um),
                         angle=0*degrees,
                         n=[0.5, 4])
        t1.draw(kind='intensity')

        t1.save_data(filename=filename + '.npz')
        save_figure_test(newpath, func_name)
        assert True

    def test_crossed_slits(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)

        x = np.linspace(-100*um, 100*um, 512)
        y = np.linspace(-100*um, 100*um, 512)
        wavelength = 1*um

        t1 = Scalar_mask_XY(x, y, wavelength)

        t1.super_ellipse(r0=(0*um, 0*um),
                         radius=(20*um, 40*um),
                         angle=0*degrees,
                         n=[0.5, 4])
        t1.draw(kind='intensity')

        t1.save_data(filename=filename + '.npz')
        save_figure_test(newpath, func_name)
        assert True

    def test_double_slit(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)

        num_data = 512
        length = 25*um
        x = np.linspace(-length/2, length/2, num_data)
        y = np.linspace(-length/2, length/2, num_data)
        wavelength = 0.6328*um

        t1 = Scalar_mask_XY(x, y, wavelength)
        t1.double_slit(x0=0,
                       size=5*um,
                       separation=15*um,
                       angle=0*degrees)
        t1.draw(kind='intensity')

        t1.save_data(filename=filename + '.npz')
        save_figure_test(newpath, func_name)
        assert True

    def test_triangle(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)

        x = np.linspace(-100*um, 100*um, 512)
        y = np.linspace(-100*um, 100*um, 512)
        wavelength = 1*um
        t1 = Scalar_mask_XY(x, y, wavelength)

        t1.triangle(r0=(0*um, 0*um), slope=1, height=50*um, angle=0*degrees)
        t1.draw(kind='intensity')

        t1.save_data(filename=filename + '.npz')
        save_figure_test(newpath, func_name)
        assert True

    def test_insert_array_masks(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)

        x = np.linspace(-250*um, 250*um, 512)
        y = np.linspace(-250*um, 250*um, 512)
        wavelength = 1*um

        s = Scalar_mask_XY(x, y, wavelength)
        s.cross(r0=(0*um, 0*um),
                size=(90*um, 25*um),
                angle=0*degrees)

        t1 = Scalar_mask_XY(x, y, wavelength)
        t1.insert_array_masks(t1=s,
                              space=[100*um, 100*um],
                              margin=50*um,
                              angle=90*degrees)
        t1.draw(kind='intensity')

        t1.save_data(filename=filename + '.npz')
        save_figure_test(newpath, func_name)
        assert True

    def test_square(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)

        num_data = 512
        length = 250*um
        x = np.linspace(-length/2, length/2, num_data)
        y = np.linspace(-length/2, length/2, num_data)
        wavelength = 0.6328*um

        t1 = Scalar_mask_XY(x, y, wavelength)
        t1.square(r0=(0*um, 0*um),
                  size=(100*um, 50*um),
                  angle=45*degrees)
        t1.draw(kind='intensity')

        t1.save_data(filename=filename + '.npz')
        save_figure_test(newpath, func_name)
        assert True

    def test_circle(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)

        num_data = 512
        length = 250*um
        x = np.linspace(-length/2, length/2, num_data)
        y = np.linspace(-length/2, length/2, num_data)
        wavelength = 0.6328*um

        t1 = Scalar_mask_XY(x, y, wavelength)
        t1.circle(r0=(0*um, 0*um), radius=(100*um, 100*um))
        t1.draw(kind='intensity')

        t1.save_data(filename=filename + '.npz')
        save_figure_test(newpath, func_name)
        assert True

    def test_super_gauss(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)

        num_data = 512
        length = 250*um
        x = np.linspace(-length/2, length/2, num_data)
        y = np.linspace(-length/2, length/2, num_data)
        wavelength = 0.6328*um

        t1 = Scalar_mask_XY(x, y, wavelength)
        t1.super_gauss_circular(r0=(0*um, 0*um),
                       radius=(length / 3, length / 3),
                       angle=45*degrees,
                       power=22)
        t1.draw(kind='intensity')

        t1.save_data(filename=filename + '.npz')
        save_figure_test(newpath, func_name)
        assert True

    def test_square_circle(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)

        num_data = 512
        length = 250*um
        x = np.linspace(-length/2, length/2, num_data)
        y = np.linspace(-length/2, length/2, num_data)
        wavelength = 0.6328*um

        t1 = Scalar_mask_XY(x, y, wavelength)
        t1.square_circle(s=.75,
                         r0=(0*um, 0*um),
                         R1=100*um,
                         R2=100*um,
                         angle=0*degrees)
        t1.draw(kind='intensity')
        # t2 = t1.fft(remove0=False)
        # t2.draw(logarithm=True)

        t1.save_data(filename=filename + '.npz')
        save_figure_test(newpath, func_name)
        assert True

    def test_angular_aperture(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)

        num_pixels = 512
        length = 200*um
        x0 = np.linspace(-length/2, length/2, num_pixels)
        y0 = np.linspace(-length/2, length/2, num_pixels)
        wavelength = 0.6238*um

        t1 = Scalar_mask_XY(x=x0, y=y0, wavelength=wavelength)
        a_coef = np.array([[0, 4, 8], [50, 25, 25]])
        t1.angular_aperture(a_coef=a_coef, angle=0*degrees)
        t1.draw()

        t1.save_data(filename=filename + '.npz')
        save_figure_test(newpath, func_name)
        assert True

    def test_ellipse(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)

        num_data = 512
        length = 250*um
        x = np.linspace(-length/2, length/2, num_data)
        y = np.linspace(-length/2, length/2, num_data)
        wavelength = 0.6328*um

        t1 = Scalar_mask_XY(x, y, wavelength)
        t1.circle(r0=(0*um, 0*um),
                  radius=(50*um, 25*um),
                  angle=45*degrees)
        t1.draw(kind='intensity')

        t1.save_data(filename=filename + '.npz')
        save_figure_test(newpath, func_name)
        assert True

    def test_ring(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)

        num_data = 512
        length = 250*um
        x = np.linspace(-length/2, length/2, num_data)
        y = np.linspace(-length/2, length/2, num_data)
        wavelength = 0.6328*um

        t1 = Scalar_mask_XY(x, y, wavelength)
        t1.ring(r0=(0*um, 0*um),
                radius1=(50*um, 50*um),
                radius2=(100*um, 150*um),
                angle=45*degrees)
        t1.draw(kind='intensity')

        t1.save_data(filename=filename + '.npz')
        save_figure_test(newpath, func_name)
        assert True

    def test_rings(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)

        length = 60*um
        num_data = 512
        x = np.linspace(-length/2, length/2, num_data)
        y = np.linspace(-length/2, length/2, num_data)
        wavelength = 0.6328*um

        inner_radius = np.array([5., 10, 15, 20, 25])

        outer_radius = np.array([6., 12, 17, 23, 29])

        t1 = Scalar_mask_XY(x, y, wavelength)
        t1.rings(r0=(0*um, 0*um),
                 inner_radius=inner_radius,
                 outer_radius=outer_radius)
        t1.draw(kind='intensity')

        t1.save_data(filename=filename + '.npz')
        save_figure_test(newpath, func_name)
        assert True
        num_data = 512

    def test_cross(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)

        num_data = 512
        length = 250*um
        x = np.linspace(-length/2, length/2, num_data)
        y = np.linspace(-length/2, length/2, num_data)
        wavelength = 0.6328*um

        t1 = Scalar_mask_XY(x, y, wavelength)
        t1.cross(r0=(0*um, 0*um),
                 size=(200*um, 75*um),
                 angle=0*degrees)
        t1.draw(kind='intensity')

        t1.save_data(filename=filename + '.npz')
        save_figure_test(newpath, func_name)
        assert True

    def test_two_levels(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)

        num_data = 512
        length = 250*um
        x = np.linspace(-length/2, length/2, num_data)
        y = np.linspace(-length/2, length/2, num_data)
        wavelength = 0.6328*um

        t1 = Scalar_mask_XY(x, y, wavelength)
        t1.two_levels(level1=0, level2=.5, x_edge=0)
        t1.draw(kind='intensity')

        t1.save_data(filename=filename + '.npz')
        save_figure_test(newpath, func_name)
        assert True

    def test_edge_series(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)

        num_pixels = 512
        length = 1000*um
        x0 = np.linspace(-length/2, length/2, num_pixels)
        y0 = np.linspace(-length/2, length/2, num_pixels)
        wavelength = 0.06238*um

        a_coef = np.array([[0, 1, 2], [0, 100, 25]])
        b_coef = np.array([[0, 3], [0, 25]])

        t1 = Scalar_mask_XY(x=x0, y=y0, wavelength=wavelength)
        t1.edge_series(r0=(0*um, 0*um),
                       period=100,
                       a_coef=a_coef,
                       b_coef=b_coef,
                       angle=0*degrees,
                       invert=False)
        t1.draw()

        t1.save_data(filename=filename + '.npz')
        save_figure_test(newpath, func_name)
        assert True

    def test_gray_scale(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)

        num_data = 512
        length = 250*um
        x = np.linspace(-length/2, length/2, num_data)
        y = np.linspace(-length/2, length/2, num_data)
        wavelength = 0.6328*um

        t1 = Scalar_mask_XY(x, y, wavelength)
        t1.gray_scale(num_levels=128, level_min=0, level_max=1)
        t1.draw(kind='intensity')

        t1.save_data(filename=filename + '.npz')
        save_figure_test(newpath, func_name)
        assert True

    def test_mask_phase_1(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)

        num_data = 512
        length = 250*um
        x = np.linspace(-length/2, length/2, num_data)
        y = np.linspace(-length/2, length/2, num_data)
        wavelength = 0.6328*um

        t1 = Scalar_mask_XY(x, y, wavelength)
        t1.mask_from_function(r0=(0*um, 0*um),
                              index=1.5,
                              f1='1*degrees*self.Y',
                              f2='1*degrees*self.X',
                              v_globals=None,
                              radius=(100*um, 100*um))
        t1.draw(kind='field')

        t1.save_data(filename=filename + '.npz')
        save_figure_test(newpath, func_name)
        assert True

    def test_mask_phase_2(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)

        # lens con una surface plana y otra esferica
        num_data = 512
        length = 250*um
        x = np.linspace(-length/2, length/2, num_data)
        y = np.linspace(-length/2, length/2, num_data)
        wavelength = 0.6328*um

        f1 = "R1-h1+np.sqrt(R1**2-(self.X-x0)**2-(self.Y-y0)**2)"
        f1 = "1*degrees*self.X"
        f1 = "np.zeros_like(self.X,dtype=float)"
        f2 = "R2-h2+np.sqrt(R2**2-(self.X-x0)**2-(self.Y-y0)**2)"
        # f2="R2-h2+(R2**4-(self.X-x0)**4-(self.Y-y0)**4)**0.25"
        v_globals = {
            'R1': 5*mm,
            'R2': 1*mm,
            'x0': 0*um,
            'y0': 0*um,
            'h1': 1*mm,
            'h2': -1*mm,
            'np': np,
        }
        index = 1.5
        print(v_globals)

        t1 = Scalar_mask_XY(x, y, wavelength)
        t1.mask_from_function(r0=(0*um, 0*um),
                              index=index,
                              f1=f1,
                              f2=f2,
                              v_globals=v_globals,
                              radius=(100*um, 100*um))
        t1.draw(kind='field')

        focal = v_globals['R2'] / (index - 1)
        t2 = t1.RS(z=focal)
        t2.draw(kind='field', logarithm=True)

        t1.save_data(filename=filename + '.npz')
        save_figure_test(newpath, func_name)
        assert True

    def test_lens(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)

        num_data = 512
        length = 250*um
        x = np.linspace(-length/2, length/2, num_data)
        y = np.linspace(-length/2, length/2, num_data)
        wavelength = 0.6328*um

        t0 = Scalar_mask_XY(x, y, wavelength)
        t0.lens(r0=(0*um, 0*um),
                radius=(100*um, 100*um),
                focal=(2.5*mm, 2.5*mm),
                angle=0*degrees)
        t0.draw(kind='phase')

        t0.save_data(filename=filename + '.npz')
        save_figure_test(newpath, func_name)

        t1 = Scalar_mask_XY(x, y, wavelength)
        t1.lens(r0=(0*um, 0*um),
                radius=(100*um, 75*um),
                focal=(5*mm, 2.5*mm),
                angle=45*degrees)
        t1.draw(kind='phase')

        t1.save_data(filename=filename + '.npz', add_name='_elliptical')
        save_figure_test(newpath, func_name, add_name='_elliptical')
        assert True

    def test_lens_cylindrical(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)

        num_data = 512
        length = 400*um
        x = np.linspace(-length/2, length/2, num_data)
        y = np.linspace(-length/2, length/2, num_data)
        wavelength = 0.6328*um

        t0 = Scalar_mask_XY(x, y, wavelength)
        t0.lens_cylindrical(x0=0*um, focal=2.5*mm, angle=0*degrees)
        t0.draw(kind='phase')
        save_figure_test(newpath, func_name)
        t0.save_data(filename=filename + '.npz')
        assert True

    def test_lens_fresnel(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)

        num_data = 512
        length = 500*um
        x = np.linspace(-length/2, length/2, num_data)
        y = np.linspace(-length/2, length/2, num_data)
        wavelength = 0.6328*um

        t1 = Scalar_mask_XY(x, y, wavelength)
        t1.fresnel_lens(r0=(0*um, 0*um),
                        radius=(250*um, 250*um),
                        focal=(5*mm, 5*mm),
                        angle=45*degrees,
                        kind='amplitude',
                        phase=np.pi)
        t1.draw(kind='intensity')
        t1.save_data(filename=filename + '.npz', add_name='_int')
        save_figure_test(newpath, func_name, add_name='_int')

        t1.fresnel_lens(r0=(0*um, 0*um),
                        radius=(250*um, 250*um),
                        focal=(5*mm, 5*mm),
                        angle=0*degrees,
                        kind='phase',
                        phase=np.pi)
        t1.draw(kind='phase')

        t1.save_data(filename=filename + '.npz', add_name='_phase')
        save_figure_test(newpath, func_name, add_name='_phase')
        assert True

    def test_biprism_fresnel(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)

        num_data = 512
        length = 250*um
        x = np.linspace(-length/2, length/2, num_data)
        y = np.linspace(-length/2, length/2, num_data)
        wavelength = 0.6328*um

        t1 = Scalar_mask_XY(x, y, wavelength)
        t1.biprism_fresnel(r0=(0*um, 0*um),
                           width=100*um,
                           height=5*um,
                           n=1.5)
        t1.draw(kind='field')
        save_figure_test(newpath, func_name)
        t1.save_data(filename=filename + '.npz')
        assert True

    def test_axicon(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)

        num_data = 512
        length = 250*um
        x = np.linspace(-length/2, length/2, num_data)
        y = np.linspace(-length/2, length/2, num_data)
        wavelength = 0.6328*um

        t1 = Scalar_mask_XY(x, y, wavelength)
        t1.axicon(r0=(0*um, 0*um),
                  radius=100*um,
                  angle=2*degrees,
                  refractive_index=1.5)
        t1.draw(kind='phase')

        save_figure_test(newpath, func_name)
        t1.save_data(filename=filename + '.npz')
        assert True

    def test_laguerre_gauss_spiral(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)

        num_data = 512
        length = 250*um
        x = np.linspace(-length/2, length/2, num_data)
        y = np.linspace(-length/2, length/2, num_data)
        wavelength = 0.6328*um

        t = Scalar_mask_XY(x, y, wavelength)
        t.laguerre_gauss_spiral(kind='intensity',
                                n=0,
                                l=4,
                                r0=(0*um, 0*um),
                                w0=20*um,
                                z=100*um)
        t.draw(kind='intensity')
        t.save_data(filename=filename + '.npz', add_name='_intensity')
        save_figure_test(newpath, func_name, add_name='_intensity')

        t2 = Scalar_mask_XY(x, y, wavelength)
        t2.laguerre_gauss_spiral(kind='phase',
                                 n=0,
                                 l=16,
                                 r0=(0*um, 0*um),
                                 w0=20*um,
                                 z=4000*um)
        t2.draw(kind='phase')

        t2.save_data(filename=filename + '.npz', add_name='_phase')
        save_figure_test(newpath, func_name, add_name='_phase')
        assert True

    def test_grating_forked(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)

        num_data = 512
        length = 250*um
        x = np.linspace(-length/2, length/2, num_data)
        y = np.linspace(-length/2, length/2, num_data)
        wavelength = 0.6328*um

        t = Scalar_mask_XY(x, y, wavelength)
        t.forked_grating(r0=(0*um, 0*um),
                         period=20*um,
                         l=3,
                         alpha=2,
                         kind='amplitude',
                         angle=0*degrees)
        t.draw(kind='intensity')

        t.save_data(filename=filename + '.npz')
        save_figure_test(newpath, func_name)
        assert True

    def test_roughness(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)

        num_data = 512
        length = 250*um
        x = np.linspace(-length/2, length/2, num_data)
        y = np.linspace(-length/2, length/2, num_data)
        wavelength = 0.6328*um

        t1 = Scalar_mask_XY(x, y, wavelength)
        t1.roughness(t=(25*um, 25*um), s=1*um)
        t1.draw(kind='phase')

        t1.save_data(filename=filename + '.npz')
        save_figure_test(newpath, func_name)
        assert True

    def test_grating_sine(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)

        x = np.linspace(-250*um, 250*um, 512)
        y = np.linspace(-250*um, 250*um, 512)
        wavelength = 0.6238*um
        period = 50*um
        red = Scalar_mask_XY(x, y, wavelength)
        red.sine_grating(period=period,
                         amp_min=0,
                         amp_max=1,
                         x0=0*um,
                         angle=0*degrees)
        red.draw(kind='intensity')

        red.save_data(filename=filename + '.npz')
        save_figure_test(newpath, func_name)
        assert True

    def test_grating_Ronchi(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)

        x = np.linspace(-250*um, 250*um, 512)
        y = np.linspace(-250*um, 250*um, 512)
        wavelength = 0.6238*um
        red = Scalar_mask_XY(x, y, wavelength)
        red.ronchi_grating(period=100*um,
                           x0=0*um,
                           angle=0*degrees,
                           fill_factor=0.333)
        red.draw(kind='intensity')

        red.save_data(filename=filename + '.npz')
        save_figure_test(newpath, func_name)
        assert True

    def test_grating_binary_amplitude(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)

        x = np.linspace(-250*um, 250*um, 512)
        y = np.linspace(-250*um, 250*um, 512)
        wavelength = 0.6238*um
        red = Scalar_mask_XY(x, y, wavelength)
        red.binary_grating(period=50*um,
                           a_min=.5,
                           a_max=.75,
                           phase=np.pi,
                           x0=0,
                           fill_factor=0.25,
                           angle=0*degrees)
        red.draw(kind='field')

        red.save_data(filename=filename + '.npz')
        save_figure_test(newpath, func_name)
        assert True

    def test_grating_binary_phase(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)

        x = np.linspace(-250*um, 250*um, 512)
        y = np.linspace(-250*um, 250*um, 512)
        wavelength = 0.6238*um
        red = Scalar_mask_XY(x, y, wavelength)
        red.binary_grating(period=50*um,
                           a_min=1,
                           a_max=1,
                           phase=np.pi/2,
                           x0=0,
                           fill_factor=0.5,
                           angle=0*degrees)
        red.draw(kind='phase')

        red.save_data(filename=filename + '.npz')
        save_figure_test(newpath, func_name)
        assert True

    def test_grating_blazed(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)

        x = np.linspace(-250*um, 250*um, 512)
        y = np.linspace(-250*um, 250*um, 512)
        wavelength = 0.6238*um
        period = 125*um
        red = Scalar_mask_XY(x, y, wavelength)
        red.blazed_grating(period=period,
                           phase_max=2 * np.pi,
                           x0=0,
                           angle=0*degrees)
        red.draw(kind='phase')

        red.save_data(filename=filename + '.npz')
        save_figure_test(newpath, func_name)
        assert True

    def test_grating_2D(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)

        x = np.linspace(-250*um, 250*um, 512)
        y = np.linspace(-250*um, 250*um, 512)
        wavelength = 0.6238*um
        period = 62.5*um
        red = Scalar_mask_XY(x, y, wavelength)
        red.grating_2D(r0=(0., 0.),
                       period=period,
                       a_min=0,
                       a_max=1.,
                       phase=0 * np.pi/2,
                       fill_factor=0.5,
                       angle=0*degrees)
        red.draw(kind='intensity')

        red.save_data(filename=filename + '.npz')
        save_figure_test(newpath, func_name)
        assert True

    def test_grating_chess(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)

        x = np.linspace(-250*um, 250*um, 512)
        y = np.linspace(-250*um, 250*um, 512)
        wavelength = 0.6238*um
        period = 125*um
        red = Scalar_mask_XY(x, y, wavelength)
        red.grating_2D_chess(r0=(0., 0.),
                             period=period,
                             a_min=0,
                             a_max=1.,
                             phase=0 * np.pi/2,
                             fill_factor=0.75,
                             angle=0*degrees)
        red.draw(kind='intensity')

        red.save_data(filename=filename + '.npz')
        save_figure_test(newpath, func_name)
        assert True

    def test_circle_rough(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)

        num_data = 512
        length = 250*um
        x = np.linspace(-length/2, length/2, num_data)
        y = np.linspace(-length/2, length/2, num_data)
        wavelength = 0.6238*um

        t = Scalar_mask_XY(x, y, wavelength)
        t.circle_rough(r0=(0*um, 0*um),
                       radius=100*um,
                       angle=0*degrees,
                       sigma=4*um)
        t.draw(kind='intensity')
        t.save_data(filename=filename + '.npz')
        save_figure_test(newpath, func_name)
        assert True

    def test_ring_rough(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)

        x = np.linspace(-250*um, 250*um, 1024)
        y = np.linspace(-250*um, 250*um, 1024)
        wavelength = 0.6238*um

        t = Scalar_mask_XY(x, y, wavelength)
        t.ring_rough(r0=(0*um, 0*um),
                     radius1=50*um,
                     radius2=100*um,
                     angle=0*degrees,
                     sigma=4*um)
        t.draw(kind='intensity')

        t.save_data(filename=filename + '.npz')
        save_figure_test(newpath, func_name)
        assert True

    def test_widen(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)

        num_data = 512
        length = 250*um
        x = np.linspace(-length/2, length/2, num_data)
        y = np.linspace(-length/2, length/2, num_data)
        wavelength = 0.6328*um

        t1 = Scalar_mask_XY(x, y, wavelength)
        t1.ring(r0=(0*um, 0*um),
                radius1=(50*um, 50*um),
                radius2=(60*um, 60*um),
                angle=45*degrees)
        t1.draw()
        t1.save_data(filename=filename + '.npz', add_name='_no_widened')
        save_figure_test(newpath, func_name, add_name='_no_widened')

        t1.widen(radius=50*um)
        # t1.u = np.abs(t1.u)
        # t1.u[t1.u < 0.5] = 0
        # t1.u[t1.u >= 0.5] = 1
        t1.draw()

        t1.save_data(filename=filename + '.npz', add_name='_widened')
        save_figure_test(newpath, func_name, add_name='_widened')
        assert True

    def test_compute_area(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)

        num_data = 512
        length = 250*um
        radius = length/2
        x = np.linspace(-radius, radius, num_data)
        y = np.linspace(-radius, radius, num_data)
        wavelength = 0.6328*um

        t1 = Scalar_mask_XY(x, y, wavelength)
        t1.circle(r0=(0*um, 0*um),
                  radius=(100*um, 100*um),
                  angle=0*degrees)
        t1.draw(title='test_ring')

        area = t1.area(percentage=0.001)
        text1 = "The area is: {:2.4f} %".format(area * 100)
        plt.title(text1)

        t1.save_data(filename=filename + '.npz')
        save_figure_test(newpath, func_name)
        assert True
