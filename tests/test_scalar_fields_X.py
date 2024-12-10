# !/usr/bin/env python3

"""tests for scalar_fields_X"""

import copyreg
import datetime
import os
import sys
import types

import numpy as np
from diffractio import eps, no_date, plt, um, mm
from diffractio.scalar_fields_X import Scalar_field_X
from diffractio.scalar_masks_X import Scalar_mask_X
from diffractio.utils_multiprocessing import _pickle_method, _unpickle_method
from diffractio.utils_tests import comparison, save_figure_test

copyreg.pickle(types.MethodType, _pickle_method, _unpickle_method)

if no_date is True:
    date = '0'
else:
    now = datetime.datetime.now()
    date = now.strftime("%Y-%m-%d_%H")

path_base = "tests_results"
path_class = "scalar_fields_X"

newpath = "{}_{}/{}/".format(path_base, date, path_class)

if not os.path.exists(newpath):
    os.makedirs(newpath)


class Test_Scalar_fields_X():

    def test_add(self):
        pass

    def test_sub(self):
        pass

    def test_mul(self):
        pass

    def test_clear_field(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)

        x = np.linspace(-50*um, 50*um, 128)
        wavelength = 1*um
        t1 = Scalar_field_X(x, wavelength)
        t1.u = np.ones_like(x, dtype=complex)

        t1.save_data(filename=filename + '.npz')

        solution = np.zeros_like(t1.u)
        t1.clear_field()
        proposal = t1.u
        assert comparison(proposal, solution, eps), func_name

    def test_save_load(self):
        """
        Tests save in savez and other functions
        """
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)

        x = np.linspace(-500*um, 500*um, 512)
        wavelength = .5*um

        t1 = Scalar_field_X(x, wavelength)
        t1.u = np.sin(x**2 / 5000)
        t1.draw()

        save_figure_test(newpath, func_name, add_name='_saved')
        t1.save_data(filename=filename + '.npz')

        t2 = Scalar_field_X(x, wavelength)
        t2.load_data(filename=filename + '.npz')
        t2.draw()
        save_figure_test(newpath, func_name, add_name='_loaded')

        assert True

    def test_amplitude_phase(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)

        x = np.linspace(-1*mm, 1*mm, 1024)
        wavelength = 0.6328*um

        u0 = Scalar_mask_X(x, wavelength)
        u0.slit(x0=0, size=1*mm)
        u0.inverse_amplitude()
        u0.draw()
        save_figure_test(newpath, func_name, add_name='_1')

        u1 = Scalar_mask_X(x, wavelength)
        u1.lens(x0=0, focal=100*mm)
        u1.inverse_phase()
        u1.draw('phase')
        save_figure_test(newpath, func_name, add_name='_2')

        assert True, func_name

    def test_cut_resample(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)

        x1 = np.linspace(-500*um, 500*um, 512)
        wavelength = .5*um

        t1 = Scalar_field_X(x1, wavelength)
        t1.u = np.sin(2 * np.pi * x1 / 50)
        t1.draw()

        t1.save_data(filename=filename)
        save_figure_test(newpath, func_name, add_name='_1')

        t2 = t1.cut_resample(x_limits=(-250*um, 250*um),
                             num_points=1024,
                             new_field=True,
                             interp_kind='quadratic')
        t2.draw()
        save_figure_test(newpath, func_name, add_name='_2')
        t2.save_data(filename=filename + '2' + '.npz')

        solution = np.array((512, 1024))
        proposal = np.array((len(t1.u), len(t2.u)))
        assert comparison(proposal, solution, eps), func_name

    def test_insert_mask(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)

        x1 = np.linspace(-500*um, 500*um, 512)
        wavelength = .5*um
        t1 = Scalar_field_X(x1, wavelength)
        t1.clear_field()

        x2 = np.linspace(-50*um, 50*um, 512)
        wavelength = .5*um
        t2 = Scalar_field_X(x2, wavelength)
        t2.u = np.sin(2 * np.pi * x2 / 50)

        t1.insert_mask(t2,
                       x0_mask1=-100*um,
                       clean=False,
                       kind_position='center')
        t1.insert_mask(t2,
                       x0_mask1=100*um,
                       clean=False,
                       kind_position='center')
        t1.draw()

        t1.save_data(filename=filename + '.npz')
        save_figure_test(newpath, func_name)
        assert True, func_name

    def test_pupil(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)

        x = np.linspace(-1*mm, 1*mm, 1024)
        wavelength = 0.6328*um

        t1 = Scalar_mask_X(x, wavelength)
        t1.lens(x0=0, focal=100*mm)
        t1.pupil(x0=0, radius=750*um)
        t1.draw('field')

        t1.save_data(filename=filename + '.npz')
        save_figure_test(newpath, func_name)
        assert True, func_name

    def test_insert_array_mask(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)

        x1 = np.linspace(-750*um, 750*um, 512)
        wavelength = .5*um

        t1 = Scalar_field_X(x1, wavelength)
        t1.clear_field()

        x2 = np.linspace(-50*um, 50*um, 512)
        wavelength = .5*um
        t2 = Scalar_field_X(x2, wavelength)
        t2.u = np.sin(2 * np.pi * x2 / 50)

        t1.insert_array_masks(t2,
                              x_pos=[-400, -200, 0, 200, 400],
                              clean=True,
                              kind_position='center')
        t1.draw()

        t1.save_data(filename=filename + '.npz')
        save_figure_test(newpath, func_name, add_name='_1')

        assert True, func_name

    def test_transitions(self):
        """
        test a binary mask and obtain positions of transitions
        """
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)

        x1 = np.linspace(-750*um, 750*um, 4096)

        x2 = np.linspace(-50*um, 50*um, 4096)
        wavelength = .5*um

        t2 = Scalar_field_X(x2, wavelength)
        t2.u = np.ones_like(x2, dtype=complex)

        t1 = Scalar_field_X(x1, wavelength)
        t1.insert_array_masks(t2,
                              x_pos=[-400, -200, 0, 200, 400],
                              clean=True,
                              kind_position='center')

        pos_transitions, type_transitions, raising, falling = t1.get_edges(
            kind_transition='amplitude', min_step=.05)
        print(pos_transitions)
        print(type_transitions)
        t1.draw(kind='intensity')

        t1.save_data(filename=filename + '.npz')
        save_figure_test(newpath, func_name)

        solution = np.linspace(-450*um, 450*um, 10)
        proposal = pos_transitions
        assert comparison(proposal, solution, 1), func_name

    assert True

    def test_fft(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)

        x = np.linspace(-500*um, 500*um, 256)
        wavelength = 1*um

        x = np.linspace(-500*um, 500*um, 128)
        wavelength = .5*um

        t1 = Scalar_field_X(x, wavelength)
        t1.u = np.sin(2 * np.pi * x / 100)

        t2 = t1.fft(z=None,
                    shift=True,
                    remove0=False,
                    matrix=False,
                    new_field=True,
                    verbose=False)
        t2.draw()
        save_figure_test(newpath, func_name, add_name='_1')
        t2.save_data(filename=filename + '.npz')
        assert True

    def test_ifft(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)

        x = np.linspace(-500*um, 500*um, 256)
        wavelength = 1*um

        x = np.linspace(-500*um, 500*um, 128)
        wavelength = .5*um

        t1 = Scalar_field_X(x, wavelength)
        t1.u = np.sin(2 * np.pi * x / 100)
        t1.save_data(filename=filename + '_direct' + '.npz')

        t2 = t1.fft(z=None,
                    shift=True,
                    remove0=False,
                    matrix=False,
                    new_field=True,
                    verbose=False)
        t2.draw()
        save_figure_test(newpath, func_name, add_name='_direct')

        t3 = t2.ifft(z=None,
                     shift=True,
                     remove0=False,
                     matrix=False,
                     new_field=True,
                     verbose=False)
        t3.draw()
        save_figure_test(newpath, func_name, add_name='_ifft')
        t3.save_data(filename=filename + '_ifft' + '.npz')

        assert True

    def test_RS(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)

        num_data = 4096

        x = np.linspace(-500*um, 500*um, num_data)
        wavelength = 1*um

        t1 = Scalar_field_X(x, wavelength)
        t1.u[np.bitwise_and(x > -200*um, x < 200*um)] = 1
        t1.draw()
        save_figure_test(newpath, func_name, add_name='_mask')

        u1 = t1.RS(z=2000*um, new_field=True)
        u1.draw(kind='intensity')
        u1.save_data(filename=filename + '2' + '.npz')

        save_figure_test(newpath, func_name, add_name='_RS')

        assert True

    def test_RS_amplification(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)

        num_data = 4096

        x = np.linspace(-500*um, 500*um, num_data)
        wavelength = 1*um

        t1 = Scalar_field_X(x, wavelength)
        t1.u[np.bitwise_and(x > -200*um, x < 200*um)] = 1
        t1.draw()
        save_figure_test(newpath, func_name, add_name='_mask')

        u1 = t1.RS(z=2000*um, new_field=True, amplification=3)
        u1.draw(kind='intensity')
        t1.save_data(filename=filename + '2' + '.npz')

        save_figure_test(newpath, func_name, add_name='_RS*3')

        assert True

    def test_BPM(self):
        pass

    def test_MTF(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)

        num_data = 1024 * 8
        x = np.linspace(-50*um, 50*um, num_data)
        wavelength = 0.6328*um

        # intensidad de una onda plana
        u0 = Scalar_field_X(x, wavelength)
        u0.u[np.bitwise_and(x > -4*um, x < 4*um)] = 1
        u0.draw(kind='intensity')
        save_figure_test(newpath, func_name, add_name='_mask')
        u0.save_data(filename=filename + '_mask' + '.npz')

        u0.MTF(kind='mm', has_draw=True)
        plt.xlim(-250, 250)
        save_figure_test(newpath, func_name, add_name='_mtf')
        assert True

    def test_intesity(self):
        pass

    def test_average_intensity(self):
        pass

    def test_draw(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)

        x = np.linspace(-500*um, 500*um, 4096)
        wavelength = .5*um

        t1 = Scalar_field_X(x, wavelength)
        t1.u = np.sin(2 * np.pi * x / 100) * np.exp(1j * 2 * np.pi * x / 100)

        t1.draw(kind='amplitude')
        save_figure_test(newpath, func_name, add_name='_amp')
        t1.draw(kind='intensity')
        save_figure_test(newpath, func_name, add_name='_int')
        t1.draw(kind='phase')
        save_figure_test(newpath, func_name, add_name='_phase')
        t1.draw(kind='fill')
        save_figure_test(newpath, func_name, add_name='_fill')
        t1.draw(kind='field')

        t1.save_data(filename=filename + '.npz')
        save_figure_test(newpath, func_name, add_name='_field')

        assert True
