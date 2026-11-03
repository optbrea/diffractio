# !/usr/bin/env python3

"""tests for scalar_fields_X"""

import copyreg
import datetime
import os
import sys
import time
import types

import numpy as np
from diffractio import degrees, mm, no_date, num_max_processors, plt, um
from diffractio.scalar_fields_X import (Scalar_field_X,
                                        extended_polychromatic_source,
                                        extended_source_multiprocessing,
                                        polychromatic_multiprocessing)
from diffractio.scalar_masks_X import Scalar_mask_X
from diffractio.scalar_sources_X import Scalar_source_X
from diffractio.utils_multiprocessing import (_pickle_method, _unpickle_method,
                                              execute_multiprocessing)
from diffractio.utils_optics import gauss_spectrum
from diffractio.utils_tests import save_figure_test

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


def __experiment_extended_source__(x0):

    x = np.linspace(-500*um, 500*um, 1024)
    wavelength = 0.850*um
    z0 = -500*mm
    period = 50*um
    focal = 5*mm

    red = Scalar_mask_X(x, wavelength)
    red.ronchi_grating(x0=0*um, period=period, fill_factor=0.5)

    lens = Scalar_mask_X(x, wavelength)
    lens.lens(x0=0, focal=focal, radius=30*mm)

    u1 = Scalar_source_X(x, wavelength)

    u1.spherical_wave(A=1., x0=x0, z0=z0, normalize=True)
    u2 = u1 * red * lens
    u2.RS(z=focal, new_field=False, verbose=False)

    return u2


def __function_polychromatic__(wavelength):
    x0 = np.linspace(-75*um, 75*um, 1024 * 8)

    t1 = Scalar_mask_X(x0, wavelength)
    t1.slit(x0=0, size=100*um)

    f1 = Scalar_source_X(x0, wavelength)
    f1.gauss_beam(x0=0*um, w0=200*um, z0=0*um, A=1, theta=0*degrees)

    u1 = f1 * t1
    u1.RS(z=10*um, new_field=False, verbose=False)

    return u1


def __benchmark_RS_propagation__(parameters):
    x = np.linspace(-100*um, 100*um, parameters['num_points'])
    field = Scalar_source_X(x, parameters['wavelength'])
    field.gauss_beam(
        x0=parameters['x0'],
        w0=parameters['w0'],
        z0=0,
        A=1,
        theta=0,
    )

    for z in parameters['distances']:
        field = field.RS(z=z, new_field=True, verbose=False)

    return field.u


def __experiment_grating_movement__(dict_params):
    delta_x = dict_params['delta_x']
    period = dict_params['period']
    t1 = dict_params['t1']
    t2 = dict_params['t2']
    t2.ronchi_grating(period=period, x0=delta_x, fill_factor=0.5)
    u2 = t1 * t2
    return u2


class Test_Scalar_fields_X():

    def test_extended_source_simple(self):
        """
        tomamos una red de difraccion y analizamos el efecto Lau con fuente extensa
        La fuente extensa la definimos como suma de fuentes puntuales esfericas
        Esquema: Fuente extensa, red, lens, vista en plano focal
        """
        func_name = sys._getframe().f_code.co_name
        # filename = '{}{}.npz'.format(newpath, func_name)

        # definiciones
        x = np.linspace(-500*um, 500*um, 1024)
        wavelength = 0.850*um
        num_fuentes = 101
        S = 3000*um
        z0 = -500*mm
        period = 50*um
        focal = 5*mm

        red = Scalar_mask_X(x, wavelength)
        red.ronchi_grating(x0=0*um, period=period, fill_factor=0.5)

        lens = Scalar_mask_X(x, wavelength)
        lens.lens(x0=0, focal=focal, radius=30*mm)

        u1 = Scalar_source_X(x, wavelength)

        # posiciones de la fuente
        x0s = np.linspace(-S/2, S/2, num_fuentes)
        intensities = Scalar_field_X(x, wavelength)

        time1 = time.time()
        for x0 in x0s:
            u1.spherical_wave(A=1, x0=x0, z0=z0, normalize=True)
            u2 = u1 * lens * red
            u2.RS(z=focal, new_field=False, verbose=False)
            intensities.u = intensities.u + abs(u2.u)**2

        intensities.u = intensities.u / intensities.u.max()
        time_proc = time.time() - time1
        print("num_proc: {}, time={}".format(1, time_proc))
        intensities.draw(kind='amplitude')

        save_figure_test(newpath, func_name)

        assert True

    def test_polychromatic(self):
        func_name = sys._getframe().f_code.co_name
        # filename = '{}{}.npz'.format(newpath, func_name)

        wavelengths = np.linspace(.3*um, .9*um, 101)
        w_central = wavelengths.mean()
        spectrum_gauss = gauss_spectrum(wavelengths=wavelengths,
                                        w_central=w_central,
                                        Dw=0.4,
                                        normalize=True)

        intensity, u_s, time_proc = polychromatic_multiprocessing(
            __function_polychromatic__,
            wavelengths,
            spectrum_gauss,
            num_processors=1,
            verbose=True)
        intensity0, u_s0, time_proc0 = polychromatic_multiprocessing(
            __function_polychromatic__,
            w_central,
            1,
            num_processors=1,
            verbose=True)

        plt.figure()
        plt.plot(u_s0.x, intensity0, 'k', lw=2, label='monochromatic')
        plt.plot(u_s[0].x, intensity, 'r', lw=2, label='polychromatic')
        plt.legend()

        save_figure_test(newpath, func_name)
        assert True

    def test_extended_source_multiprocessing(self):
        """Tomamos una red de difraccion y analizamos el efecto Lau con fuente extensa
            La fuente extensa la definimos como suma de fuentes puntuales esfericas
            Esquema: Fuente extensa, red, lens, vista en plano focal
        """
        func_name = sys._getframe().f_code.co_name
        # filename = '{}{}.npz'.format(newpath, func_name)

        x0s = np.linspace(-1500*um, 1500*um, 101)
        x0_central = 0*um

        intensity, u_s, time_proc = extended_source_multiprocessing(
            __experiment_extended_source__,
            x0s,
            num_processors=1,
            verbose=True)
        intensity0, u_s0, time_proc0 = extended_source_multiprocessing(
            __experiment_extended_source__,
            x0_central,
            num_processors=1,
            verbose=True)

        plt.figure()
        plt.plot(u_s0.x, intensity0, 'k', lw=2, label='punctual source')
        plt.plot(u_s[0].x, intensity, 'r', lw=2, label='extended source')
        plt.legend()

        save_figure_test(newpath, func_name)
        assert True

    def test_grating_movement(self):
        func_name = sys._getframe().f_code.co_name

        # filename = '{}{}.npz'.format(newpath, func_name)

        def creation_dictionary(deltas_x, period, t1, t2):
            # create Args: for multiprocessing
            dict_Parameters = []
            for i, delta_x in enumerate(deltas_x):
                dict_Parameters.append(
                    dict(delta_x=delta_x, period=period, t1=t1, t2=t2))
            return dict_Parameters

        x0 = np.linspace(-400*um, 400*um, 1024 * 8)
        wavelength = 0.85*um
        period = 50*um
        z_talbot = 2 * period**2 / wavelength
        z0 = z_talbot/2
        t1 = Scalar_mask_X(x0,
                           wavelength,
                           info="__experiment_grating_movement__")
        t1.ronchi_grating(period=period, x0=0*um, fill_factor=0.5)
        t1.RS(z=z0, new_field=False)

        t2 = Scalar_mask_X(x0,
                           wavelength,
                           info="__experiment_grating_movement__")
        t2.ronchi_grating(period=period, x0=0*um, fill_factor=0.5)

        deltas_x = np.linspace(-60*um, 60*um, 51)  # 512
        num_processors = 1

        dict_Parameters = creation_dictionary(deltas_x=deltas_x,
                                              period=period,
                                              t1=t1,
                                              t2=t2)

        u_s, time_proc = execute_multiprocessing(
            __experiment_grating_movement__,
            dict_Parameters,
            num_processors,
            verbose=True)

        x = u_s[0].x

        perfil = np.zeros_like(deltas_x)

        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        h1, = ax1.plot(x, np.zeros_like(x), 'k', lw=2)
        ax1.set_xlim(x[0], x[-1])
        ax1.set_ylim(0, 2)
        ax1.set_xlabel(r'$x (\mu m)$')

        ax2 = fig.add_subplot(212)
        h2, = ax2.plot(deltas_x, perfil, 'k', lw=2)
        ax2.set_xlim(deltas_x[0], deltas_x[-1])
        ax2.set_ylim(0, .5)
        ax2.set_xlabel(r'$\Delta x (\mu m)$')

        incr_frames = 1
        for i in range(0, len(deltas_x), incr_frames):
            intensidad = abs(u_s[i].u)**2  # sacar fuera
            perfil[i] = intensidad.mean()
            plt.suptitle(r"$\delta x={:6.2f}\,\mu m$".format(deltas_x[i]),
                         fontsize=18)
            h1.set_ydata(intensidad)
            h2.set_ydata(perfil)
            plt.draw()
        save_figure_test(newpath, func_name)

        assert True


def __experiment_double_slit_dictionary__(dict_params):
    x0 = dict_params['x0']
    wavelength = dict_params['wavelength']
    z = dict_params['z']
    slit_size = dict_params['slit_size']
    separation = dict_params['separation']
    t1 = Scalar_mask_X(x0, wavelength)
    t1.slit(x0=0, size=20*um)
    t1.double_slit(x0=0, size=slit_size, separation=separation)

    f1 = Scalar_source_X(x0, wavelength)
    f1.gauss_beam(x0=0*um, w0=200*um, z0=0*um, A=1, theta=0*degrees)

    u1 = f1 * t1
    u1.RS(z, new_field=False, verbose=False)

    return u1


def __experiment_double_slit_array__(slit_size):
    x0 = np.linspace(-75*um, 75*um, 1024 * 8)
    wavelength = 0.6328*um
    separation = 50*um
    z = 100*um

    t1 = Scalar_mask_X(x0, wavelength)
    t1.slit(x0=0, size=20*um)
    t1.double_slit(x0=0, size=slit_size, separation=separation)

    f1 = Scalar_source_X(x0, wavelength)
    f1.gauss_beam(x0=0*um, w0=200*um, z0=0*um, A=1, theta=0*degrees)

    u1 = f1 * t1
    u1.RS(z, new_field=False, verbose=False)

    return u1


def __experiment_extended_polychromatic_source__(dict_params):
    wavelength = dict_params['wavelength']
    x0 = dict_params['x0']

    x = np.linspace(-1250*um, 1250*um, 1024 * 8)
    periodo = 100*um
    z = 5 * periodo**2 / (0.6*um)
    z0 = -50*mm

    t1 = Scalar_mask_X(x, wavelength)
    t1.ronchi_grating(period=periodo, x0=0*um, fill_factor=0.5)

    f1 = Scalar_source_X(x, wavelength)
    f1.spherical_wave(A=1, x0=x0, z0=z0, normalize=True)

    u1 = f1 * t1
    u1.RS(z, new_field=False, verbose=False)

    return u1


class Test_Scalar_fields_X_multiprocessing():

    def test_multiprocessing_propagation_benchmark(self):
        parameters = [
            {
                'num_points': 131072,
                'wavelength': (0.5 + index * 0.01) * um,
                'x0': (index - 5.5) * um,
                'w0': (25 + index) * um,
                'distances': (10 * um, 20 * um, 30 * um, 40 * um),
            }
            for index in range(48)
        ]
        num_processors = num_max_processors-2

        start = time.perf_counter()
        sequential_results = [
            __benchmark_RS_propagation__(item) for item in parameters
        ]
        sequential_time = time.perf_counter() - start

        parallel_results, parallel_time = execute_multiprocessing(
            __benchmark_RS_propagation__,
            parameters,
            num_processors,
            verbose=False,
        )

        for sequential, parallel in zip(sequential_results, parallel_results):
            np.testing.assert_allclose(parallel, sequential, rtol=1e-10, atol=1e-12)

        speedup = sequential_time / parallel_time
        print(
            f"Propagation benchmark: sequential={sequential_time:.3f}s, "
            f"{num_processors} processes={parallel_time:.3f}s, "
            f"speedup={speedup:.2f}x"
        )

    def test_multiprocessing_dictionary(self):
        func_name = sys._getframe().f_code.co_name

        # filename = '{}{}.npz'.format(newpath, func_name)

        def creation_dictionary(wavelengths, x0, z, slit_size, separation):
            # create Args: for multiprocessing
            dict_Parameters = []
            for i, wavelength in enumerate(wavelengths):
                dict_Parameters.append(
                    dict(x0=x0,
                         wavelength=wavelength,
                         z=z,
                         slit_size=slit_size,
                         separation=separation))
            return dict_Parameters

        x0 = np.linspace(-100*um, 100*um, 1024)
        wavelengths = np.linspace(0.3*um, 0.8*um, 101)
        spectrum_gauss = gauss_spectrum(wavelengths=wavelengths,
                                        w_central=0.6,
                                        Dw=0.1,
                                        normalize=True)
        num_processors = 1

        dict_Parameters0 = dict(x0=x0,
                                wavelength=0.6*um,
                                z=.05*mm,
                                slit_size=50*um,
                                separation=75*um)

        dict_Parameters = creation_dictionary(wavelengths=wavelengths,
                                              x0=x0,
                                              z=.05*mm,
                                              slit_size=50*um,
                                              separation=75*um)

        I0 = __experiment_double_slit_dictionary__(dict_Parameters0)

        I_wavelengths, time_proc = execute_multiprocessing(
            __experiment_double_slit_dictionary__,
            dict_Parameters,
            num_processors,
            verbose=False)

        print("num_proc= {}, time={}".format(num_processors, time_proc))

        intensity = np.zeros_like(I_wavelengths[0].x)
        for i in range(len(wavelengths)):
            intensity = intensity + spectrum_gauss[i] * abs(
                I_wavelengths[i].u)**2

        plt.figure()
        plt.plot(x0, abs(I0.u)**2, 'k', lw=2, label='monochromatic')
        plt.plot(x0, intensity, 'r', lw=2, label='polychromatic')
        plt.title('polychromatic')
        plt.legend()

        save_figure_test(newpath, func_name)

        assert True

    def test_multiprocessing_array(self):
        func_name = sys._getframe().f_code.co_name
        # filename = '{}{}.npz'.format(newpath, func_name)

        slit_sizes = np.linspace(5*um, 50*um, 10)
        num_processors = 1

        u_s, time_proc = execute_multiprocessing(
            __experiment_double_slit_array__,
            slit_sizes,
            num_processors,
            verbose=True)

        plt.figure()
        ax = plt.subplot(111)
        for i, slit_size in enumerate(slit_sizes):
            ax.plot(u_s[i].x,
                    abs(u_s[i].u)**2 + 2 * i,
                    lw=2,
                    label=r"${:2.2f}\,\mu m$".format(slit_size))
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        # Put a legend to the right of the current axis
        ax.legend(fontsize=8,
                  frameon=False,
                  loc='center left',
                  bbox_to_anchor=(1, 0.5))

        save_figure_test(newpath, func_name)
        assert True

    def test_extended_polychromatic_source(self):
        func_name = sys._getframe().f_code.co_name
        # filename = '{}{}.npz'.format(newpath, func_name)

        x0s = np.linspace(-150*um, 150*um, 21)
        wavelengths = np.linspace(.45*um, .65*um, 11)
        w_central = wavelengths.mean()
        spectrum_gauss = gauss_spectrum(wavelengths=wavelengths,
                                        w_central=w_central,
                                        Dw=0.4,
                                        normalize=True)

        u0 = __experiment_extended_polychromatic_source__(
            dict(x0=0, wavelength=w_central))

        intensity, u_s, time_proc = extended_polychromatic_source(
            __experiment_extended_polychromatic_source__,
            x0s,
            wavelengths,
            spectrum_gauss,
            num_processors=1,
            verbose=True)

        plt.figure()
        plt.plot(u_s[0].x, abs(u0.u)**2, 'k', lw=1, label='mono')
        plt.plot(u_s[0].x, intensity, 'r', lw=2, label='LED')
        plt.legend()

        save_figure_test(newpath, func_name)
        assert True
