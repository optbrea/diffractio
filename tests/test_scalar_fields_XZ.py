# !/usr/bin/env python3

"""Tests for Scalar_mask_XZ"""
import datetime
import os
import sys
import time
import pytest

from diffractio import (degrees, eps, mm, no_date, np, num_max_processors, plt,
                        um)
from diffractio.scalar_fields_XZ import Scalar_field_XZ
from diffractio.scalar_masks_X import Scalar_mask_X
from diffractio.scalar_masks_XZ import Scalar_mask_XZ
from diffractio.scalar_sources_X import Scalar_source_X
from diffractio.utils_tests import comparison, save_figure_test

if no_date is True:
    date = '0'
else:
    now = datetime.datetime.now()
    date = now.strftime("%Y-%m-%d_%H")

path_base = "tests_results"
path_class = "scalar_fields_XZ"

newpath = "{}_{}/{}/".format(path_base, date, path_class)

if not os.path.exists(newpath):
    os.makedirs(newpath)


def _func_polychromatic_RS_(wavelength):
    x0 = np.linspace(-100*um, 100*um, 512)

    f1 = Scalar_source_X(x0, wavelength)
    f1.gauss_beam(A=1, x0=0, z0=0*um, w0=50*um, theta=0*degrees)

    t1 = Scalar_mask_X(x0, wavelength)
    t1.ronchi_grating(period=10*um, x0=0*um, fill_factor=0.5)

    initial_field = t1
    return initial_field


def _func_polychromatic_BPM_(wavelength):
    x0 = np.linspace(-200*um, 200*um, 512)

    t1 = Scalar_mask_X(x0, wavelength)
    t1.ronchi_grating(period=15*um, x0=0*um, fill_factor=0.5)

    f1 = Scalar_source_X(x0, wavelength)
    f1.gauss_beam(A=1, x0=0, z0=0*um, w0=100*um, theta=0*degrees)

    z = np.linspace(0*um, 5*mm, 512)
    u1 = Scalar_mask_XZ(x0, z, wavelength, n_background=1)
    u1.incident_field(f1 * t1)

    u1.square(r0=(-100*um, 1500*um),
                 size=(200*um, 1000*um),
                 angle=0*degrees,
                 refractive_index=4)

    return u1


def generate_BPM_field():
    x0 = np.linspace(-25*um, 25*um, 512)
    z0 = np.linspace(0*um, 75*um, 256)
    wavelength = .5*um
    u0 = Scalar_source_X(x=x0, wavelength=wavelength)
    u0.plane_wave(A=1, theta=0*degrees)
    u1 = Scalar_mask_XZ(x=x0, z=z0, wavelength=wavelength)
    u1.incident_field(u0)
    u1.cylinder(r0=(0*um, 0*um),
              radius=(25*um, 25*um),
              refractive_index=2)
    u1.BPM(verbose=False)

    return u1


def generate_BPM_gauss():

    length = 500*um
    wavelength = .5*um
    x0 = np.linspace(-length/2, length/2, 512)
    z0 = np.linspace(0*um, 500*um, 256)

    x_ini = -250*um * np.tan(30*degrees)
    print("x_ini={}".format(x_ini))
    # source
    f1 = Scalar_source_X(x0, wavelength)
    f1.gauss_beam(A=1, x0=0, z0=250*um, w0=10*um, theta=0*degrees)

    u1 = Scalar_field_XZ(x=x0, z=z0, wavelength=wavelength)
    u1.clear_refractive_index()
    u1.incident_field(f1)
    u1.BPM(verbose=False)
    return u1


u_focus = generate_BPM_field()
u_gauss = generate_BPM_gauss()


class Test_Scalar_fields_XZ():

    def test_rotate_field(self):
        func_name = sys._getframe().f_code.co_name
        # filename = '{}{}'.format(newpath, func_name)

        x0 = np.linspace(-200*um, 200*um, 512)
        z0 = np.linspace(-100*um, 600*um, 512)
        wavelength = 5*um

        u0 = Scalar_source_X(x=x0, wavelength=wavelength)
        u0.plane_wave(A=1, theta=0*degrees)
        u1 = Scalar_mask_XZ(x=x0, z=z0, wavelength=wavelength)
        u1.incident_field(u0)

        u1.lens(r0=(0, 100),
                           size=300*um,
                           radii=(1000*um, -250*um),
                           thickness=100*um,
                           refractive_index=2,
                           angle=0*degrees,
                           mask=(10*um, 3 + 0.05j))

        u1.smooth_refractive_index(type_filter=3,
                                   pixels_filtering=2,
                                   max_diff_filter=0.01,
                                   draw_check=False)
        u1.BPM(verbose=False)

        u1.draw(logarithm=True,
                normalize='maximum',
                scale='equal',
                draw_borders=True)
        save_figure_test(newpath, func_name, add_name='_wo')

        u1.draw_refractive_index(draw_borders=True)

        save_figure_test(newpath, func_name, add_name='_no')

        u1.rotate_field(angle=22.5*degrees,
                        center_rotation=(0, 100),
                        kind='jones_ap')
        u1.draw_refractive_index(draw_borders=True)
        save_figure_test(newpath, func_name, add_name='_n')
        u1.draw(logarithm=True, draw_borders=True)
        save_figure_test(newpath, func_name, add_name='_wi')

        u1.clear_field()
        u1.BPM(verbose=False)
        u1.draw(logarithm=True, draw_borders=True)

        save_figure_test(newpath, func_name, add_name='_recomputed')

        assert True

    def test_clear_field(self):
        func_name = sys._getframe().f_code.co_name
        # filename = '{}{}.hkl'.format(newpath, func_name)

        u0 = generate_BPM_gauss()
        proposal = 0 * u0.u

        u0.clear_field()
        solution = u0.u

        u0.draw(kind='intensity', colorbar_kind='horizontal')
        plt.clim(0, 1)

        save_figure_test(newpath, func_name, add_name='')
        assert comparison(proposal, solution, eps)

    def test_save_load(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)

        x0 = np.linspace(-25*um, 25*um, 512)
        z0 = np.linspace(0*um, 75*um, 256)
        wavelength = .5*um

        u1 = Scalar_mask_XZ(x=x0, z=z0, wavelength=wavelength)
        u1.cylinder(r0=(0*um, 0*um),
                  radius=(25*um, 25*um),
                  refractive_index=2)

        u1.info = """info:
            test_save_load():
            se graba una máscara para ver si se carga luego bien
            Se pueder en nombre, la fecha, etc.
            name: prueba
            date: 170731
            purpose: check testing
            """
        u1.save_data(filename=filename + '.npz', add_name='')
        time.sleep(0.5)

        u2 = Scalar_field_XZ(x0, z0, wavelength)
        u2.load_data(filename=filename + '.npz')
        u2.draw(logarithm=True, normalize='maximum', draw_borders=True)
        save_figure_test(newpath, func_name, add_name='_loaded')
        assert True

    def test_surface_detection(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)

        x0 = np.linspace(-100*um, 100*um, 512)
        z0 = np.linspace(-5*um, 200*um, 512)
        wavelength = 5*um
        u0 = Scalar_source_X(x=x0, wavelength=wavelength)
        u0.plane_wave(theta=0*degrees)
        u1 = Scalar_mask_XZ(x=x0, z=z0, wavelength=wavelength)
        u1.incident_field(u0)
        u1.biprism(r0=(0*um, 0*um),
                   length=200*um,
                   height=50*um,
                   refractive_index=1.5,
                   angle=0)

        u1.draw_refractive_index(draw_borders=True, scale='equal')
        u1.BPM()
        u1.draw(logarithm=True,
                normalize='maximum',
                draw_borders=True,
                scale='equal')
        u1.draw(kind='phase', draw_borders=True, scale='equal')

        u1.save_data(filename=filename + '.npz', add_name='')
        save_figure_test(newpath, func_name, add_name='')
        assert True

    def test_cut_resample(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)
        """
        generates a field and I cut_resample it
        """
        x0 = np.linspace(-150*um, 150*um, 256)
        z0 = np.linspace(-50*um*um, 300*um, 256)
        wavelength = 5*um
        u0 = Scalar_source_X(x=x0, wavelength=wavelength)
        u0.plane_wave(A=1, theta=0*degrees)

        u1 = Scalar_mask_XZ(x=x0, z=z0, wavelength=wavelength, n_background=4)
        u1.incident_field(u0)
        u1.slit(r0=(0*um, 10*um),
                aperture=40*um,
                depth=10*um,
                refractive_index=1,
                refractive_index_center='',
                angle=0*degrees)

        u1.BPM(verbose=False)
        u1.draw(kind='intensity', draw_borders=True)

        u1.save_data(filename=filename + '.npz', add_name='_before')
        save_figure_test(newpath, func_name, add_name='_before')

        u1.draw_refractive_index(draw_borders=True)
        u1.save_data(filename=filename + '.npz', add_name='_after')
        save_figure_test(newpath, func_name, add_name='_n_before')

        u1.cut_resample(x_limits=(-75, 75),
                        z_limits=(0, 30),
                        num_points=(512, 512),
                        new_field=False)
        u1.draw(kind='intensity', draw_borders=True)

        u1.save_data(filename=filename + '.npz', add_name='_after')
        save_figure_test(newpath, func_name, add_name='_after')

        u1.draw_refractive_index(draw_borders=True)

        u1.save_data(filename=filename + '.npz', add_name='_after')
        save_figure_test(newpath, func_name, add_name='_n_after')

        assert True

    def test_incident_field_1(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)

        x0 = np.linspace(-60*um, 60*um, 512)
        z0 = np.linspace(0*um, 100*um, 512)
        wavelength = 5*um
        u0 = Scalar_source_X(x=x0, wavelength=wavelength)
        u0.gauss_beam(A=1,
                      x0=0*um,
                      z0=0*um,
                      w0=10*um,
                      theta=0*degrees)

        t0 = Scalar_source_X(x=x0, wavelength=wavelength)
        t0.gauss_beam(A=1,
                      x0=40*um,
                      z0=0*um,
                      w0=10*um,
                      theta=-45.*degrees)

        t1 = Scalar_source_X(x=x0, wavelength=wavelength)
        t1.gauss_beam(A=1,
                      x0=-40*um,
                      z0=0*um,
                      w0=10*um,
                      theta=45.*degrees)

        u1 = Scalar_mask_XZ(x=x0, z=z0, wavelength=wavelength)
        u1.incident_field(u0)
        u1.BPM()
        u1.draw(kind='intensity', logarithm=True)

        u1.save_data(filename=filename + '.npz', add_name='')
        save_figure_test(newpath, func_name, add_name='')
        assert True

    def test_incident_field_n(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)

        x0 = np.linspace(-60*um, 60*um, 512)
        z0 = np.linspace(0*um, 100*um, 512)
        wavelength = 5*um
        u0 = Scalar_source_X(x=x0, wavelength=wavelength)
        u0.gauss_beam(A=1,
                      x0=0*um,
                      z0=0*um,
                      w0=10*um,
                      theta=0*degrees)

        t0 = Scalar_source_X(x=x0, wavelength=wavelength)
        t0.gauss_beam(A=1,
                      x0=40*um,
                      z0=0*um,
                      w0=10*um,
                      theta=-45.*degrees)

        t1 = Scalar_source_X(x=x0, wavelength=wavelength)
        t1.gauss_beam(A=1,
                      x0=-40*um,
                      z0=0*um,
                      w0=10*um,
                      theta=45.*degrees)

        u1 = Scalar_mask_XZ(x=x0, z=z0, wavelength=wavelength)
        u1.incident_field(u0, z0=10*um)
        u1.incident_field(t0, z0=25*um)
        u1.incident_field(t1, z0=25*um)
        u1.draw(kind='intensity', logarithm=True)

        u1.save_data(filename=filename + '.npz', add_name='_0')
        save_figure_test(newpath, func_name, add_name='_0')

        u1.BPM()
        u1.draw(kind='intensity', logarithm=True)

        u1.save_data(filename=filename + '.npz', add_name='_prop')
        save_figure_test(newpath, func_name, add_name='_prop')
        assert True

    def test_final_field(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)

        x0 = np.linspace(-150*um, 150*um, 512)
        z0 = np.linspace(0*um, 500*um, 512)
        wavelength = 5*um
        u0 = Scalar_source_X(x=x0, wavelength=wavelength)
        u0.plane_wave(A=1, theta=0*degrees)

        u1 = Scalar_mask_XZ(x=x0, z=z0, wavelength=wavelength)
        u1.incident_field(u0)
        u1.slit(r0=(10*um, 0*um),
                aperture=200*um,
                depth=15*um,
                refractive_index=1 + 5j,
                refractive_index_center='',
                angle=0*degrees)
        u1.biprism(r0=(10*um, 0*um),
                   length=200*um,
                   height=15*um,
                   refractive_index=1.5,
                   angle=0*degrees)
        u1.BPM()
        u1.draw(kind='intensity', draw_borders=True)
        u1.save_data(filename=filename + '.npz', add_name='_field')
        save_figure_test(newpath, func_name, add_name='_field')

        u_final = u1.final_field()
        u_final.draw()

        u_final.save_data(filename=filename + '.npz', add_name='_final')
        save_figure_test(newpath, func_name, add_name='_final')
        assert True

    def test_RS(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)

        wavelength = .5*um
        x0 = np.linspace(-200*um, 200*um, 512)
        z0 = np.linspace(500*um, 3*mm, 512)

        t1 = Scalar_mask_X(x0, wavelength)
        t1.lens(x0=0*um, radius=100*um, focal=2*mm)
        # t1.draw(kind='phase')

        f1 = Scalar_source_X(x0, wavelength)
        f1.plane_wave(A=1, theta=0*degrees)
        # f1.draw()

        u1 = Scalar_field_XZ(x=x0, z=z0, wavelength=wavelength)
        u1.incident_field(f1 * t1)
        u1.RS()
        u1.draw(logarithm=1, normalize='intensity', draw_borders=False)
        x_f, z_f = u1.search_focus()
        text = "positions focus: x={:2.2f} um, z={:2.2f} mm".format(
            x_f, z_f / mm)
        plt.title(text)

        u1.save_data(filename=filename + '.npz', add_name='')
        save_figure_test(newpath, func_name, add_name='')
        assert True

    def test_BPM(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)

        x0 = np.linspace(-100*um, 100*um, 512)
        z0 = np.linspace(0*um, 200*um, 512)
        wavelength = 5*um
        u0 = Scalar_source_X(x=x0, wavelength=wavelength)
        u0.gauss_beam(A=1,
                      x0=0*um,
                      z0=0*um,
                      w0=10*um,
                      theta=0*degrees)
        u0.plane_wave(A=1, theta=0*degrees)
        u1 = Scalar_mask_XZ(x=x0, z=z0, wavelength=wavelength)
        u1.incident_field(u0)
        u1.square(r0=(0*um, 100*um),
                     size=(150*um, 50*um),
                     angle=45*degrees,
                     refractive_index=1.5 - 0 * .00025j)

        u1.draw_refractive_index()
        u1.BPM(verbose=False)
        u1.draw(logarithm=True, normalize='maximum', draw_borders=True)

        u1.save_data(filename=filename + '.npz', add_name='')
        save_figure_test(newpath, func_name, add_name='')
        assert True

    def test_WPM(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)

        x0 = np.linspace(-100*um, 100*um, 512)
        z0 = np.linspace(0*um, 200*um, 512)
        wavelength = 5*um
        u0 = Scalar_source_X(x=x0, wavelength=wavelength)
        u0.gauss_beam(A=1,
                      x0=0*um,
                      z0=0*um,
                      w0=10*um,
                      theta=0*degrees)
        u0.plane_wave(A=1, theta=0*degrees)
        u1 = Scalar_mask_XZ(x=x0, z=z0, wavelength=wavelength)
        u1.incident_field(u0)
        u1.square(r0=(0*um, 100*um),
                     size=(150*um, 50*um),
                     angle=45*degrees,
                     refractive_index=1.5 - 0 * .00025j)

        u1.draw_refractive_index()
        u1.WPM(verbose=False)
        u1.draw(logarithm=True, normalize='maximum', draw_borders=True)

        u1.save_data(filename=filename + '.npz', add_name='')
        save_figure_test(newpath, func_name, add_name='')
        assert True

    def test_compare_methods(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)

        length = 200*um
        wavelength = 5*um
        x0 = np.linspace(-length/2, length/2, 512)
        z0 = np.linspace(2*um, 500*um, 512)

        # source
        f1 = Scalar_source_X(x0, wavelength)
        f1.gauss_beam(A=1, x0=0*um, z0=0*um, w0=10*um, theta=0*degrees)
        f1.draw(kind='intensity')

        # RS
        u1 = Scalar_field_XZ(x=x0, z=z0, wavelength=wavelength)
        u1.incident_field(f1)
        u1.RS()
        u1.draw(kind='intensity',
                logarithm=1,
                normalize='intensity',
                draw_borders=False)
        save_figure_test(newpath, func_name, add_name='_RS')

        u1.save_data(filename=filename + '.npz', add_name='_RS')
        u_RS = u1.u

        u1.clear_field()
        u1.incident_field(f1)
        u1.BPM(verbose=False)
        u1.draw(kind='intensity',
                logarithm=1,
                normalize='intensity',
                draw_borders=False)
        save_figure_test(newpath, func_name, add_name='_BPM')

        u1.save_data(filename=filename + '.npz', add_name='_BPM')

        u_BPM = u1.u

        diferencias = np.abs(u_RS)**2 - np.abs(u_BPM)**2
        u1.u = diferencias
        u1.draw(kind='intensity', logarithm=0., normalize=False)
        u1.save_data(filename=filename + '.npz', add_name='_diff')
        save_figure_test(newpath, func_name, add_name='_diff')
        assert True

    
    @pytest.mark.skip(reason="test_draw_profiles: mp4 generation not working at github")
    def test_draw_profiles(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)

        length = 200*um
        wavelength = 5*um
        period = 25*um
        z_talbot = 2 * period**2 / wavelength
        x0 = np.linspace(-length/2, length/2, 512)
        z0 = np.linspace(25*um, 1 * z_talbot, 64)

        u0 = Scalar_source_X(x0, wavelength)
        u0.gauss_beam(A=1,
                      x0=0*um,
                      z0=-100*um,
                      w0=100*um,
                      theta=0*degrees)

        t1 = Scalar_mask_X(x0, wavelength)
        t1.ronchi_grating(period=25*um, x0=0*um, fill_factor=0.5)

        u1 = Scalar_field_XZ(x=x0, z=z0, wavelength=wavelength)
        u1.incident_field(t1 * u0)
        u1.RS()

        u1.draw(kind='intensity',
                logarithm=0.,
                normalize='maximum',
                draw_borders=True,
                filename='')
        save_figure_test(newpath, func_name, add_name='_int')

        filename = '{}{}{}.{}'.format(newpath, func_name, '_video', 'mp4')

        seconds = 1
        u1.video(kind='intensity',
                 logarithm=True,
                 normalize=False,
                 time_video=10 * seconds,
                 frames_reduction=5,
                 filename=filename,
                 dpi=100)

        assert True

    def test_BPM_profile_automatico(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)

        x0 = np.linspace(-25*um, 25*um, 512)
        z0 = np.linspace(0*um, 75*um, 128)
        wavelength = 5*um
        u0 = Scalar_source_X(x=x0, wavelength=wavelength)
        u0.plane_wave(A=1, theta=0*degrees)
        u1 = Scalar_mask_XZ(x=x0, z=z0, wavelength=wavelength)
        u1.incident_field(u0)
        u1.mask_field(size_edge=5*um)
        u1.cylinder(r0=(0*um, 20*um),
                  radius=(20*um, 20*um),
                  refractive_index=1.5)
        u1.BPM(verbose=False)
        u1.draw_profiles_interactive(kind='intensity',
                                     logarithm=True,
                                     normalize='maximum')

        u1.save_data(filename=filename + '.npz', add_name='')
        save_figure_test(newpath, func_name, add_name='')
        assert True

    def test_BPM_profile_longitudinal(self):
        func_name = sys._getframe().f_code.co_name
        # filename = '{}{}'.format(newpath, func_name)

        x0 = np.linspace(-25*um, 25*um, 512)
        z0 = np.linspace(0*um, 75*um, 512)
        wavelength = .5*um
        u0 = Scalar_source_X(x=x0, wavelength=wavelength)
        u0.plane_wave(A=1, theta=0*degrees)
        u1 = Scalar_mask_XZ(x=x0, z=z0, wavelength=wavelength)
        u1.incident_field(u0)
        u1.mask_field(size_edge=5*um)
        u1.cylinder(r0=(0*um, 0*um),
                  radius=(25*um, 25*um),
                  refractive_index=2)
        u1.BPM(verbose=False)
        u1.draw(logarithm=True, normalize='maximum', draw_borders=True)
        save_figure_test(newpath, func_name, add_name='')

        u1.profile_longitudinal(x0=0*um)
        save_figure_test(newpath, func_name, add_name='_prof')
        assert True

    def test_BPM_profile_transversal(self):
        func_name = sys._getframe().f_code.co_name
        # filename = '{}{}'.format(newpath, func_name)

        x0 = np.linspace(-25*um, 25*um, 512)
        z0 = np.linspace(0*um, 75*um, 512)
        wavelength = .5*um
        u0 = Scalar_source_X(x=x0, wavelength=wavelength)
        u0.plane_wave(A=1, theta=0*degrees)
        u1 = Scalar_mask_XZ(x=x0, z=z0, wavelength=wavelength)
        u1.incident_field(u0)
        u1.mask_field(size_edge=5*um)
        u1.cylinder(r0=(0*um, 0*um),
                  radius=(25*um, 25*um),
                  refractive_index=2)
        u1.BPM(verbose=False)
        u1.draw(logarithm=True, normalize='maximum', draw_borders=True)
        save_figure_test(newpath, func_name, add_name='')

        u1.profile_transversal(z0=46*um)
        save_figure_test(newpath, func_name, add_name='_prof')
        assert True

    def test_find_focus(self):
        func_name = sys._getframe().f_code.co_name
        # filename = '{}{}'.format(newpath, func_name)

        # dielectrico = 2
        # radius_sphere = 8*um
        # x0 = np.linspace(-5 * radius_sphere*um, 5 * radius_sphere*um, 512)
        # z0 = np.linspace(0*um, 4 * radius_sphere, 512)
        # wavelength = 5*um
        # u0 = Scalar_source_X(x=x0, wavelength=wavelength)
        # u0.gauss_beam(
        #     A=1,
        #     x0=0*um,
        #     z0=5000*um,
        #     w0=radius_sphere/2,
        #     theta=0*degrees)
        # u1 = Scalar_mask_XZ(x=x0, z=z0, wavelength=wavelength)
        # u1.incident_field(u0)
        # u1.mask_field(size_edge=5*um)
        #
        # u1.cylinder(
        #     r0=(0, 2 * radius_sphere),
        #     radius=(radius_sphere, radius_sphere),
        #     refractive_index=dielectrico)
        # u1.draw_refractive_index()
        # u1.BPM(verbose=False)
        u1 = u_focus
        u1.draw(logarithm=0., normalize=False, draw_borders=True)
        save_figure_test(newpath, func_name, add_name='')

        x_max, z_max = u1.search_focus()

        u1.profile_transversal(z0=z_max)
        save_figure_test(newpath, func_name, add_name='_trans')
        u1.profile_longitudinal(x0=x_max)
        save_figure_test(newpath, func_name, add_name='_long')
        assert True

    def test_BPM_inverse(self):
        func_name = sys._getframe().f_code.co_name
        # filename = '{}{}'.format(newpath, func_name)

        x0 = np.linspace(-25*um, 25*um, 512)
        z0 = np.linspace(0*um, 75*um, 512)
        wavelength = 5*um
        u0 = Scalar_source_X(x=x0, wavelength=wavelength)
        u0.gauss_beam(A=1, x0=0*um, z0=0*um, w0=10*um, theta=0*degrees)

        u1 = Scalar_mask_XZ(x=x0, z=z0, wavelength=wavelength)
        u1.incident_field(u0)

        u1.square(r0=(0*um, 45*um),
                     size=(25*um, 25*um),
                     angle=0*degrees,
                     refractive_index=1.5)

        u1.draw_refractive_index(draw_borders=True, min_incr=0.001)
        u1.BPM(verbose=False)
        u1.draw(logarithm=True,
                normalize='maximum',
                draw_borders=True,
                min_incr=0.001)

        save_figure_test(newpath, func_name, add_name='_direct')

        u2 = u1.BPM_inverse()
        u2.draw(logarithm=True,
                normalize='maximum',
                draw_borders=True,
                min_incr=0.001)
        save_figure_test(newpath, func_name, add_name='_inverse')

        differences = u1 - u2
        differences.u = np.abs(u1.u) - np.abs(u2.u)
        differences.draw(logarithm=True,
                         normalize='maximum',
                         draw_borders=True,
                         min_incr=0.001)
        save_figure_test(newpath, func_name, add_name='_diff')

        assert True

    def test_BPM_backpropagation(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)
        """
        here we do two experiments
        1. propagate and after backpropagate the last field
        2. backpropagate u0=1
        """
        x0 = np.linspace(-25*um, 25*um, 512)
        z0 = np.linspace(0*um, 75*um, 512)
        wavelength = 5*um
        u0 = Scalar_source_X(x=x0, wavelength=wavelength)
        u0.plane_wave(theta=20*degrees)
        u0.gauss_beam(A=1,
                      x0=-15*um,
                      z0=0*um,
                      w0=10*um,
                      theta=0*degrees)

        u1 = Scalar_mask_XZ(x=x0, z=z0, wavelength=wavelength)
        u1.incident_field(u0)

        u1.square(r0=(0*um, 45*um),
                     size=(25*um, 25*um),
                     angle=0*degrees,
                     refractive_index=4)

        u1.draw_refractive_index(draw_borders=True, min_incr=0.001)
        u1.BPM(verbose=False)
        u1.draw(logarithm=True,
                normalize='maximum',
                draw_borders=True,
                min_incr=0.001)
        save_figure_test(newpath, func_name, add_name='_for')

        # Hago la inverse y drawing los resultados
        u1.u[:, -1] = 1
        u2 = u1.BPM_back_propagation()
        # u2.draw_refractive_index(draw_borders=True, min_incr=0.001)
        u2.draw(logarithm=True,
                normalize='maximum',
                draw_borders=True,
                min_incr=0.001)

        u2.save_data(filename=filename + '.npz', add_name='')
        save_figure_test(newpath, func_name, add_name='_back')
        assert True

    def test_BPM_backpropagation_2(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)
        """
        we place a field at a certain position and after we backpropagate
        """
        x0 = np.linspace(-25*um, 25*um, 512)
        z0 = np.linspace(0*um, 100*um, 512)
        wavelength = 5*um
        u0 = Scalar_source_X(x=x0, wavelength=wavelength)
        u0.plane_wave(theta=20*degrees)
        u0.gauss_beam(A=1,
                      x0=-5*um,
                      z0=0*um,
                      w0=5*um,
                      theta=15.*degrees)

        u1 = Scalar_mask_XZ(x=x0, z=z0, wavelength=wavelength)

        u1.square(r0=(0*um, 45*um),
                     size=(25*um, 25*um),
                     angle=0*degrees,
                     refractive_index=2)

        u1.draw_refractive_index(draw_borders=True, min_incr=0.001)
        u1.incident_field(u0, z0=80*um)

        # Hago la inverse y drawing los resultados
        u2 = u1.BPM_back_propagation()
        # u2.draw_incident_field(
        #     kind='intensity', logarithm=0., normalize=False, filename: str = '')
        u2.draw_refractive_index(draw_borders=True, min_incr=0.001)
        u2.draw(logarithm=True,
                normalize='maximum',
                draw_borders=True,
                min_incr=0.001)

        u2.save_data(filename=filename + '.npz', add_name='')
        save_figure_test(newpath, func_name, add_name='')
        assert True

    def test_BPM_n_background(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)

        x0 = np.linspace(-60*um, 60*um, 512)
        z0 = np.linspace(0*um, 120*um, 512)
        wavelength = 4*um

        radius_sphere = 30*um

        u0 = Scalar_source_X(x=x0, wavelength=wavelength)
        u0.gauss_beam(A=1,
                      x0=0*um,
                      z0=60*um,
                      w0=25*um,
                      theta=0*degrees)

        u1 = Scalar_mask_XZ(x=x0, z=z0, wavelength=wavelength, n_background=4)
        u1.incident_field(u0)
        u1.mask_field(size_edge=5*um)

        u1.cylinder(r0=(0, 40*um),
                  radius=(radius_sphere, radius_sphere),
                  refractive_index=1)

        u1.smooth_refractive_index(type_filter=3,
                                   pixels_filtering=2,
                                   max_diff_filter=0.01,
                                   draw_check=False)
        save_figure_test(newpath, func_name, add_name='_n_diff')

        u1.BPM(verbose=False)
        u1.draw_refractive_index(scale='scaled')
        save_figure_test(newpath, func_name, add_name='_n')

        u1.draw(kind='intensity',
                logarithm=True,
                normalize='maximum',
                draw_borders=True,
                scale='scaled')

        u1.save_data(filename=filename + '.npz', add_name='')
        save_figure_test(newpath, func_name, add_name='')
        assert True

    def test_RS_polychromatic(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)

        wavelengths = np.linspace(0.4, 0.8, 5)
        w_central = 0.6
        Dw = 0.15*um
        spectrum = np.exp(-(wavelengths - w_central)**2 / (2 * Dw**2))
        initial_field = _func_polychromatic_RS_(wavelengths[0])
        x0 = initial_field.x

        z0 = np.linspace(250*um, 1000*um, 512)

        u1 = Scalar_mask_XZ(x0, z0, wavelengths[0], n_background=1)
        initial_field = _func_polychromatic_RS_(wavelengths[0])
        u1.incident_field(initial_field)
        u1.RS()
        u1.draw(logarithm=True, normalize='intensity')

        u1 = Scalar_mask_XZ(x0, z0, wavelengths[0], n_background=1)
        u_poly = u1.RS_polychromatic(_func_polychromatic_RS_,
                                     wavelengths,
                                     spectrum=spectrum,
                                     verbose=False,
                                     num_processors=1)

        u_poly.draw(logarithm=True, normalize='intensity', draw_borders=True)

        save_figure_test(newpath, func_name, add_name='_int')

        u_poly.save_data(filename=filename + '.npz', add_name='')
        assert True

    def test_BPM_polychromatic(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)

        wavelengths = np.linspace(0.4, 0.8, 11)
        spectrum = ''
        initial_field = _func_polychromatic_BPM_(wavelengths[0])
        z0 = initial_field.z
        x0 = initial_field.x

        u1 = Scalar_mask_XZ(x0, z0, wavelengths[0], n_background=1)
        u_poly = u1.BPM_polychromatic(_func_polychromatic_BPM_,
                                      wavelengths,
                                      spectrum,
                                      verbose=True,
                                      num_processors=1)
        u_poly.draw(logarithm=True, normalize='intensity', draw_borders=True)
        save_figure_test(newpath, func_name, add_name='_int')

        u_poly.save_data(filename=filename + '.npz', add_name='')
        assert True
