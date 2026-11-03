# !/usr/bin/env python3


import datetime
import os
import sys
import time
import pytest

from diffractio import degrees, eps, mm, no_date, np, plt, um
from diffractio.scalar_fields_XY import Scalar_field_XY
from diffractio.scalar_masks_XY import Scalar_mask_XY
from diffractio.scalar_sources_XY import Scalar_source_XY
from diffractio.utils_drawing import draw_several_fields
from diffractio.utils_tests import comparison, save_figure_test

try:
    import screeninfo
except:
    print("screeninfo not imported.")

if no_date is True:
    date = '0'
else:
    now = datetime.datetime.now()
    date = now.strftime("%Y-%m-%d_%H")

path_base = "tests_results"
path_class = "scalar_fields_XY"
newpath = "{}_{}/{}/".format(path_base, date, path_class)

if not os.path.exists(newpath):
    os.makedirs(newpath)


def gauss_beam_test():
    x0 = np.linspace(-100, 100, 256)
    y0 = np.linspace(-100, 100, 256)
    wavelength = 0.6328

    field = Scalar_source_XY(x0, y0, wavelength)
    field.gauss_beam(w0=(50*um, 50*um),
                     r0=(0*um, 0*um),
                     z0=0,
                     theta=0*degrees,
                     phi=0*degrees)

    return field


def gauss_beam_narrow_test():
    x0 = np.linspace(-100, 100, 256)
    y0 = np.linspace(-100, 100, 256)
    wavelength = 0.6328

    field = Scalar_source_XY(x0, y0, wavelength)
    field.gauss_beam(w0=(5*um, 5*um),
                     r0=(0*um, 0*um),
                     theta=0*degrees,
                     z0=0,
                     phi=0*degrees)

    return field


def plane_wave_test():
    x0 = np.linspace(-100, 100, 256)
    y0 = np.linspace(-100, 100, 256)
    wavelength = 0.6328

    field = Scalar_source_XY(x0, y0, wavelength)
    field.plane_wave(A=1, theta=0*degrees, phi=0*degrees)

    return field


def square_test():
    x0 = np.linspace(-100, 100, 256)
    y0 = np.linspace(-100, 100, 256)
    wavelength = 0.6328

    t = Scalar_mask_XY(x=x0, y=y0, wavelength=wavelength)
    t.square(r0=(0*um, 0*um), size=(100*um, 100*um), angle=0*degrees)

    return t


field_gauss = gauss_beam_test()
field_gauss_narrow = gauss_beam_narrow_test()
square = square_test()
plane_wave = plane_wave_test()


class Test_Scalar_fields_XY():

    def test_add(self):
        func_name = sys._getframe().f_code.co_name
        # filename = '{}{}.npz'.format(newpath, func_name)

        t1 = field_gauss
        t2 = field_gauss

        t1.draw(kind='amplitude', has_colorbar='horizontal')
        save_figure_test(newpath, func_name, add_name='_1')

        t = t1 + t2

        t.draw(kind='amplitude', has_colorbar='horizontal')

        proposal = 2 * t1.u
        solution = t.u
        save_figure_test(newpath, func_name, add_name='_2')
        assert comparison(proposal, solution, eps)

    def test_sub(self):
        func_name = sys._getframe().f_code.co_name
        # filename = '{}{}.npz'.format(newpath, func_name)

        t1 = field_gauss
        t2 = field_gauss

        t1.draw(kind='amplitude', has_colorbar='horizontal')
        save_figure_test(newpath, func_name, add_name='_1')

        t = t1 - t2

        t.draw(kind='amplitude', has_colorbar='horizontal')

        proposal = 0 * t1.u
        solution = t.u
        save_figure_test(newpath, func_name, add_name='_2')
        assert comparison(proposal, solution, eps)

    def test_mult(self):
        func_name = sys._getframe().f_code.co_name
        # filename = '{}{}.npz'.format(newpath, func_name)

        u0 = field_gauss
        t0 = square

        u1 = u0 * t0

        u1.draw(kind='intensity', has_colorbar='horizontal')

        save_figure_test(newpath, func_name, add_name='')
        assert True

    def test_rotate(self):
        func_name = sys._getframe().f_code.co_name
        # filename = '{}{}.npz'.format(newpath, func_name)

        t0 = square
        t0.draw(kind='intensity', has_colorbar='horizontal')
        save_figure_test(newpath, func_name, add_name='_00')

        t0.rotate(angle=45*degrees, position=(-50, -50))

        t0.draw(kind='intensity', has_colorbar='horizontal')

        save_figure_test(newpath, func_name, add_name='_45')
        assert True

    def test_clear_field(self):
        func_name = sys._getframe().f_code.co_name
        # filename = '{}{}.npz'.format(newpath, func_name)

        u0 = field_gauss
        proposal = 0 * u0.u

        u0.clear_field()
        solution = u0.u

        u0.draw(kind='intensity', has_colorbar='horizontal')
        plt.clim(0, 1)

        save_figure_test(newpath, func_name, add_name='')
        assert comparison(proposal, solution, eps)

    def test_save_load(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)

        x = np.linspace(-100, 100, 256)
        y = np.linspace(-100, 100, 256)
        wavelength = 0.6328

        t1 = Scalar_source_XY(x, y, wavelength)
        t1.gauss_beam(w0=(50*um, 50*um),
                      r0=(0*um, 0*um),
                      z0=0,
                      theta=0*degrees,
                      phi=0*degrees)

        t1.draw()
        save_figure_test(newpath, func_name, add_name='_saved')
        t1.save_data(filename=filename + '.npz')

        time.sleep(1)
        t2 = Scalar_field_XY(x, y, wavelength)
        t2.load_data(filename=filename + '.npz')
        t2.draw()
        save_figure_test(newpath, func_name, add_name='_loaded')
        assert True

    def test_cut_resample(self):
        func_name = sys._getframe().f_code.co_name
        # filename = '{}{}.npz'.format(newpath, func_name)

        u0 = field_gauss_narrow

        u0.draw(kind='intensity')
        save_figure_test(newpath, func_name, add_name='_0before')
        u0.cut_resample(x_limits=(-10*um, 10*um),
                        y_limits=(-10*um, 10*um),
                        num_points=(512, 512),
                        new_field=False)
        u0.draw(kind='intensity')
        save_figure_test(newpath, func_name, add_name='_1after')
        assert True

    def test_fft(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}.npz'.format(newpath, func_name)

        field = field_gauss_narrow

        fieldfft1 = field.fft(z=0.1*mm,
                              remove0=False,
                              shift=True,
                              matrix=False,
                              new_field=True)
        fieldfft1.draw(kind='intensity')

        fieldfft1.save_data(filename=filename + '.npz', add_name='')
        save_figure_test(newpath, func_name, add_name='')
        assert True

    def test_RS(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}.npz'.format(newpath, func_name)

        length = 160*um
        num_data = 256
        x0 = np.linspace(-length/2, length/2, num_data)
        y0 = np.linspace(-length / 1, length / 1, 2 * num_data)
        wavelength0 = 0.6238*um

        u1 = Scalar_source_XY(x=x0, y=y0, wavelength=wavelength0)
        u1.plane_wave(A=1, theta=0*degrees, phi=0*degrees)

        t1 = Scalar_mask_XY(x=x0, y=y0, wavelength=wavelength0)
        t1.square(r0=(0*um, 0*um),
                  size=(40*um, 40*um),
                  angle=0*degrees)

        u2 = u1 * t1
        u2.draw()
        u3 = u2.RS(z=4*mm, new_field=True)
        u3.draw(kind='field')
        u4 = u3.RS(z=-2*mm, new_field=True)
        u4.draw(kind='field')

        u4.save_data(filename=filename + '.npz', add_name='')
        save_figure_test(newpath, func_name, add_name='')
        assert True

    def test_RS_amplification(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}.npz'.format(newpath, func_name)

        num_pixels = 256
        lengthx = 200*um
        lengthy = 200*um
        wavelength = 0.6238*um

        x0 = np.linspace(-lengthx/2, lengthx/2, num_pixels)
        y0 = np.linspace(-lengthy/2, lengthy/2, num_pixels)

        u1 = Scalar_source_XY(x=x0, y=y0, wavelength=wavelength)
        u1.gauss_beam(r0=(0., 0.), w0=100*um, z0=0, A=1, theta=0, phi=0)
        u2 = u1.RS(amplification=(2, 3), z=1*mm)
        u2.draw('field')

        u2.save_data(filename=filename + '.npz', add_name='')
        save_figure_test(newpath, func_name, add_name='')
        assert True

    def test_profile(self):
        func_name = sys._getframe().f_code.co_name
        # filename = '{}{}.npz'.format(newpath, func_name)

        field = gauss_beam_test()

        p1 = [0, 0]
        p2 = [field.x[0], field.y[0]]

        h, z_profile, point1, point2 = field.profile(point1=p1,
                                                     point2=p2,
                                                     kind='intensity',
                                                     order=1,
                                                     has_draw=False)

        plt.figure()
        plt.plot(h, z_profile, 'k', linewidth=2)
        plt.xlabel('h (profile)')
        plt.ylabel('intensity')
        save_figure_test(newpath, func_name, add_name='_prof')

        field.draw()
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r', linewidth=2)

        save_figure_test(newpath, func_name, add_name='')
        assert True

    @pytest.mark.skip(reason="test_profile: manual click")
    def test_profile_manual(self):
        func_name = sys._getframe().f_code.co_name
        # filename = '{}{}.npz'.format(newpath, func_name)

        field = gauss_beam_test()

        h, z_profile, point1, point2 = field.draw_profile(point1='',
                                                          point2='',
                                                          kind='intensity',
                                                          order=1)
        field.draw()
        plt.plot([point1[0], point2[0]], [point1[1], point2[1]],
                 'r',
                 linewidth=2)

        save_figure_test(newpath, func_name, add_name='')

        field.draw_profile(point1=point1,
                           point2=point2,
                           kind='intensity',
                           order=1)

        save_figure_test(newpath, func_name, add_name='_prof')
        assert True

    def test_search_focus(self):
        # func_name = sys._getframe().f_code.co_name
        # filename = '{}{}.npz'.format(newpath, func_name)

        u = gauss_beam_narrow_test()
        solution = np.array(u.search_focus())

        proposal = np.array([0, 0])
        assert comparison(proposal, solution, 2*um)

    def test_MTF(self):

        func_name = sys._getframe().f_code.co_name
        # filename = '{}{}.npz'.format(newpath, func_name)

        u = gauss_beam_narrow_test()
        u.draw()
        save_figure_test(newpath, func_name, add_name='_field')

        fx = np.linspace(-250,250,1000)
        fy = np.linspace(-250,250,1000)

        u.MTF(fx=fx, fy=fy, has_draw=True)        
        save_figure_test(newpath, func_name, add_name='_mtf')

    def test_average_intensity(self):
        # func_name = sys._getframe().f_code.co_name
        # filename = '{}{}.npz'.format(newpath, func_name)

        u = gauss_beam_narrow_test()
        inten1 = u.average_intensity(verbose=True)
        print(inten1)
        assert True

    @pytest.mark.skip(reason="test_send_image_screen: manual click")
    def test_send_image_screen(self):
        # func_name = sys._getframe().f_code.co_name
        # filename = '{}{}.npz'.format(newpath, func_name)

        screens = screeninfo.get_monitors()
        id_screen = 0  # 1
        pixel_size_x = 20*um
        pixel_size_y = 20*um
        num_pixels_x = screens[id_screen].width
        num_pixels_y = screens[id_screen].height
        # Definition of input parameters
        x0 = np.linspace(-pixel_size_x * num_pixels_x/2,
                         pixel_size_x * num_pixels_x/2, num_pixels_x)
        y0 = np.linspace(-pixel_size_y * num_pixels_y/2,
                         pixel_size_y * num_pixels_y/2, num_pixels_y)
        wavelength = 0.6238*um

        t = Scalar_mask_XY(x=x0, y=y0, wavelength=wavelength)
        t.circle(r0=(0*um, 0*um), radius=(500*um, 500*um), angle=0*degrees)
        t.lens(r0=(0*um, 0*um),
               focal=(1000*mm, 1000*mm),
               radius=(10*mm, 10*mm),
               angle=0*degrees)
        t.send_image_screen(id_screen, kind='phase')
        assert True

    def test_get_amplitude(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}.npz'.format(newpath, func_name)

        length = 250*um
        x0 = np.linspace(-length/2, length/2, 256)
        y0 = np.linspace(-length/2, length/2, 256)
        wavelength = 0.6328*um

        # field total
        field = Scalar_source_XY(x=x0, y=y0, wavelength=wavelength)
        field.spherical_wave(r0=(0*um, 0*um), z0=-1*mm)
        field.draw(kind='field')

        real_field = Scalar_source_XY(x=x0, y=y0, wavelength=wavelength)
        real_field.u = field.get_amplitude(matrix=True)
        real_field.draw(kind='field')

        # Generando fields news
        real_fieldNuevo = field.get_amplitude(new_field=True)
        real_fieldNuevo.draw(kind='field')

        # en el mismo field de entrada
        field.get_amplitude()
        field.draw(kind='field')

        field.save_data(filename=filename + '.npz', add_name='')
        save_figure_test(newpath, func_name, add_name='')
        assert True

    def test_get_phase(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}.npz'.format(newpath, func_name)

        length = 250*um
        x0 = np.linspace(-length/2, length/2, 256)
        y0 = np.linspace(-length/2, length/2, 256)
        wavelength = 0.6328*um

        # field total
        field = Scalar_source_XY(x=x0, y=y0, wavelength=wavelength)
        field.spherical_wave(r0=(0*um, 0*um), z0=-1*mm)
        field.draw(kind='field')

        fieldImag = Scalar_source_XY(x=x0, y=y0, wavelength=wavelength)
        fieldImag.u = field.get_phase(matrix=True)
        fieldImag.draw(kind='field')

        fieldImagNuevo = field.get_phase(new_field=True)
        fieldImagNuevo.draw(kind='field')

        # en el mismo field de entrada
        field.get_phase()
        field.draw(kind='field')
        # no se ve nada, porque se ha quitado la amplitude y la phase

        field.save_data(filename=filename + '.npz', add_name='')
        save_figure_test(newpath, func_name, add_name='')
        assert True

    def test_remove_phase(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}.npz'.format(newpath, func_name)

        length = 250*um
        x0 = np.linspace(-length/2, length/2, 256)
        y0 = np.linspace(-length/2, length/2, 256)
        wavelength = 0.6*um

        # field total
        field = Scalar_source_XY(x=x0, y=y0, wavelength=wavelength)
        field.spherical_wave(r0=(0*um, 0*um), z0=-1*mm)
        field.draw(kind='field')

        field.remove_phase(sign=False)
        field.draw(kind='field')

        field.save_data(filename=filename + '.npz', add_name='')
        save_figure_test(newpath, func_name, add_name='')

        solution = np.angle(field.u).sum()

        proposal = 0
        assert comparison(proposal, solution, eps)

        assert True

    def test_binarizeAmplitud(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}.npz'.format(newpath, func_name)

        length = 250*um
        x0 = np.linspace(-length/2, length/2, 256)
        y0 = np.linspace(-length/2, length/2, 256)
        wavelength = 0.6*um

        # field total
        field = Scalar_mask_XY(x=x0, y=y0, wavelength=wavelength)
        field.gray_scale(num_levels=255, level_min=0, level_max=1)
        field.draw(kind='intensity', normalize=None)

        field.binarize(kind="amplitude",
                       bin_level=0.5,
                       level0=0,
                       level1=1,
                       new_field=False,
                       matrix=False)
        field.draw(kind='intensity', normalize=None)

        field.save_data(filename=filename + '.npz', add_name='')
        save_figure_test(newpath, func_name, add_name='')
        assert True

    def test_binarizeFase(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}.npz'.format(newpath, func_name)

        length = 250*um
        wavelength = 0.6*um
        x0 = np.linspace(-length/2, length/2, 256)
        y0 = np.linspace(-length/2, length/2, 256)

        # field total
        field = Scalar_source_XY(x=x0, y=y0, wavelength=wavelength)
        field.spherical_wave(r0=(0*um, 0*um), z0=-1*mm)
        field.draw(kind='phase', normalize=None)

        field2 = field.binarize(kind="phase",
                                bin_level=None,
                                level0=None,
                                level1=None,
                                new_field=True,
                                matrix=False)
        field2.draw(kind='phase', normalize=None)

        field3 = field.binarize(kind="phase",
                                bin_level=0,
                                level0=-np.pi/2,
                                level1=np.pi/2,
                                new_field=True,
                                matrix=False)

        field3.save_data(filename=filename + '.npz', add_name='')
        save_figure_test(newpath, func_name, add_name='')
        assert True

    def test_discretize_1(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}.npz'.format(newpath, func_name)

        # amplitude
        length = 250*um
        x0 = np.linspace(-length/2, length/2, 256)
        y0 = np.linspace(-length/2, length/2, 256)

        # field total
        field = Scalar_mask_XY(x=x0, y=y0)
        field.gray_scale(num_levels=255, level_min=0, level_max=1)
        field.draw(kind='field')

        fieldAmplitud = field.discretize(kind='amplitude',
                                         num_levels=2,
                                         factor=1,
                                         new_field=True,
                                         matrix=False)
        fieldAmplitud.draw(kind='field')

        fieldAmplitud = field.discretize(kind='amplitude',
                                         num_levels=2,
                                         factor=.5,
                                         new_field=True,
                                         matrix=False)
        fieldAmplitud.draw(kind='field')

        fieldAmplitud = field.discretize(kind='amplitude',
                                         num_levels=3,
                                         factor=1,
                                         new_field=True,
                                         matrix=False)
        fieldAmplitud.draw(kind='field')

        fieldAmplitud = field.discretize(kind='amplitude',
                                         num_levels=4,
                                         factor=1,
                                         new_field=True,
                                         matrix=False)
        fieldAmplitud.draw(kind='field')

        fieldAmplitud = field.discretize(kind='amplitude',
                                         num_levels=4,
                                         factor=.5,
                                         new_field=True,
                                         matrix=False)
        fieldAmplitud.draw(kind='field')

        fieldAmplitud.save_data(filename=filename + '.npz', add_name='')
        save_figure_test(newpath, func_name, add_name='')
        assert True

    def test_discretize_2(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}.npz'.format(newpath, func_name)

        # phase
        from diffractio.scalar_masks_XY import Scalar_mask_XY
        length = 250*um
        wavelength = 0.6*um
        x0 = np.linspace(-length/2, length/2, 256)
        y0 = np.linspace(-length/2, length/2, 256)

        # field total
        field = Scalar_mask_XY(x=x0, y=y0, wavelength=wavelength)
        field.gray_scale(num_levels=255, level_min=0, level_max=1)
        field.set_phase(q=1, phase_min=-np.pi, phase_max=np.pi)
        field.draw(kind='field')

        fieldFase = field.discretize(kind='phase',
                                     num_levels=2,
                                     factor=1,
                                     new_field=True,
                                     matrix=False)
        fieldFase.draw(kind='field')

        fieldFase = field.discretize(kind='phase',
                                     num_levels=2,
                                     factor=.5,
                                     new_field=True,
                                     matrix=False)
        fieldFase.draw(kind='field')

        fieldFase = field.discretize(kind='phase',
                                     num_levels=4,
                                     factor=1,
                                     new_field=True,
                                     matrix=False)
        fieldFase.draw(kind='field')

        fieldFase = field.discretize(kind='phase',
                                     num_levels=4,
                                     factor=.5,
                                     new_field=True,
                                     matrix=False)
        fieldFase.draw(kind='field')

        fieldFase.save_data(filename=filename + '.npz', add_name='')
        save_figure_test(newpath, func_name, add_name='')
        assert True

    def test_discretize_3(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}.npz'.format(newpath, func_name)

        # lens
        length = 250*um
        wavelength = 0.6*um
        x0 = np.linspace(-length/2, length/2, 256)
        y0 = np.linspace(-length/2, length/2, 256)

        # field total
        field = Scalar_mask_XY(x=x0, y=y0, wavelength=wavelength)

        field.lens(r0=(0*um, 0*um),
                   radius=(150*um, 150*um),
                   focal=(2*mm, 2*mm),
                   angle=0*degrees)
        field.draw(kind='field')
        fieldFase = field.discretize(kind='phase',
                                     num_levels=2,
                                     factor=1,
                                     new_field=True,
                                     matrix=False)
        fieldFase.draw(kind='field')

        field.lens(r0=(0*um, 0*um),
                   radius=(150*um, 150*um),
                   focal=(2*mm, 2*mm),
                   angle=0*degrees)
        field.draw(kind='field')
        fieldFase = field.discretize(kind='phase',
                                     num_levels=2,
                                     factor=.5,
                                     new_field=True,
                                     matrix=False)
        fieldFase.draw(kind='field')

        fieldFase = field.discretize(kind='phase',
                                     num_levels=4,
                                     factor=1,
                                     new_field=True,
                                     matrix=False)
        fieldFase.draw(kind='field')

        fieldFase = field.discretize(kind='phase',
                                     num_levels=4,
                                     factor=.5,
                                     new_field=True,
                                     matrix=False)
        fieldFase.draw(kind='field')

        fieldFase.save_data(filename=filename + '.npz', add_name='')
        save_figure_test(newpath, func_name, add_name='')
        assert True

    def test_normalize(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}.npz'.format(newpath, func_name)

        wavelength = 0.6328*um
        length = 100*um
        x0 = np.linspace(-length/2, length/2, 256)
        y0 = np.linspace(-length/2, length/2, 256)

        field1 = Scalar_source_XY(x=x0, y=y0, wavelength=wavelength)
        field1.gauss_beam(r0=(0*um, 0*um), w0=(20*um, 20*um), z0=0, A=3)
        field1.normalize()
        field1.draw(kind='intensity', normalize=False)

        field1.save_data(filename=filename + '.npz', add_name='')
        save_figure_test(newpath, func_name, add_name='')
        assert True

    def test_draw_several_fields(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}.npz'.format(newpath, func_name)
        """Diversos test para verificar el functionamiento de la
         transformada de Fraunhofer"""
        length = 500*um
        x0 = np.linspace(-length/2, length/2, 256)
        y0 = np.linspace(-length/2, length/2, 256)
        wavelength = 0.6328*um

        field1 = Scalar_source_XY(x=x0, y=y0, wavelength=wavelength)
        field1.spherical_wave(r0=(0*um, 0*um), z0=-1*mm)

        field2 = Scalar_source_XY(x=x0, y=y0, wavelength=wavelength)
        field2.gauss_beam(w0=(50*um, 100*um),
                          r0=(0*um, 0*um),
                          z0=0,
                          theta=0*degrees,
                          phi=0*degrees)

        field3 = Scalar_source_XY(x=x0, y=y0, wavelength=wavelength)
        field3.laguerre_beam(A=1,
                             r0=(0*um, 0*um),
                             w0=10*um,
                             z=50*um,
                             z0=0,
                             n=1,
                             l=1)

        draw_several_fields(fields=(field1, field2, field3),
                            titles=('(a)', '(b)', '(c)'))

        field3.save_data(filename=filename + '.npz', add_name='')
        save_figure_test(newpath, func_name, add_name='')
        assert True
