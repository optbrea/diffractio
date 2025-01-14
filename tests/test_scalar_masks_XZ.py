# !/usr/bin/env python3

"""Tests for Scalar_mask_XZ"""
import datetime
import os
import sys

from diffractio import degrees, mm, no_date, np, um
from diffractio.scalar_masks_X import Scalar_mask_X
from diffractio.scalar_masks_XZ import Scalar_mask_XZ
from diffractio.utils_tests import comparison, save_figure_test
from numpy import loadtxt

if no_date is True:
    date = '0'
else:
    now = datetime.datetime.now()
    date = now.strftime("%Y-%m-%d_%H")

path_base = "tests_results"
path_class = "scalar_masks_XZ"

newpath = "{}_{}/{}/".format(path_base, date, path_class)

if not os.path.exists(newpath):
    os.makedirs(newpath)


class Test_Scalar_masks_XZ():

    def test_extrude_mask_z_n_var(self):
        """
        Here the refractive index is a function of positions z
        """
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)

        x0 = np.linspace(-100*um, 100*um, 256)
        z0 = np.linspace(0*um, 400*um, 256)
        wavelength = .55*um

        t0 = Scalar_mask_X(x=x0, wavelength=wavelength)
        t0.double_slit(x0=0, size=20*um, separation=50*um)
        t0.draw()

        t1 = Scalar_mask_XZ(x=x0, z=z0, wavelength=wavelength, n_background=1)
        z0 = 10*um
        z1 = 50*um
        v_globals = dict(z0=z0, z1=z1)
        t1.extrude_mask(t=t0,   z0=z0,   z1=z1,   refractive_index='1+0.25*(z-z0)/(z1-z0)',   v_globals=v_globals)
        t1.draw_refractive_index(draw_borders=False, )

        t1.save_data(filename=filename + '.npz', add_name='')
        save_figure_test(newpath, func_name, add_name='')
        assert True

    def test_extrude_mask_z_n_cte(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)

        x0 = np.linspace(-100*um, 100*um, 256)
        z0 = np.linspace(0*um, 400*um, 256)
        wavelength = .55*um

        t0 = Scalar_mask_X(x=x0, wavelength=wavelength)
        t0.double_slit(x0=0, size=20*um, separation=50*um)
        t0.draw()

        t1 = Scalar_mask_XZ(x=x0, z=z0, wavelength=wavelength, n_background=1)
        z0 = 10*um
        z1 = 50*um
        v_globals = dict(z0=z0, z1=z1)
        t1.extrude_mask(t=t0,   z0=z0,   z1=z1,  refractive_index=1.5,  v_globals=v_globals)
        t1.draw_refractive_index(draw_borders=False, )

        t1.save_data(filename=filename + '.npz', add_name='')
        save_figure_test(newpath, func_name, add_name='')
        assert True

    def test_mask_from_function(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)

        x0 = np.linspace(-100*um, 100*um, 256)
        z0 = np.linspace(0*um, 500*um, 256)
        wavelength = 0.6238*um

        f1 = '50*um'
        f2 = "175*um+np.tan(45*degrees)*(self.X-0*um)"
        z_sides = (-75*um, 75*um)
        v_globals = {'um': 1, 'np': np}

        t1 = Scalar_mask_XZ(x=x0, z=z0, wavelength=wavelength)
        t1.mask_from_function(r0=(0*um, 0*um), refractive_index=1.5, f1=f1, f2=f2, z_sides=z_sides, angle=0*degrees, v_globals=v_globals)

        t1.draw_refractive_index()

        t1.save_data(filename=filename + '.npz', add_name='')
        save_figure_test(newpath, func_name, add_name='')
        assert True

    def test_mask_from_array(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)

        x = np.linspace(-15*mm, 15*mm, 256)
        z = np.linspace(0*mm, 15*mm, 256)
        wavelength = 0.6328*um

        t1 = Scalar_mask_XZ(x, z, wavelength)

        script_dir = os.path.dirname(__file__)
        rel_path1 = "profile1.txt"
        abs_file_path1 = os.path.join(script_dir, rel_path1)

        rel_path2 = "profile2.txt"
        abs_file_path2 = os.path.join(script_dir, rel_path2)

        profile1 = loadtxt(abs_file_path1)
        profile2 = loadtxt(abs_file_path2)
        profile1[:, 1] = np.abs(profile1[:, 1])
        profile2[:, 1] = np.abs(profile2[:, 1])
        t1.mask_from_array(
            r0=(0*um, 0*um),
            refractive_index=1.5,
            array1=profile1 * 1000,  # pasar a micras
            array2=profile2 * 1000,  # pasar a micras
            x_sides=(-15*mm, 15*mm),
            angle=0*degrees,
            v_globals={},
            interp_kind='nearest')

        t1.draw_refractive_index(draw_borders=False)

        t1.save_data(filename=filename + '.npz', add_name='')
        save_figure_test(newpath, func_name, add_name='')
        assert True

    def test_object_by_surfaces(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)

        x0 = np.linspace(-20*um, 20*um, 256)
        z0 = np.linspace(0*um, 2000*um, 256)
        wavelength = 2*um

        r0=(0*um, 0*um)
        refractive_index = 4
        Fs = ['Xrot<3*um', 'Xrot>-3*um', 'Zrot>25*um', 'Zrot<1750*um']
        t1 = Scalar_mask_XZ(x=x0, z=z0, wavelength=wavelength, n_background=1)

        t1.object_by_surfaces(r0, refractive_index, Fs, angle=0*degrees, v_globals={})

        t1.draw_refractive_index(draw_borders=True)

        t1.save_data(filename=filename + '.npz', add_name='')
        save_figure_test(newpath, func_name, add_name='')
        assert True

    def test_add_surfaces(self):
        assert True

    def test_variable_refractive_index_1(self):
        """
        Here the refractive index is a function of positions x,z
        """
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)

        x0 = np.linspace(-100*um, 100*um, 256)
        print("Dx={}".format(x0[1] - x0[0]))
        z0 = np.linspace(0*um, 400*um, 256)
        wavelength = 50*um

        t0 = Scalar_mask_XZ(x=x0, z=z0, wavelength=wavelength, n_background=1.0)

        pn = dict(n_out=1.5,
                  n_center=4,
                  cx=0*um,
                  cz=100*um,
                  radius=75*um)

        center = (pn['cx'], pn['cz'])
        radius = pn['radius']
        # ref_index = '2*(((X-0)**2+(Z-300)**2)/75**2-0)'
        ref_index = "{p[n_out]}+({p[n_center]}-{p[n_out]})*(1-((X-{p[cx]})**2+(Z-{p[cz]})**2)/{p[radius]}**2)".format(
            p=pn)

        t0.cylinder(r0=center, radius=(radius, radius), refractive_index=ref_index, angle=0)

        t0.draw_refractive_index(draw_borders=False, scale='equal')

        t0.save_data(filename=filename + '.npz', add_name='')
        save_figure_test(newpath, func_name, add_name='')
        assert True

    def test_variable_refractive_index_2(self):
        """Here the refractive index is a function of positions z"""
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)

        x0 = np.linspace(-100*um, 100*um, 256)
        z0 = np.linspace(0*um, 400*um, 256)
        wavelength = .55*um

        t0 = Scalar_mask_X(x=x0, wavelength=wavelength)
        t0.double_slit(x0=0, size=20*um, separation=50*um)
        t0.draw()

        z_min = 10*um
        z_max = 50*um
        v_globals = dict(np=np)
        t1 = Scalar_mask_XZ(x=x0, z=z0, wavelength=wavelength, n_background=1)
        t1.extrude_mask(t=t0, z0=z_min, z1=z_max, refractive_index='1+0.25*np.abs(x/200)**2', v_globals=v_globals)
        t1.draw_refractive_index(draw_borders=False)

        t1.save_data(filename=filename + '.npz', add_name='')
        save_figure_test(newpath, func_name, add_name='')
        assert True

    def test_discretize_refractive_index(self):

        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)

        x0 = np.linspace(-100*um, 100*um, 256)
        print("Dx={}".format(x0[1] - x0[0]))
        z0 = np.linspace(0*um, 400*um, 256)
        wavelength = 50*um

        t0 = Scalar_mask_XZ(x=x0, z=z0, wavelength=wavelength, n_background=1.0)

        pn = dict(n_out=1.5,
                  n_center=4,
                  cx=0*um,
                  cz=100*um,
                  radius=75*um)

        center = (pn['cx'], pn['cz'])
        radius = pn['radius']
        # ref_index = '2*(((X-0)**2+(Z-300)**2)/75**2-0)'
        ref_index = "{p[n_out]}+({p[n_center]}-{p[n_out]})*(1-((X-{p[cx]})**2+(Z-{p[cz]})**2)/{p[radius]}**2)".format(
            p=pn)

        t0.cylinder(r0=center,
                  radius=(radius, radius),
                  refractive_index=ref_index,
                  angle=0)

        t0.discretize_refractive_index(num_layers=6)

        t0.draw_refractive_index(draw_borders=False, scale='equal')

        t0.save_data(filename=filename + '.npz', add_name='')
        save_figure_test(newpath, func_name, add_name='')
        assert True

    def test_add_masks(self):
        """
        several objects in the same
        """

        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)

        x0 = np.linspace(-100*um, 100*um, 256)
        z0 = np.linspace(0*um, 400*um, 256)
        wavelength = 2*um

        t1 = Scalar_mask_XZ(x=x0, z=z0, wavelength=wavelength, n_background=1)

        t1.cylinder(r0=(0, 100*um),
                  radius=(40*um, 40*um),
                  refractive_index=2.5,
                  angle=0)

        t1.cylinder(r0=(0, 100*um),
                  radius=(10*um, 10*um),
                  refractive_index=1,
                  angle=0)

        for pos_slit in [200, 250, 300, 350]:
            t1.slit(r0=(0*um, pos_slit*um),
                    aperture=100*um,
                    depth=10*um,
                    refractive_index=1.5 - 1.5j,
                    refractive_index_center='',
                    angle=0*degrees)

        t1.draw_refractive_index(draw_borders=False, scale='equal')

        t1.save_data(filename=filename + '.npz', add_name='')
        save_figure_test(newpath, func_name, add_name='')
        assert True

    def test_extrude_mask_simple(self):
        """
        takes an refractive index and convert a mask without depth, at one with
        depth
        """
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)

        x0 = np.linspace(-100*um, 100*um, 256)
        z0 = np.linspace(0*um, 400*um, 256)
        wavelength = .55*um

        t0 = Scalar_mask_X(x=x0, wavelength=wavelength)
        t0.double_slit(x0=0, size=20*um, separation=50*um)
        t0.draw()
        t1 = Scalar_mask_XZ(x=x0, z=z0, wavelength=wavelength, n_background=1)
        t1.extrude_mask(t=t0,  z0=10*um,  z1=50*um,  refractive_index=1.5 + 1.5j)
        t1.draw_refractive_index(draw_borders=True)

        t1.save_data(filename=filename + '.npz', add_name='')
        save_figure_test(newpath, func_name, add_name='')
        assert True

    def test_image(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)
        x0 = np.linspace(-100*um, 100*um, 1024)
        z0 = np.linspace(0*um, 200*um, 1024)
        wavelength = 0.6238*um

        t1 = Scalar_mask_XZ(x=x0, z=z0, wavelength=wavelength)
        script_dir = os.path.dirname(__file__)
        rel_path1 = "star_hole.png"
        image_name = os.path.join(script_dir, rel_path1)

        t1.image(filename=image_name,
                 n_max=2,
                 n_min=1,
                 angle=0*degrees,
                 invert=False)
        t1.draw_refractive_index()

        t1.save_data(filename=filename + '.npz', add_name='')
        save_figure_test(newpath, func_name, add_name='')
        assert True

    def test_semi_plane(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)
        x0 = np.linspace(-400*um, 400*um, 256)
        z0 = np.linspace(-100*um, 100*um, 256)
        wavelength = .5*um

        t1 = Scalar_mask_XZ(x=x0, z=z0, wavelength=wavelength)
        t1.semi_plane(r0=(0*um, 0*um),
                      refractive_index=2,
                      angle=0*degrees,
                      rotation_point=None)
        t1.draw_refractive_index()

        t1.save_data(filename=filename + '.npz', add_name='')
        save_figure_test(newpath, func_name, add_name='')
        assert True

    def test_layer(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)
        x0 = np.linspace(-200*um, 200*um, 256)
        z0 = np.linspace(-100*um, 100*um, 256)
        wavelength = .5*um

        t1 = Scalar_mask_XZ(x=x0, z=z0, wavelength=wavelength)
        t1.layer(r0=(50*um, 0*um),
                 depth=75*um,
                 refractive_index=2,
                 angle=0*degrees,
                 rotation_point=None)
        t1.draw_refractive_index()

        t1.save_data(filename=filename + '.npz', add_name='')
        save_figure_test(newpath, func_name, add_name='')
        assert True

    def test_rectangle(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)
        x0 = np.linspace(-100*um, 100*um, 256)
        z0 = np.linspace(0*um, 200*um, 256)
        wavelength = 0.6238*um

        t1 = Scalar_mask_XZ(x=x0, z=z0, wavelength=wavelength)
        t1.square(r0=(0*um, 100*um),
                     size=(150*um, 50*um),
                     angle=0*degrees,
                     refractive_index=1.5)
        t1.draw_refractive_index()

        t1.save_data(filename=filename + '.npz', add_name='')
        save_figure_test(newpath, func_name, add_name='')
        assert True

    def test_slit(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)
        x0 = np.linspace(-100*um, 100*um, 256 * 4)
        z0 = np.linspace(0*um, 250*um, 256 * 4)
        wavelength = 0.6238*um

        t1 = Scalar_mask_XZ(x=x0, z=z0, wavelength=wavelength)

        t1.slit(r0=(0*um, 50*um),
                aperture=50*um,
                depth=20*um,
                refractive_index=1.5 + 1j,
                refractive_index_center='',
                angle=0*degrees)

        t1.draw_refractive_index()

        t1.save_data(filename=filename + '.npz', add_name='')
        save_figure_test(newpath, func_name, add_name='')
        assert True

    def test_sphere(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)
        x0 = np.linspace(-100*um, 100*um, 256)
        z0 = np.linspace(0*um, 200*um, 256)
        wavelength = 0.6238*um

        t1 = Scalar_mask_XZ(x=x0, z=z0, wavelength=wavelength)

        t1.cylinder(r0=(0, 100*um),
                  radius=(75*um, 75*um),
                  refractive_index=1.5,
                  angle=0*degrees)

        t1.draw_refractive_index()

        t1.save_data(filename=filename + '.npz', add_name='')
        save_figure_test(newpath, func_name, add_name='')
        assert True

    def test_semi_cylinder(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)
        x0 = np.linspace(-200*um, 200*um, 256)
        z0 = np.linspace(-120*um, 120*um, 256)
        wavelength = .5*um

        t1 = Scalar_mask_XZ(x=x0, z=z0, wavelength=wavelength)
        t1.semi_cylinder(r0=(0*um, 0*um),  radius=(100, 100),  refractive_index=2,  angle=0*degrees)

        t1.draw_refractive_index(draw_borders=True,   min_incr=0.01,  scale='equal')

        t1.save_data(filename=filename + '.npz', add_name='')
        save_figure_test(newpath, func_name, add_name='')
        assert True

    def test_aspheric_surface_z(self):
        assert True

    def test_aspheric_lens(self):
        assert True

    def test_wedge(self):

        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)
        x0 = np.linspace(-100*um, 100*um, 256)
        z0 = np.linspace(0*um, 200*um, 256)
        wavelength = 0.6238*um

        t1 = Scalar_mask_XZ(x=x0, z=z0, wavelength=wavelength)

        t1.wedge(r0=(0*um, 0*um),
                 length=100*um,
                 refractive_index=1.5,
                 angle_wedge=22.5*degrees,
                 angle=0*degrees,
                 rotation_point=None)
        t1.draw_refractive_index()

        t1.save_data(filename=filename + '.npz', add_name='')
        save_figure_test(newpath, func_name, add_name='')
        assert True

    def test_prism(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)
        x0 = np.linspace(-150*um, 150*um, 256)
        z0 = np.linspace(0*um, 500*um, 4096)
        wavelength = 2*um

        t1 = Scalar_mask_XZ(x=x0, z=z0, wavelength=wavelength)
        t1.prism(r0=(100*um, 150*um),
                 length=200*um,
                 refractive_index=2,
                 angle_prism=60*degrees,
                 angle=90*degrees)
        t1.draw_refractive_index()

        t1.save_data(filename=filename + '.npz', add_name='')
        save_figure_test(newpath, func_name, add_name='')
        assert True

    def test_biprism(self):

        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)
        x0 = np.linspace(-100*um, 100*um, 256)
        z0 = np.linspace(-5*um, 395*um, 4096)
        wavelength = 4*um

        t1 = Scalar_mask_XZ(x=x0, z=z0, wavelength=wavelength)
        t1.biprism(r0=(0*um, 0*um),
                   length=200*um,
                   height=50*um,
                   refractive_index=1.5,
                   angle=0)
        t1.draw_refractive_index(draw_borders=True, scale='equal')

        t1.save_data(filename=filename + '.npz', add_name='')
        save_figure_test(newpath, func_name, add_name='')
        assert True

    def test_ronchi_grating(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)
        x0 = np.linspace(-500*um, 500*um, 256)
        z0 = np.linspace(0*um, 1400*um, 256)
        wavelength = 0.5*um

        t1 = Scalar_mask_XZ(x=x0, z=z0, wavelength=wavelength)
        t1.ronchi_grating(period=50*um, fill_factor=.5, length=500*um, height=20*um, r0=(0*um, 100*um), Dx=2*um, refractive_index=1.5 + 0.5j, heigth_substrate=25*um, refractive_index_substrate=1.5, angle=0*degrees)
        t1.draw_refractive_index()

        t1.save_data(filename=filename + '.npz', add_name='')
        save_figure_test(newpath, func_name, add_name='')
        assert True

    def test_ronchi_grating_convert(self):
        """
        generate a diffraction grating with substrate
        """
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)

        x0 = np.linspace(-500*um, 500*um, 256)
        z0 = np.linspace(0*um, 400*um, 256)
        wavelength = .55*um

        t0 = Scalar_mask_X(x=x0, wavelength=wavelength)
        t0.slit(x0=0, size=0*um)

        t1 = Scalar_mask_X(x=x0, wavelength=wavelength)
        t1.ronchi_grating(period=20*um, x0=0*um, fill_factor=0.5)
        t1.draw()

        t2 = Scalar_mask_XZ(x=x0, z=z0, wavelength=wavelength, n_background=1)

        t2.extrude_mask(t=t0, z0=10*um, z1=50*um, refractive_index=1.5)
        t2.extrude_mask(t=t1,   z0=50*um,   z1=55.5*um,   refractive_index=1.5 - 1.5)

        t2.draw_refractive_index(draw_borders=False)

        t2.save_data(filename=filename + '.npz', add_name='')
        save_figure_test(newpath, func_name, add_name='')
        assert True

    def test_sine_grating(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)
        x0 = np.linspace(-250*um, 250*um, 256)
        z0 = np.linspace(0*um, 1000*um, 256)
        wavelength = 0.6238*um

        t1 = Scalar_mask_XZ(x=x0, z=z0, wavelength=wavelength)

        t1.sine_grating(period=20*um,   heigth_sine=10*um,   heigth_substrate=100*um,   r0=(0*um, 200*um),   length=500*um,   Dx=2*um,   refractive_index=1.5,   angle=0*degrees)

        t1.draw_refractive_index()

        t1.save_data(filename=filename + '.npz', add_name='')
        save_figure_test(newpath, func_name, add_name='')
        assert True

    def test_probe(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)
        x0 = np.linspace(-12*um, 12*um, 256)
        z0 = np.linspace(0*um, 500*um, 256)
        wavelength = .6*um

        t1 = Scalar_mask_XZ(x=x0, z=z0, wavelength=wavelength)
        t1.probe(r0=(0, 50*um),
                 base=10*um,
                 length=200*um,
                 refractive_index=1.5,
                 angle=0*degrees)

        t1.draw_refractive_index()

        t1.save_data(filename=filename + '.npz', add_name='')
        save_figure_test(newpath, func_name, add_name='')
        assert True


    def test_rough_sheet(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)
        x0 = np.linspace(-150*um, 150*um, 256)
        z0 = np.linspace(-150*um, 150*um, 256)
        wavelength = 0.6238*um

        t1 = Scalar_mask_XZ(x=x0, z=z0, wavelength=wavelength)
        t1.rough_sheet(r0=(0*um, 0*um),  size=(200*um, 25*um),  t=10*um,  s=10*um,  refractive_index=1.5,  angle=0,  rotation_point=None)
        t1.draw_refractive_index()

        t1.save_data(filename=filename + '.npz', add_name='')
        save_figure_test(newpath, func_name, add_name='')
        assert True
