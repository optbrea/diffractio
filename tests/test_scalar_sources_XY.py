# !/usr/bin/env python3

"""
Test for scalar_sources_XY"""

import datetime
import os
import sys

from diffractio import degrees, mm, no_date, np, plt, um
from diffractio.scalar_masks_XY import Scalar_mask_XY
from diffractio.scalar_sources_XY import Scalar_source_XY
from diffractio.utils_math import nearest2
from diffractio.utils_tests import comparison, save_figure_test

if no_date is True:
    date = '0'
else:
    now = datetime.datetime.now()
    date = now.strftime("%Y-%m-%d_%H")

path_base = "tests_results"
path_class = "scalar_sources_XY"

newpath = "{}_{}/{}/".format(path_base, date, path_class)

if not os.path.exists(newpath):
    os.makedirs(newpath)


class Test_Scalar_sources_XY():

    def test_plane_wave(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)

        # definición de parámetros iniciales
        length = 250*um
        npixels = 256

        # Generación de la clase
        u = Scalar_source_XY(x=np.linspace(-length/2, length/2, npixels),
                             y=np.linspace(-length/2, length/2, npixels),
                             wavelength=0.6328*um)

        # carga de la onda plana
        u.plane_wave(phi=2*degrees, theta=5*degrees)

        # draw y guardar
        u.draw(kind='phase')

        u.save_data(filename=filename + '.npz')
        save_figure_test(newpath, func_name)
        assert True

    def test_gauss_beam(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)
        """
        Generación de un haz gaussiano
        """

        # definición de parámetros iniciales
        length = 100*um
        x0 = np.linspace(-length/2, length/2, 256)
        y0 = np.linspace(-length/2, length/2, 256)
        wavelength0 = 0.6238*um

        # Generación de la clase
        u = Scalar_source_XY(x=x0, y=y0, wavelength=wavelength0)

        # carga del haz gaussiano
        u.gauss_beam(A=1,
                     r0=(0*um, 0*um),
                     z0=0,
                     w0=(25*um, 25*um),
                     phi=2*degrees,
                     theta=5*degrees)

        # draw y guardar
        u.draw(kind='field')

        u.save_data(filename=filename + '.npz')
        save_figure_test(newpath, func_name)
        assert True

    def test_spherical_wave(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)
        """
        Generación de una onda esférica
        """

        # definición de parámetros iniciales
        length = 750*um
        x0 = np.linspace(-length/2, length/2, 256)
        y0 = np.linspace(-length/2, length/2, 256)
        wavelength0 = 0.6238*um

        # Generación de la clase
        u = Scalar_source_XY(x=x0, y=y0, wavelength=wavelength0)

        # carga de la onda esferica
        u.spherical_wave(A=1, r0=(0*um, 0*um), z0=-5*mm, radius=300*um)

        # draw la onda esferica
        u.draw(kind='field', normalize='maximum')

        u.save_data(filename=filename + '.npz')
        save_figure_test(newpath, func_name)
        assert True

    def test_spherical_wave_colimated(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)
        """
        Generación de una onda esférica
        """

        # definición de parámetros iniciales
        length = 750*um
        x0 = np.linspace(-length/2, length/2, 256)
        y0 = np.linspace(-length/2, length/2, 256)
        wavelength0 = 0.6238*um

        # Generación de la clase
        u = Scalar_source_XY(x=x0, y=y0, wavelength=wavelength0)

        # carga de la onda esferica
        u.spherical_wave(A=1,
                         r0=(0*um, 0*um),
                         z0=-25.0*mm,
                         radius=1300*um)

        t = Scalar_mask_XY(x0, y0, wavelength0)
        t.lens(r0=(0*um, 0*um),
               radius=(300*um, 300*um),
               focal=(25*mm, 25*mm),
               angle=0)
        u_salida = u * t
        u.draw(kind='field')
        t.draw(kind='field')

        # draw la onda esferica
        u_salida.draw(kind='field', normalize='maximum')

        u.save_data(filename=filename + '.npz')
        save_figure_test(newpath, func_name)
        assert True

    def test_vortex_beam(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)
        """
        Generación de un vórtice
        """

        # definición del vortex_beam
        m = 3

        # definición de parámetros iniciales
        length = 750*um
        x0 = np.linspace(-length/2, length/2, 256)
        y0 = np.linspace(-length/2, length/2, 256)
        wavelength0 = 0.6238*um

        # Generación de la clase
        u = Scalar_source_XY(x=x0, y=y0, wavelength=wavelength0)

        # carga del vortex_beam
        u.vortex_beam(A=1, r0=(0*um, 0*um), w0=250*um, m=m)

        # draw el vortex_beam
        title = '$m_{vortice}= %d$' % (m)
        u.draw(kind='field', title=title)

        u.save_data(filename=filename + '.npz')
        save_figure_test(newpath, func_name)
        assert True

    def test_vortices(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)

        # matrix size
        M = 5

        # definición de parámetros iniciales
        length = 750*um
        x0 = np.linspace(-length/2, length/2, 256)
        y0 = np.linspace(-length/2, length/2, 256)
        wavelength0 = 0.6238*um
        u = Scalar_source_XY(x=x0, y=y0, wavelength=wavelength0)

        # Dos drawings, uno para la amplitude y otro para phase
        plt.figure(figsize=(10, 4.5))
        plt.suptitle("Vortices", fontsize=20)

        for m in range(M):
            u.vortex_beam(A=1, r0=(0*um, 0*um), w0=100*um, m=m)

            # carga de los drawings
            title = "(%d)" % (m)
            plt.subplot(2, M, m + 1)
            plt.axis('off')
            plt.title(title)
            h1 = plt.imshow(np.abs(u.u)**2)
            h1.set_cmap("gist_heat")

            plt.subplot(2, M, m + M + 1)
            plt.axis('off')
            h2 = plt.imshow(np.angle(u.u))
            h2.set_cmap("seismic")

        u.save_data(filename=filename + '.npz')
        save_figure_test(newpath, func_name)
        assert True

    def test_1laguerre(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)
        """
        Generación de haz de laguerre_beam
        """

        # matrix size
        N = 2
        K = 2

        # definición de parámetros iniciales
        length = 750*um
        x0 = np.linspace(-length/2, length/2, 256)
        y0 = np.linspace(-length/2, length/2, 256)
        wavelength0 = 0.6238*um

        # Generación de la clase
        u = Scalar_source_XY(x=x0, y=y0, wavelength=wavelength0)

        # carga del haz de Laguerre
        u.laguerre_beam(A=1,
                        n=N,
                        l=K,
                        r0=(0*um, 0*um),
                        w0=100*um,
                        z=0.01*um,
                        z0=0)

        # drawing
        title = r'$n=%d, k=%d$' % (N, K)
        u.draw(kind='field', title=title)

        u.save_data(filename=filename + '.npz')
        save_figure_test(newpath, func_name)
        assert True

    def test_num_data(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)
        """
        Generación de varios haces de num_data = 256
        """

        # matrix size
        N = 3
        K = 3

        # definición de parámetros iniciales
        length = 750*um
        x0 = np.linspace(-length/2, length/2, 256)
        y0 = np.linspace(-length/2, length/2, 256)
        wavelength0 = 0.6238*um
        u = Scalar_source_XY(x=x0, y=y0, wavelength=wavelength0)

        # Dos drawings, uno para la amplitude y otro para phase
        ID1 = plt.figure()
        plt.suptitle("amplitude", fontsize=20)
        ID2 = plt.figure()
        plt.suptitle("phase", fontsize=20)

        # Generación de la clase

        for n in range(N + 1):
            for k in range(K + 1):
                u.laguerre_beam(A=1,
                                n=n,
                                l=k,
                                r0=(0*um, 0*um),
                                w0=100*um,
                                z=0.01*um,
                                z0=0)

                title = "(%d,%d)" % (n, k)
                plt.figure(ID1.number)
                plt.subplot(N + 1, K + 1, (N + 1) * (k) + n + 1)
                plt.axis('off')
                plt.title(title, fontsize=14)
                h1 = plt.imshow(np.abs(u.u)**2)
                h1.set_cmap("gist_heat")

                plt.figure(ID2.number)
                plt.subplot(N + 1, K + 1, (N + 1) * (k) + n + 1)
                plt.axis('off')
                plt.title(title, fontsize=14)
                h2 = plt.imshow(np.angle(u.u))
                h2.set_cmap("seismic")

        plt.figure(ID1.number)
        plt.figure(ID2.number)

        u.save_data(filename=filename + '.npz')
        save_figure_test(newpath, func_name)
        assert True

    def test_Hermite_Gauss(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)

        # definción del tamaño y parámetros iniciales
        length = 250*um
        x0 = np.linspace(-length/2, length/2, 256)
        y0 = np.linspace(-length/2, length/2, 256)
        wavelength0 = 0.6238*um

        u = Scalar_source_XY(x=x0, y=y0, wavelength=wavelength0)

        # u.hermite_gauss_beam(I0=1, w=100*um, n = [0,0,1,1,3],     m =[0,1,0,1,3],  c_mn=[1,1,1,1,1])
        u.hermite_gauss_beam(A=1,
                             r0=(0*um, 0*um),
                             w0=100*um,
                             n=1,
                             m=1,
                             z=0,
                             z0=(0, 0))
        u.draw(kind='intensity')

        u.save_data(filename=filename + '.npz')
        save_figure_test(newpath, func_name)
        assert True

    def test_1zernike(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)

        # parámetros del polinomio de Zernike
        N = np.array([1, 1, 2, 2, 2])
        M = np.array([-1, 1, -2, 0, 2])
        c_nm = np.array([1, 1, 1, 1, 1])
        # n=1,2,3,...
        # m=-n:-n+2, ...., n-2, n

        # definción del tamaño y parámetros iniciales
        length = 1000*um
        x0 = np.linspace(-length/2, length/2, 256)
        y0 = np.linspace(-length/2, length/2, 256)
        wavelength0 = 0.6238*um

        # creación del u
        u = Scalar_source_XY(x=x0, y=y0, wavelength=wavelength0)

        u.zernike_beam(A=1, r0=(0*um, 0*um), radius=length/2, n=N, m=M, c_nm=c_nm)
        u.pupil()
        u.draw(kind='field')

        u.save_data(filename=filename + '.npz')
        save_figure_test(newpath, func_name)
        assert True

    def test_zernikes(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)
        """
        Generación de varios haces de Zernike
        """

        # matrix size
        N = 4

        # definición de parámetros iniciales
        length = 2*mm
        x0 = np.linspace(-length/2, length/2, 256)
        y0 = np.linspace(-length/2, length/2, 256)
        wavelength0 = 0.6238*um

        # Dibujo
        plt.figure(figsize=(6, 7))
        plt.suptitle(u"polinomios de Zernike (n,m)", fontsize=20)

        # Generación de la clase
        u = Scalar_source_XY(x=x0, y=y0, wavelength=wavelength0)

        for n in range(0, N + 1):
            pos = 0
            for m in range(-n, n + 1, 2):
                pos = pos + 1

                u.zernike_beam(A=1,
                               r0=(0*um, 0*um),
                               radius=length/2,
                               n=np.array([n]),
                               m=np.array([m]),
                               c_nm=np.array([1]))
                u.pupil()

                # carga de los drawings
                title = "(%d,%d)" % (n, m)
                plt.subplot(N + 1, N + 1, n * (N + 1) + pos)
                plt.axis('off')
                plt.title(title, fontsize=14)
                h2 = plt.imshow(np.angle(u.u))
                plt.clim(vmin=-np.pi, vmax=np.pi)
                h2.set_cmap("seismic")

        u.save_data(filename=filename + '.npz')
        save_figure_test(newpath, func_name)
        assert True

    def test_plane_waves_several_inclined(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)

        x0 = np.linspace(-500*um, 500*um, 256)
        y0 = np.linspace(-500*um, 500*um, 256)
        wavelength = 0.6328*um

        u0 = Scalar_source_XY(x=x0, y=y0, wavelength=wavelength)
        u0.plane_waves_several_inclined(A=1,
                                        num_beams=(5, 5),
                                        max_angle=(5*degrees, 180*degrees))
        u0.draw(kind='field')

        u0.save_data(filename=filename + '.npz')
        save_figure_test(newpath, func_name)
        assert True

    def test_plane_waves_dict(self):
        assert True

    def test_gauss_beams_several_parallel(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)

        x0 = np.linspace(-500*um, 500*um, 256)
        y0 = np.linspace(-500*um, 500*um, 256)
        wavelength = 0.6328*um

        u0 = Scalar_source_XY(x=x0, y=y0, wavelength=wavelength)
        u0.gauss_beams_several_parallel(A=1,
                                        num_beams=(5, 5),
                                        w0=50*um,
                                        z0=0*um,
                                        r0=(0*um, 0*um),
                                        r_range=(750*um, 750*um),
                                        theta=0*degrees,
                                        phi=0*degrees)
        u0.draw(kind='field')

        u0.save_data(filename=filename + '.npz')
        save_figure_test(newpath, func_name)
        assert True

    def test_gauss_beams_several_inclined(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)

        x0 = np.linspace(-500*um, 500*um, 256)
        y0 = np.linspace(-500*um, 500*um, 256)
        wavelength = 0.6328*um

        u0 = Scalar_source_XY(x=x0, y=y0, wavelength=wavelength)
        u0.gauss_beams_several_inclined(A=1,
                                        num_beams=(5, 5),
                                        w0=250*um,
                                        r0=(0*um, 0*um),
                                        z0=0*um,
                                        max_angle=(10*degrees,
                                                   180*degrees))
        u0.draw(kind='field')

        u0.save_data(filename=filename + '.npz')
        save_figure_test(newpath, func_name)
        assert True

    def test_interferences(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)

        length = 2*mm
        x0 = np.linspace(-length/2, length/2, 256)
        y0 = np.linspace(-length/2, length/2, 256)
        wavelength0 = 0.6238*um

        u1 = Scalar_source_XY(x=x0, y=y0, wavelength=wavelength0)
        u2 = Scalar_source_XY(x=x0, y=y0, wavelength=wavelength0)

        u1.gauss_beam(A=1,
                      r0=(0*um, 0*um),
                      w0=(500*um, 500*um),
                      z0=0*um,
                      phi=2*degrees,
                      theta=10*degrees)
        u2.gauss_beam(A=1,
                      r0=(0*um, 0*um),
                      w0=(500*um, 500*um),
                      z0=0*um,
                      phi=2*degrees,
                      theta=-10*degrees)

        u3 = u1 + u2
        u3.draw(kind='intensity', title="$interferencias$")

        u3.save_data(filename=filename + '.npz')
        save_figure_test(newpath, func_name)
        assert True

    def test_extent_source(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)
        """in this test I get a previous mask and then I fill the mask with sperical-waves
        """

        # definitions of space and wavelength
        num_data = 256
        length = 250*um
        x = np.linspace(-length/2, length/2, num_data)
        y = np.linspace(-length/2, length/2, num_data)
        wavelength = 0.6328*um

        # location of spherical sources
        z0 = -5*mm

        # I define the mask where point sources are defined
        t_mask = Scalar_mask_XY(x, y, wavelength)
        t_mask.circle(r0=(0*um, 0*um), radius=(25*um, 25*um))
        t_mask.draw()

        # distance between point sources (if possible all pixels)
        dist_sources_x = 2*um
        dist_sources_y = 2*um

        pos_x = np.arange(x[0], x[-1], dist_sources_x)
        pos_y = np.arange(y[0], y[-1], dist_sources_y)

        ix, vx, distx = nearest2(x, pos_x + 1)
        iy, vy, disty = nearest2(y, pos_y + 1)

        u1 = Scalar_source_XY(x, y, wavelength)
        x0, y0 = np.meshgrid(ix, iy)
        u1.u[x0, y0] = 1

        u_mask = u1 * t_mask  # I have to extract positions from here

        xpos = u_mask.X[u_mask.u > 0.5]
        ypos = u_mask.Y[u_mask.u > 0.5]

        u_final = Scalar_source_XY(x, y, wavelength)
        u_temp = Scalar_source_XY(x, y, wavelength)
        for x_i, y_i in zip(xpos, ypos):
            u_temp.spherical_wave(A=1, r0=(x_i, y_i), z0=z0, radius=0*um)
            u_final = u_final + u_temp

        u_final.draw(kind='field')

        u_final.save_data(filename=filename + '.npz')
        save_figure_test(newpath, func_name)
        assert True
