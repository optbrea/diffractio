Algorithms
=============

The algorithms implemented for vector propagation are:

* Vector Fast Fourier Tranform (VFFT).
* Vector Rayleigh-Sommerfeld (VRS).
* Vector Chirped z-transform (VCZT).
* Fast Polarized Wave Propagation Method (FPWPM).
*



**Vector Rayleigh-Sommerfeld (VRS)**

- H. Ye,  C. W. Qiu, K. Huang, J.Teng, B. Luk’Yanchuk,  S.P. Yeo,(2013). Creation of a longitudinally polarized subwavelength hotspot with an ultra-thin planar lens: Vectorial Rayleigh-Sommerfeld method. Laser Physics Letters, 10(6). https://doi.org/10.1088/1612-2011/10/6/065004

The VRS method propagates (Ex,Ey,Ez) fields offering the advantage of significant reduction in computation, from flat diffractive elements (Thin Element Approximation) with full control of polarization. It addresses simultaneously both longitudinal polarization. This approach offers the advantage of significant reduction in computation.


**Vector Chirped Z Transform (VCZT)** 

-  Y. Hu, Z. Wang, X.Wang, S. Ji, C.Zhang, J. Li, W. Zhu, D.Wu, J. Chu (2020). Efficient full-path optical calculation of scalar and vector diffraction using the Bluestein method. Light: Science and Applications, 9(1). https://doi.org/10.1038/s41377-020-00362-z


**Fast Polarized Wave Propagation Method (FPWPM)** 

* M.Wende, J. Drozella, A. Toulouse, A.M. Herkommer (2022).   Fast algorithm for the simulation of 3D-printed microoptics based on the vector wave propagation method," Opt. Express 30(22), 40161-40173 https://doi.org/10.1364/OE.469178

    It is an efficient method for vector wave optical simulations of microoptics. The FPWPM is capable of handling comparably large simulation volumes while maintaining quick runtime.  By considering polarization in simulations, the FPWPM facilitates the analysis of optical elements which employ this property of electromagnetic waves as a feature in their optical design, e.g., diffractive elements, gratings, or optics with high angle of incidence like high numerical aperture lenses.


**Fast bidirectional vector wave propagation method (BWPM)** 

* M. Wende, J. Drozella, and A. Herkommer, "Fast bidirectional vector wave propagation method showcased on targeted noise reduction in imaging fiber bundles using 3D-printed micro optics," Opt. Express  31, 28874-28890 (2023). https://doi.org/10.1364/OE.497244

   BWPM allows for the additional consideration of reflected and backwards propagating electric and magnetic fields.

.. toctree::
   :maxdepth: 2
   :numbered:
   :glob:


   VFFT.ipynb
   VRS.ipynb
   VCZT.ipynb
   FPWPM.ipynb
   BWPM.ipynb
   comparison/VCZT_vs_VRS.ipynb