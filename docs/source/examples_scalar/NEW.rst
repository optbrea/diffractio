================================================
New Features
================================================


1.0.0 (2025-01-14)
--------------------------------

* Improvements:

  - New Algorithms for Vector propagation, in particular FPWPM.

  - New 3D visualization tools, using pyvista.

  - Use of Ezdxf for 3X DXF parts.

  - Improvement in documentation and examples.

  - Typing variables in functions.
 

* Fix bugs:

  - ndgrid function -> meshgrid (for >2D matrices).


* Other:

  - Change license to GPLv3.


0.2.3 (2023-11-21)
--------------------------------

* Improvement:

  - XYZ drawings is removed temporally.

  - utils_slicer.py is deprecated.



* Fix bugs:

  - Bug in XYZ video

  - Blazed grating wrong defined. 
  
  - bug in Scalar_field_XY.kernelRSinverse

* New features:

  - XY masks: new polygon, regular_polygon, star, and superformula functions. 

  - XY masks: new circular_sector function.

  - XY masks: new lens_cyliindrical function.

  - X, XY, XYZ fields: new conjugate function.

  - WPM function without storing all the data.

  - X fields: inverse_amplitude, inverse_phase, pupil

