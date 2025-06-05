#!/usr/bin/env python3

"""The setup script."""

from setuptools import find_packages, setup

with open('README.rst', encoding='utf8') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst', encoding='utf8') as history_file:
    history = history_file.read()

with open('requirements.txt', encoding='utf8') as req:
    requirements = req.readlines()

# requirements = [
#     'matplotlib', 'numpy', 'scipy', 'ezdxf',
#     'screeninfo', 'Pillow', 'numexpr', 'pandas', 'py_pol',
#     'ipywidgets', 'ipympl>=0.9.3', 'opencv-python', 'psutil', 'pyvista', 'python-ffmpeg', 'pyqt5',
# ]

setup_requirements = [
    'pytest-runner',
]

test_requirements = [
    'pytest',
]

setup(
    author="Luis Miguel Sanchez Brea",
    author_email='optbrea@ucm.es',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GPLv3 License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],
    description="Optical Diffraction and Interference (scalar and vectorial)",
    entry_points={
        'console_scripts': [
            'diffractio=diffractio.cli:main',
        ],
    },
    install_requires=requirements,
    license="GPLv3 license",
    # long_description=readme + '\n\n' + history,
    long_description=readme,
    include_package_data=True,
    keywords=[
        'diffractio', 'optics', 'diffraction', 'interference',
        'BPM', 'WPM', 'CZT', 'RS', 'VRS', 'VCZT', 'FPWPM'
    ],
    name='diffractio',
    packages=find_packages(include=['diffractio']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/optbrea/diffractio',
    version='1.0.0',
    zip_safe=False,
)