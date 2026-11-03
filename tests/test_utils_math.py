# !/usr/bin/env python3


# -------------------------------------------------------------------------------
# Name:       scalar_fields_XZ.py
# Purpose:    tests
#
# Author:    Luis Miguel Sanchez Brea
#
# Created:    2017
# Copyright:
# Licence:    GPL
# -------------------------------------------------------------------------------
import datetime
import os
import sys

import numpy as np
from diffractio import eps, no_date
from diffractio.utils_math import (amplitude2phase, binarize, distance,
                                   nearest, nearest2, normalize)
from diffractio.utils_tests import comparison

if no_date is True:
    date = '0'
else:
    now = datetime.datetime.now()
    date = now.strftime("%Y-%m-%d_%H")

path_base = "tests_results"
path_class = "utils_math"

newpath = "{}_{}/{}/".format(path_base, date, path_class)

if not os.path.exists(newpath):
    os.makedirs(newpath)


class Test_utils_math():
    def test_distance(self):
        func_name = sys._getframe().f_code.co_name

        solution = 1.
        x1 = np.array([0, 1, 1])
        x2 = np.array([0, 0, 1])
        proposal = distance(x1, x2)
        assert comparison(proposal, solution, eps), func_name

        solution = np.sqrt(3)
        x1 = np.array([0, 0, 0])
        x2 = np.array([1, 1, 1])
        proposal = distance(x1, x2)
        assert comparison(proposal, solution, eps), func_name

        solution = np.sqrt(2)
        x1 = np.array([0, 0, 1j])
        x2 = np.array([0, 0, 1])
        proposal = distance(x1, x2)
        assert comparison(proposal, solution, eps), func_name

    def test_nearest(self):
        func_name = sys._getframe().f_code.co_name

        solution = 10
        x = np.linspace(0, 10, 21)
        x0 = 5
        proposal, _, _ = nearest(x, x0)
        assert comparison(proposal, solution, eps), func_name

    def test_nearest2(self):
        func_name = sys._getframe().f_code.co_name

        x = np.linspace(0, 10, 11)

        solution = x
        proposal, _, _ = nearest2(x, x)
        assert comparison(proposal, solution, eps), func_name

    def test_meshgrid(self):
        func_name = sys._getframe().f_code.co_name

        solution = np.array([[0, 1], [0, 1], [0, 1]])
        x, y = [0, 1], [2, 3, 4]
        X, Y = np.meshgrid(x, y)
        print(X)
        print(solution)
        proposal = X
        assert comparison(proposal, solution, eps), func_name

    def test_binarize(self):
        func_name = sys._getframe().f_code.co_name

        solution = np.array([0., 0., 0., 1., 1., 1.])
        vector = np.linspace(0, 1, 6)
        proposal = binarize(vector, min_value=0, max_value=1)
        assert comparison(proposal, solution, eps), func_name

        solution = np.array([-1., -1., -1., 1., 1., 1.])
        vector = np.linspace(-1, 1, 6)
        proposal = binarize(vector, min_value=-1, max_value=1)
        assert comparison(proposal, solution, eps), func_name


    def test_normalize(self):
        func_name = sys._getframe().f_code.co_name

        solution = np.array([1, 1]) / np.sqrt(2)
        v = np.array([1, 1])
        proposal = normalize(v, 2)
        assert comparison(proposal, solution, eps), func_name

        solution = np.array([1 + 1j, 0]) / np.sqrt(2)
        v = np.array([1 + 1j, 0])
        proposal = normalize(v, 2)
        assert comparison(proposal, solution, eps), func_name

