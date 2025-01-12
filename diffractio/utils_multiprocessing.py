# !/usr/bin/env python3

# ----------------------------------------------------------------------
# Name:        utils_multiprocessing.py
# Purpose:     Utility functions for multiprocessing operations
#
# Author:      Luis Miguel Sanchez Brea
#
# Created:     2024
# Licence:     GPLv3
# ----------------------------------------------------------------------

# flake8: noqa

import copyreg
import multiprocessing
import time
import types
from multiprocessing import Pool

import numpy as np


from .utils_typing import npt, Any, NDArray,  NDArrayFloat, NDArrayComplex


def _pickle_method(method):
    """function for multiprocessing in class

    """
    func_name = method.im_func.__name__
    obj = method.im_self
    cls = method.im_class
    if func_name.startswith(
            '__') and not func_name.endswith('__'):  # deal with mangled names
        cls_name = cls.__name__.lstrip('_')
        func_name = '_' + cls_name + func_name
    return _unpickle_method, (func_name, obj, cls)


def _unpickle_method(func_name, obj, cls):
    """
    function for multiprocessing in class
    """
    for cls in cls.__mro__:
        try:
            func = cls.__dict__[func_name]
        except KeyError:
            pass
        else:
            break
    return func.__get__(obj, cls)


copyreg.pickle(types.MethodType, _pickle_method, _unpickle_method)


# Funcion inversa a la anterior
def separate_from_iterable(iterable, shape=0):
    """This function does somehow the opposite of the previous one, it takes an iterable made of lists and separates each one in a different variable, reshaped with the desired shape
    """
    # Averiguar el numero de variables diferentes que habra
    N_var = len(iterable[0])
    # Make iterable array
    iterable = np.array(iterable)
    # Cambiar la forma de las variables
    variables = range(N_var)
    for indV in range(N_var):
        if shape == 0:
            variables[indV] = iterable[:, indV]
        else:
            variables[indV] = np.reshape(iterable[:, indV], shape)
    # Fin
    return variables


class auxiliar_multiprocessing():

    def __init__(self):
        pass

    # Method that executes the multiprocessing
    def execute_multiprocessing(self,
                                function,
                                var_iterable,
                                dict_constants=dict(),
                                Ncores=8):
        """_summary_

        Args:
            function (_type_): _description_
            var_iterable (_type_): _description_
            dict_constants (_type_, optional): _description_. Defaults to dict().
            Ncores (int, optional): num of cores. Defaults to 8.

        Returns:
            _type_: _description_
        """
        # Store data in object
        self.external_function = function
        self.dict_constants = dict_constants
        # Start multiprocessing if more than one core is required
        if Ncores > 1:
            pool = Pool(Ncores)
            print('Starting multiprocessing')
            result = pool.map(self.method_single_proc, var_iterable)
            print('Multiprocessing finished')
            pool.close()
            pool.join()
        # When only one core is asked, don't go to multiprocessing
        else:
            N = len(var_iterable)
            result = range(N)
            print('Starting process in only 1 core')
            for ind, elem in enumerate(var_iterable):
                result[ind] = function(elem, dict_constants)

        # Save and extract resultado
        self.resultado = result
        return result

    def method_single_proc(self, elem_iterable):
        # Method that is called in each iteration of the multiprocessing
        return self.external_function(elem_iterable, self.dict_constants)


def execute_multiprocessing(__function_process__,
                            dict_Parameters,
                            num_processors,
                            verbose: bool = False):
    """Executes multiprocessing reading a dictionary.

    Args:
        __function_process__ function tu process, it only accepts a dictionary
        dict_Parameters, dictionary / array with Args:
        num_processors, if 1 no multiprocessing is used
        verbose, prints processing time

    Returns:
        data: reults of multiprocessing
        processing time

    Examples:
        def __function_process__(xd):
            x = xd['x']
            y = xd['y']
            # grt = copy.deepcopy(grating)
            suma = x + y
            return dict(sumas=suma, ij=xd['ij'])

        def creation_dictionary_multiprocessing():
            # create Args: for multiprocessing
            t1 = time.time()
            X = np.linspace(1, 2, 10)
            Y = np.linspace(1, 2, 1000)
            dict_Parameters = []
            ij = 0
            for i, x in enumerate(X):
                for j, y in enumerate(Y):
                    dict_Parameters.append(dict(x=x, y=y, ij=[ij]))
                    ij += 1
            t2 = time.time()
            print("time creation dictionary = {}".format(t2 - t1))
            return dict_Parameters
    """
    t1 = time.time()
    if num_processors == 1 or len(dict_Parameters) < 2:
        data_pool = [__function_process__(xd) for xd in dict_Parameters]
    else:
        pool = multiprocessing.Pool(processes=num_processors)
        data_pool = pool.map(__function_process__, dict_Parameters)
        pool.close()
        pool.join()
    t2 = time.time()
    if verbose is True:
        print("num_proc: {}, time={}".format(num_processors, t2 - t1))
    return data_pool, t2 - t1
