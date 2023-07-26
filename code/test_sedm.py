import numpy as np
import scipy.spatial as spt
from numpy.testing import assert_almost_equal as aae
from pytest import raises
import sedm


def test_sedm_functions_bad_shape():
    'code must stop if input has bad shape'
    # list of function in sedm
    functions = [sedm.naive, sedm.avoid_sqrt, sedm.vectorized,
                 sedm.naive_numba, sedm.avoid_sqrt_numba]

    # wrong number of rows (it must be 3)
    P = np.empty((2, 5))
    S = np.empty((3, 4))

    for f in functions:
        raises(AssertionError, f, P, S)
    for f in functions:
        raises(AssertionError, f, S, P)


def test_sedm_functions_bad_ndim():
    'code must stop if input has ndim different from 2'
    # list of function in sedm
    functions = [sedm.naive, sedm.avoid_sqrt, sedm.vectorized,
                 sedm.naive_numba, sedm.avoid_sqrt_numba]

    # wrong ndim (it must be 2)
    S = np.empty(3)
    P = np.identity(3)

    for f in functions:
        raises(AssertionError, f, P, S)
    for f in functions:
        raises(AssertionError, f, S, P)


def test_comparison_functions():
    'check if all functions produce the same result as scipy'
    np.random.seed(13)
    P = np.random.rand(3, 8)
    S = np.random.rand(3,11)

    # list of function in sedm
    functions = [sedm.naive, sedm.avoid_sqrt, sedm.vectorized,
                 sedm.naive_numba, sedm.avoid_sqrt_numba]

    # compute a reference output with scipy
    scipy_result = spt.distance.cdist(P.T, S.T, 'sqeuclidean')

    for f in functions:
        aae(f(P, S), scipy_result, decimal=10)


def test_comparison_vectorized_dictionary_scipy():
    'check if vectorized_dictionary produces the same result as scipy'
    np.random.seed(13)
    P = np.random.rand(3, 8)
    S = np.random.rand(3,11)

    P_dict = {
        'x' : np.array(P[0]),
        'y' : np.array(P[1]),
        'z' : np.array(P[2])
    }

    S_dict = {
        'x' : np.array(S[0]),
        'y' : np.array(S[1]),
        'z' : np.array(S[2])
    }

    # compute with sedm.vectorized_dictionary
    computed = sedm.vectorized_dictionary(P_dict, S_dict)

    # compute a reference output with scipy
    scipy_result = spt.distance.cdist(P.T, S.T, 'sqeuclidean')

    aae(computed, scipy_result, decimal=10)
