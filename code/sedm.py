'''
This file contains Python codes for computing the Squared Euclidean Distance
Matrix between points based on the functions proposed by Bauckhage (2014).

Bauckhage, C. (2014). NumPy / SciPy Recipes for Data Science: Squared Euclidean
Distance Matrices, https://dx.doi.org/10.13140/2.1.4426.1127
'''

# import numpy
import numpy as np
# import numpy linear algebra module
import numpy.linalg as la
#import numba
from numba import njit


def naive(P, S):
    '''
    Compute the Squared Euclidean Distance Matrix between the points in P and S.
    This code uses the function numpy.linalg.norm in a "doubly-nested for".

    parameters
    ----------
    P: numpy array 2D
        3 x N matrix containing the coordinates x (1rt row), y (2nd row),
        z (3rd row) of N points. The ith column represents the ith point.
    S: numpy array 2d
        3 x M matrix containing the coordinates x (1rt row), y (2nd row),
        z (3rd row) of M points. The ith column represents the ith point.

    returns
    -------
    D: numpy array 2D
        N x M matrix whose elment ij of the squared Euclidean distance between
        the ith point in P and the jth point in S.
    '''

    # verify input and get number of rows/columns
    Mp, Np, Ms, Ns = _check_input(P, S)

    # initialize squared EDM D
    D = np.zeros((Np,Ns))

    # iterate over all rows of D
    for i in range(Np):
        for j in range(Ns):
            D[i,j] = la.norm(P[:,i] - S[:,j])**2
    return D


def avoid_sqrt(P, S):
    '''
    Compute the Squared Euclidean Distance Matrix between the points in P and S.
    This code avoids computing sqrt by using the function numpy.dot in a
    "doubly-nested for".

    parameters
    ----------
    P: numpy array 2D
        3 x N matrix containing the coordinates x (1rt row), y (2nd row),
        z (3rd row) of N points. The ith column represents the ith point.
    S: numpy array 2D
        3 x M matrix containing the coordinates x (1rt row), y (2nd row),
        z (3rd row) of M points. The ith column represents the ith point.

    returns
    -------
    D: numpy array 2D
        N x M matrix whose element ij of the squared Euclidean distance between
        the ith point in P and the jth point in S.
    '''

    # verify input and get number of rows/columns
    Mp, Np, Ms, Ns = _check_input(P, S)

    # initialize squared EDM D
    D = np.zeros((Np,Ns))

    # iterate over all rows of D
    for i in range(Np):
        for j in range(Ns):
            d = P[:,i] - S[:,j]
            D[i,j] = np.dot(d, d)
    return D


def vectorized(P, S):
    '''
    Compute the Squared Euclidean Distance Matrix between the points in P and S.
    This code avoids computing sqrt and inner loops by using a vectorized
    approach.

    parameters
    ----------
    P: numpy array 2D
        3 x N matrix containing the coordinates x (1rt row), y (2nd row),
        z (3rd row) of N points. The ith column represents the ith point.
    S: numpy array 2D
        3 x M matrix containing the coordinates x (1rt row), y (2nd row),
        z (3rd row) of M points. The ith column represents the ith point.

    returns
    -------
    D: numpy array 2D
        N x M matrix whose element ij of the squared Euclidean distance between
        the ith point in P and the jth point in S.
    '''

    # verify input and get number of rows/columns
    Mp, Np, Ms, Ns = _check_input(P, S)

    # compute components of matrix D
    D1 = np.sum(a=P*P, axis=0)
    D2 = np.sum(a=S*S, axis=0)
    D3 = 2*np.dot(P.T, S)

    # use broadcasting rules to add D1, D2 and D3
    D = D1[:,np.newaxis] + D2[np.newaxis,:] - D3

    return D


@njit
def naive_numba(P, S):
    '''
    Compute the Squared Euclidean Distance Matrix between the points in P and S.
    This code uses numba to optimize the naive approach with the function
    numpy.linalg.norm in a "doubly-nested for".

    parameters
    ----------
    P: numpy array 2D
        3 x N matrix containing the coordinates x (1rt row), y (2nd row),
        z (3rd row) of N points. The ith column represents the ith point.
    S: numpy array 2d
        3 x M matrix containing the coordinates x (1rt row), y (2nd row),
        z (3rd row) of M points. The ith column represents the ith point.

    returns
    -------
    D: numpy array 2D
        N x M matrix whose elment ij of the squared Euclidean distance between
        the ith point in P and the jth point in S.
    '''

    # verify input and get number of rows/columns
    assert P.ndim == S.ndim == 2, 'P and S must be 2D arrays'
    # determine dimensions of data matrices P and S
    Mp,Np = P.shape
    Ms,Ns = S.shape
    assert Mp == Ms == 3, 'P and S must have 3 rows'

    # initialize squared EDM D
    D = np.zeros((Np,Ns))

    # iterate over all rows of D
    for i in range(Np):
        for j in range(Ns):
            D[i,j] = la.norm(P[:,i] - S[:,j])**2
    return D


@njit
def avoid_sqrt_numba(P, S):
    '''
    Compute the Squared Euclidean Distance Matrix between the points in P and S.
    This code uses numba to optimize the approach with the function numpy.dot
    in a "doubly-nested for".

    parameters
    ----------
    P: numpy array 2D
        3 x N matrix containing the coordinates x (1rt row), y (2nd row),
        z (3rd row) of N points. The ith column represents the ith point.
    S: numpy array 2D
        3 x M matrix containing the coordinates x (1rt row), y (2nd row),
        z (3rd row) of M points. The ith column represents the ith point.

    returns
    -------
    D: numpy array 2D
        N x M matrix whose element ij of the squared Euclidean distance between
        the ith point in P and the jth point in S.
    '''

    # verify input and get number of rows/columns
    assert P.ndim == S.ndim == 2, 'P and S must be 2D arrays'
    # determine dimensions of data matrices P and S
    Mp,Np = P.shape
    Ms,Ns = S.shape
    assert Mp == Ms == 3, 'P and S must have 3 rows'

    # initialize squared EDM D
    D = np.zeros((Np,Ns))

    # iterate over all rows of D
    for i in range(Np):
        for j in range(Ns):
            d = P[:,i] - S[:,j]
            D[i,j] = np.dot(d, d)
    return D


def _check_input(P, S):
    '''
    Verify input validity.

    parameters
    ----------
    P: numpy array 2D
        3 x N matrix containing the coordinates x (1rt row), y (2nd row),
        z (3rd row) of N points. The ith column represents the ith point.
    S: numpy array 2D
        3 x M matrix containing the coordinates x (1rt row), y (2nd row),
        z (3rd row) of M points. The ith column represents the ith point.

    returns
    -------
    Mp, Np, Ms, Ns: integers
        Number of rows (Mp, Ms) and columns (Np, Ns) of P and S.
    '''
    assert P.ndim == S.ndim == 2, 'P and S must be 2D arrays'
    # determine dimensions of data matrices P and S
    Mp,Np = P.shape
    Ms,Ns = S.shape
    assert Mp == Ms == 3, 'P and S must have 3 rows'

    return Mp, Np, Ms, Ns
