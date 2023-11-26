import math
import numpy as np
from scipy.optimize import nnls
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
from typing import Optional, Tuple


def fnorm(m: np.ndarray) -> float:
    '''
    Returns the Frobenious norm of a matrix

    Parameters
    ----------
    m : ndarray of shape (rows, cols)

    Returns
    -------
    float
        The Frobenious norm of m
    '''

    return math.sqrt(np.square(m).sum())


def initialize(A: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Performs k-means clustering and returns both the
    indicator matrix and the centroid matrix

    Parameters
    ----------
    A : ndarray of shape (rows, cols)

    k : int
        The number of clusters for k-means

    Returns
    -------
    H : ndarray of shape (rows, k)
        The matrix of coefficients returned by k-means

    W : ndarray of shape (k, cols)
        The matrix of centroids returned by k-means
    '''

    # The matrix A is defined in the task to be an mxn matrix where m
    # is the number of features and n is the number of samples.
    # However, sklearn takes the tranpose of this matrix as input.
    # As a result, if A is given as in the task what we are really
    # solving is A^T = H^T W^T

    kmeans = KMeans(n_clusters=k, n_init=10).fit(A)

    # build indicator matrix
    OHE = OneHotEncoder(sparse_output=False)
    labels = kmeans.labels_
    cols, rows = np.shape(A)
    H = OHE.fit_transform(labels.reshape(cols, 1))
    W = kmeans.cluster_centers_

    return H, W


def update(A: np.ndarray,
           H: np.ndarray,
           W: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Performs one step of the NNMF alternating update rule

    Parameters
    ----------
    A : ndarray of shape (rows, cols)

    H : ndarray of shape (rows, k)

    W : ndarray of shape (k, cols)

    Returns
    -------
    H : ndarray of shape (rows, k)
        The updated matrix

    W : ndarray of shape (k, cols)
        The updated matrix
    '''

    # fix W
    for row in range(A.shape[0]):
        H[row, :] = nnls(np.transpose(W), A[row, :])[0]

    # fix H
    for col in range(A.shape[1]):
        W[:, col] = nnls(H, A[:, col])[0]

    return H, W


def loss(A: np.ndarray, H: np.ndarray, W: np.ndarray) -> float:
    '''
    Returns the Frobenius norm of (A - HW)

    Parameters
    ----------
    A : ndarray of shape (rows, cols)

    H : ndarray of shape (rows, k)

    W : ndarray of shape (k, cols)

    Returns
    -------
    float
        The Frobenious norm of (A - HW)
    '''

    HW = H@W
    return fnorm(A - HW)


def nnmf(A: np.ndarray,
         k: int,
         max_iter: Optional[int] = 1000,
         tol: Optional[float] = 0.001) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Perform the nnmf algorithm

    Parameters
    ----------
    A : ndarray of shape (rows, cols)
        The non-negative matrix to factorize

    k : int
        The number of clusters for k-means

    max_iter : int, default=1000
        The maximum number of iterations of the update rule

    tol : float, default=0.001
        The tolerance threshhold. After an update, if the Frobenious
        norm of (A-HW) decreases by a value less than the threshhold
        stop iterating and return.

    Returns
    -------
    H : ndarray of shape (rows, k)
        The matrix of coefficients in the factorization A=HW

    W : ndarray of shape (k, cols)
        The matrix of centroids in the factorization A=HW
    '''

    H, W = initialize(A, k)
    for x in range(0, max_iter):
        if x % 10 == 0:
            error_0 = loss(A, H, W)
            update(A, H, W)
            error_1 = loss(A, H, W)
            if abs(error_0 - error_1) < tol:
                return H, W
        else:
            update(A, H, W)

    return H, W
