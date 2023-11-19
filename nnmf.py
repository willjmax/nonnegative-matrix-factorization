import math
import numpy as np
from scipy.optimize import nnls
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
from typing import Optional, Tuple


def fnorm(m: np.ndarray) -> float:
    '''Returns the Frobenious norm of a matrix'''

    return math.sqrt(np.square(m).sum())


def initialize(A: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    '''Performs k-means clustering and returns the indicator matrix'''

    kmeans = KMeans(n_clusters=k, n_init=10).fit(A)

    # build indicator matrix
    OHE = OneHotEncoder(sparse_output=False)
    labels = kmeans.labels_
    cols, rows = np.shape(A)
    W = OHE.fit_transform(labels.reshape(cols, 1))
    H = kmeans.cluster_centers_

    return W, H


def update(A: np.ndarray,
           W: np.ndarray,
           H: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    '''Performs one step of the NMF alternating update rule'''

    # fix H
    for row in range(A.shape[0]):
        W[row, :] = nnls(np.transpose(H), A[row, :])[0]

    # fix W
    for col in range(A.shape[1]):
        H[:, col] = nnls(W, A[:, col])[0]

    return W, H


def loss(A: np.ndarray, W: np.ndarray, H: np.ndarray) -> float:
    '''Returns the Frobenius norm of A - WH'''

    WH = W@H
    return fnorm(A - WH)


def nnmf(A: np.ndarray,
         k: int,
         max_iter: Optional[int] = 1000,
         tol: Optional[float] = 0.001) -> Tuple[np.ndarray, np.ndarray]:
    '''Perform the nnmf algorithm'''

    W, H = initialize(A, k)
    for x in range(0, max_iter):
        if x % 10 == 0:
            error_0 = loss(A, W, H)
            update(A, W, H)
            error_1 = loss(A, W, H)
            if abs(error_0 - error_1) < tol:
                return W, H
        else:
            update(A, W, H)
