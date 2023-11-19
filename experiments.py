import numpy as np
from nnmf import initialize, update, loss
from sklearn.decomposition import NMF
from typing import Optional, Tuple


def sklearn_nmf(A: np.ndarray, W: np.ndarray, H: np.ndarray, k: int) -> float:
    '''Returns the error achieved by sklearn's implementation of NMF'''

    nmf = NMF(k, init='custom', solver='cd')
    W_sk = nmf.fit_transform(A, W=W.copy(), H=H.copy())
    H_sk = nmf.components_
    error_sk = loss(A, W_sk, H_sk)

    return error_sk


def test_random(rows: int, cols: int, k: int,
                it: Optional[int] = 1000) -> Tuple[float, float, float]:
    '''Performs NMF on a rectangular matrix using our implementation and
       sklearn's implementation
       Returns a triple containing the error immediately after initialization
       in the first component, the error of our implementation in the second
       component, and the error of sklearn's implementation in the third
       component'''

    A = np.random.rand(rows, cols)
    W, H = initialize(A, k)

    error_0 = loss(A, W, H)
    error_sk = sklearn_nmf(A, W, H, k)

    for x in range(it):
        update(A, W, H)

    error = loss(A, W, H)

    return error_0, error, error_sk
