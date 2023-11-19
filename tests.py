import math
import numpy as np
import unittest
from nnmf import initialize, fnorm, update


class TestNNMF(unittest.TestCase):
    def test_fnorm(self):
        assert fnorm(np.eye(5)) == math.sqrt(5)
        assert fnorm(np.ones((5, 5))) == math.sqrt(25)

    def test_dimensions_initial(self):
        A = np.random.rand(100, 120)
        W, H = initialize(A, 10)
        assert W.shape == (100, 10)
        assert H.shape == (10, 120)

    def test_dimensions_update(self):
        A = np.random.rand(100, 120)
        W, H = initialize(A, 10)
        W, H = update(A, W, H)
        assert W.shape == (100, 10)
        assert H.shape == (10, 120)

    def test_nonnegativity_initial(self):
        A = np.random.rand(100, 120)
        W, H = initialize(A, 10)
        assert W.min() >= 0
        assert H.min() >= 0

        A = np.random.rand(300, 200)
        W, H = initialize(A, 20)
        assert W.min() >= 0
        assert H.min() >= 0

    def test_nonnegativity_update(self):
        A = np.random.rand(100, 120)
        W, H = initialize(A, 10)
        W, H = update(A, W, H)
        assert W.min() >= 0
        assert H.min() >= 0

        A = np.random.rand(300, 200)
        W, H = initialize(A, 20)
        W, H = update(A, W, H)
        assert W.min() >= 0
        assert H.min() >= 0

    def test_nan_initial(self):
        A = np.random.rand(100, 120)
        W, H = initialize(A, 10)
        assert any(map(any, np.isnan(W))) is False
        assert any(map(any, np.isnan(H))) is False

        A = np.random.rand(300, 200)
        W, H = initialize(A, 20)
        assert any(map(any, np.isnan(W))) is False
        assert any(map(any, np.isnan(H))) is False

    def test_nan_update(self):
        A = np.random.rand(100, 120)
        W, H = initialize(A, 10)
        W, H = update(A, W, H)
        assert any(map(any, np.isnan(W))) is False
        assert any(map(any, np.isnan(H))) is False

        A = np.random.rand(300, 200)
        W, H = initialize(A, 20)
        W, H = update(A, W, H)
        assert any(map(any, np.isnan(W))) is False
        assert any(map(any, np.isnan(H))) is False

    def test_inf_initial(self):
        A = np.random.rand(100, 120)
        W, H = initialize(A, 10)
        assert any(map(any, np.isinf(W))) is False
        assert any(map(any, np.isinf(H))) is False

        A = np.random.rand(300, 200)
        W, H = initialize(A, 20)
        assert any(map(any, np.isinf(W))) is False
        assert any(map(any, np.isinf(H))) is False

    def test_inf_update(self):
        A = np.random.rand(100, 120)
        W, H = initialize(A, 10)
        W, H = update(A, W, H)
        assert any(map(any, np.isinf(W))) is False
        assert any(map(any, np.isinf(H))) is False

        A = np.random.rand(300, 200)
        W, H = initialize(A, 20)
        W, H = update(A, W, H)
        assert any(map(any, np.isinf(W))) is False
        assert any(map(any, np.isinf(H))) is False


if __name__ == '__main__':
    unittest.main()
