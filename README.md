# Research Software Engineer Pre-Interview Task
This repo contains all of the materials for the pre-interview task.

1. The NNMF library is found in ```code/nnmf.py```
2. A Jupyter notebook containing a comparison of the NNMF implementation to ```sklearn.decomposition.NMF``` is found in ```code/experiment.ipynb```
3. Unit tests are contained in ```code/tests.py```
4. The TeX files for the math questions are in the ```math/``` folder
5. Continuous integration is in ```.github/workflows/```. On each push and pull request we perform linting with ```flake8``` and run the unit tests in ```code/tests.py```.


## The NNMF library
Let $A$ be a nonnegative $m \times n$ matrix where the $n$ column vectors are viewed at data points with $m$ features.
The goal is to factor $A$ into two nonnegative matrices $A \approx WH$ where the dimensions of $W$ and $H$ are 
$m \times k$ and $k \times n$. $W$ is the matrix of centroids, an $H$ is the coefficient matrix.

Our implementation follows three steps:
1. Initialization - we initialize $W$ and $H$ via $k$-means clustering. $W$ is initialized to the matrix consisting of the $k$ centroids, and $H$ is initialized to the indicator matrix which assigns each vector to a cluster
2. Update - we update $W$ and $H$ by applying non-negative least squares (NLS) in an alternating manner. That is, we update the rows of $W$ by applying NLS to $H$ and $A$, then update the columns of $H$ by applying NLS to $W$ and $A$.
3. Evalutation - we evaluate the solution with the enius norm $\||A - WH\||_F$.

#### A note on notation
In this implementation we rely on ```sklearn.cluster.KMeans``` for our initialization. This package assumes the $m \times n$
matrix consists of $m$ data points with $n$ features. This is the transpose of the setup given in the pre-interview task.
As a result, our implementation formulates the problem as $A^T \approx H^T W^T$.

### Documentation
The library is contained in ```code/nnmf.py``` and consists of the five functions defined here.
| Function   | Input                                                                                                                                               | Output                                        | Description                                                                                                                            |
|------------|-----------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------|
| initialize | <code>A: numpy.ndarray</code><br><br> <code>k: int</code>                                                                                               | <code>(numpy.ndarray, numpy.ndarray)</code>   | Returns the initial factorization.                                                                                                     |
| update     | <code>A: numpy.ndarray</code><br><br> <code>H: numpy.ndarray</code><br><br> <code>W: numpy.ndarray</code>                                                   | <code> (numpy.ndarray, numpy.ndarray) </code> | Performs one step of the alternating NLS update and return <code>(H, W)</code>                                                                                     |
| fnorm      | <code>m: numpy.ndarray</code>                                                                                                                       | <code>float</code>                            | Returns the Frobenius norm of a matrix.                                                                                               |
| loss       | <code> A: numpy.ndarray </code><br><br> <code> H: numpy.ndarray </code><br><br> <code> W: numpy.ndarray </code>                                             | <code>float</code>                            | Returns <code>fnorm(A - H@W)</code>.                                                                                                   |
| nnmf       | <code>A: numpy.ndarray</code><br><br> <code>k: int</code><br><br> <code>max_iter: Optional[int] = 1000</code><br><br> <code>tol: Optional[float] = 0.001</code> | <code> (numpy.ndarray, numpy.ndarray) </code> | Performs the NNMS algorithm. Terminate after <code>max_iter</code> iterations or after achieving an error tolerance of <code>tol</code>. Returns <code>(H, W)</code> |
