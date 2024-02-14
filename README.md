# svd-tests
Testing implementations of singular value decomposition (SVD) in Python.

The sklearn.decomposition.PCA module produces spurious results for input data arrays of dtype float32. Converting the same dataset to dtype float64 produces better PCA results (see pca_tests.py). Results for both datasets do not fall within the default tolerances of np.allclose()

The numpy.linalg.svd module performs SVD on both test datasets within the default tolerances of np.allclose() (see svd_tests.py).

Test data:
X_t_test_noise_float32.npy
X_t_test_noise_float64.npy

The data is compressed as the files are too large to be uploaded directly.
