# svd-tests
Testing implementations of singular value decomposition (SVD) in Python.

The sklearn.decomposition.PCA module produces spurious results for input data arrays of dtype float32. Converting the same dataset to dtype float64 produces better PCA results.
