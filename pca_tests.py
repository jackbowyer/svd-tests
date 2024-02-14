import numpy as np
from sklearn.decomposition import PCA

X = np.load('X_t_test_noise_float32.npy')
# X = np.load('X_t_test_noise_float64.npy')

# Percentage of variance to retain in PCA
variance = 0.999

# PCA model fitting and transformations
pca_model = PCA(n_components=variance, svd_solver='auto')

X_compressed = pca_model.fit_transform(X)
print('Original shape:', X.shape)
print('X mean:', np.mean(X))
print('X max:', np.max(X))
print('X min:', np.min(X))

X_reconstructed = pca_model.inverse_transform(X_compressed)
print('X_reconstructed mean:', np.mean(X_reconstructed))
print('X_reconstructed max:', np.max(X_reconstructed))
print('X_reconstructed min:', np.min(X_reconstructed))

# Calculate the maximum discrepancy (error) with respect to the original data
X_diff = X - X_reconstructed
print('Max discrepancy:', np.max(np.abs(X_diff)))
print(np.allclose(X, X_reconstructed))
