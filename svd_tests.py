import numpy as np


X = np.load('X_t_test_noise_float32.npy')
# X = np.load('X_t_test_noise_float64.npy')

print('Original data shape:', X.shape)
print('X mean:', np.mean(X))
print('X max:', np.max(X))
print('X min:', np.min(X))

U, S, Vh = np.linalg.svd(X, full_matrices=False)
print(U.shape)
print(S.shape)
print(Vh.shape)

X_reconstructed = np.dot(U*S, Vh)
print('Reconstructed data shape:', X_reconstructed.shape)
print('X_reconstructed mean:', np.mean(X_reconstructed))
print('X_reconstructed max:', np.max(X_reconstructed))
print('X_reconstructed min:', np.min(X_reconstructed))

X_diff = X - X_reconstructed
print('Max discrepancy:', np.max(np.abs(X_diff)))
print(np.allclose(X, X_reconstructed))

smat = np.diag(S)
X_reconstructed = np.dot(U, np.dot(smat, Vh))
print('Reconstructed data shape:', X_reconstructed.shape)
print('X_reconstructed mean:', np.mean(X_reconstructed))
print('X_reconstructed max:', np.max(X_reconstructed))
print('X_reconstructed min:', np.min(X_reconstructed))

X_diff = X - X_reconstructed
print('Max discrepancy:', np.max(np.abs(X_diff)))
print(np.allclose(X, X_reconstructed))

