import numpy as np
from sklearn.decomposition import NMF
import sklearn.decomposition as decomp
from sklearn.decomposition.nmf import _beta_divergence 




X = np.array([[1, 1], [2, 1], [3, 1.2], [4, 1], [5, 0.8], [6, 1]])
H_init = np.array([[2.09783018, 0.30560234], [2.13443044, 2.13171694]])
W,H,n_iter = decomp.non_negative_factorization(X, H=H_init,
	regularization='transformation', n_components=2, update_H=False)

error = _beta_divergence(X,W,H_init, 'frobenius', square_root=True)
print(error)
'''
it is not possible to fixe W matrix, just H matrix.
'''

print("************ W matrix *****************")
print(W)
print("************ H matrix *****************")
print(H)
print("************ X matrix *****************")
print(X)
print("************ X' matrix *****************")
print(np.dot(W,H))
#print("************ Error *****************")
#print(error)

# train
# test
# evaluate