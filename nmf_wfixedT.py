import numpy as np
from sklearn.decomposition import NMF
import sklearn.decomposition as decomp
from sklearn.decomposition.nmf import _beta_divergence 




X = np.array([[1, 1], [2, 1], [3, 1.2], [4, 1], [5, 0.8], [6, 1]])
X = X.T
#model = NMF(n_components=2, init='custom', random_state=0, beta_loss='frobenius')
W_init = np.array([[0, 0.46880687], [0.55762104, 0.38906185], [1.00396651, 0.41888706], [1.67369191, 0.22910467], [2.34341731, 0.03932227], [2.78976277, 0.06914749]])
#H_init = np.array([[2.09783018, 0.30560234], [2.13443044, 2.13171694]])
H_init = W_init.T
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