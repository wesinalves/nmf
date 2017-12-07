import numpy as np
X = np.array([[1, 1], [2, 1], [3, 1.2], [4, 1], [5, 0.8], [6, 1]])
from sklearn.decomposition import NMF


model = NMF(n_components=2, init='random', random_state=0)
W = model.fit_transform(X)
H = model.components_
error = model.reconstruction_err_

# W_init is slight changed

print("************ W matrix *****************")
print(W)
print("************ H matrix *****************")
print(H)
print("************ X matrix *****************")
print(X)
print("************ X' matrix *****************")
print(np.dot(W,H))
print("************ Error *****************")
print(error)

# train
# test
# evaluate