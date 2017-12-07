import numpy as np
X = np.array([[1, 1], [2, 1], [3, 1.2], [4, 1], [5, 0.8], [6, 1]])
from sklearn.decomposition import NMF


model = NMF(n_components=2, init='custom', random_state=0)
W_init = np.array([[0, 0.468], [0.55, 0.38], [1, 0.41], [1.67, 0.22], [2.34, 0.39], [2.78, 0.06]])
H_init = np.array([[0.5, 0.5], [0.5, 0.5]])
W = model.fit_transform(X, W=W_init, H=H_init)
H = model.components_
error = model.reconstruction_err_

# W_init is slight changed

print("************ W matrix *****************")
print(W)
print("************ H matrix *****************")
print(H)
print("************ X matrix *****************")
print(X)

w0 =W[:,0].reshape(6,1)
h0 = H[0,:].reshape(1,2)
w1 =W[:,1].reshape(6,1)
h1 = H[1,:].reshape(1,2)

x1 = np.multiply(w0,h0)
x2 = np.multiply(w1,h1)

print("************ X1' matrix *****************")
print(x1)
print("************ X2' matrix *****************")
print(x2)
print("************ X1'+X2' matrix *****************")
print(x1+x2)
print("************ X' matrix *****************")
print(np.dot(W,H))

Y = X - np.dot(W,H)

print(error)
R = np.sum(0.5 * (Y * Y))
print(R)

'''
print(np.dot(np.array(W[:,0]),np.array(H[0,:])))
print("************ Error *****************")
print(error)
'''
# train
# test
# evaluate