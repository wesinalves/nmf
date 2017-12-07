from sklearn.decomposition import NMF
from sklearn.decomposition.nmf import _beta_divergence 
import numpy as np

""" Dastaset """
X_train = np.array([[1, 1], [2, 1], [3, 1.2], [4, 1], [5, 0.8], [6, 1]])

X_test = np.array([[1, 2], [2, 2], [3.9, 2.2], [1.3, 1.2], [5.6, 0.9], [6, 1]])

""" Create model """
model = NMF(n_components=2, random_state=0, alpha=.1, l1_ratio=0.5)
train = model.fit(X_train)

""" Train error """
W_train = train.transform(X_train)
train_error = _beta_divergence(X_train,W_train,train.components_, 'frobenius', square_root=True)
print('Train error: ', train_error)

""" Test error """
W_test = train.transform(X_test)
test_error = _beta_divergence(X_test, W_test, train.components_, 'frobenius', square_root=True)
print('Test error: ', test_error)