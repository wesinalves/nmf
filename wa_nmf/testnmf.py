from nmf import *
import numpy as np 

X = np.abs(np.random.randn(1440,7))
model = NonnegativeMatrixFactorization(n_components=19,alpha=0.2, max_iter=1000, solver='mu')
D,A = model.initialize(X,type=2)
newD, newA = model.train(X,D,A)
print('******predict**********')
print(model.predict(newD,newA))
print('******original**********')
print(X)
'''
print('****D matrix******')
print(D)
print('****A matrix******')
print(A)
'''
X_train = X
X_test = X + 0.02
print('******train_error x test_error**********')
print(model.evaluate(X_train,X_test,newD,newA))