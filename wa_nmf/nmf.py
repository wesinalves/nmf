'''
Wesin Alves
2017-11-14
This code implements non negative matrix factorization (NMF) algorithm with sum to k constraint
for energy disaggregation

Paper:
Non-Intrusive Energy disaggregation using Non-negative Matrix factorization with sum-to-k constraint
by Alizera Rahimpour et al, 2017
####################
Math:
####################
classic NMF
A'1_k = argmin(frob(X - [D1,...,Dk][A1, A2 ... Ak]t)**2)

S2k-NMF
A'1_k = argmin( frob(X - [D1,...,Dk][A1, A2 ... Ak]t)**2 + beta(frob(U-QA)**2) )

frob means frobenious norm
X in R[m x d] is the aggregate signal
Dk in R[m x n] is devices base matrix / dictionary
Ak is actvation matrix for Dk
U is unitary matrix
beta is a small weight
Q is the matrix including 0 or 1 such that it forces the 
summation of activation coefficientes to be equal to one
ex: Q = [
		 111 00
		 000 11
		]


after calculate activation matrix for each device, use the next equation to estimate
Xi' = DiAi'
####################
Parameters:
####################
d: number of testing days
n: number of training days
m: number of samples in each column (1440)
k = number of all devices at home (19 in paper)


####################
Methods:
####################
init(type=2)
- returns D,A
train(X, initalD, initialA, fixedD=false)
- returns D,A
predict(D,A)
- returns X'
evaluate(X,X')
- returns errors

#######################
#Returns:
#######################
D
A
'''
from sklearn.decomposition.nmf import _beta_divergence 
import numpy as np
from methods import grad_desc, mult_update


class NonnegativeMatrixFactorization:
	
	def __init__(self, n_components=None, solver='cd', beta_loss='frobenious', tol=1e-4, max_iter=200,
		alpha=0., l1_ratio=0.):
		self.n_components = n_components
		self.solver = solver
		self.beta_loss = beta_loss
		self.tol = tol
		self.max_iter = max_iter
		self.alpha = alpha
		self.l1_ratio = l1_ratio

	def initialize(self, X, type=2):
		
		n_samples, n_features = X.shape
		
		if type == 2:
			avg = np.sqrt(X.mean() / self.n_components)
			A = avg * np.random.randn(self.n_components,n_features) 
			D = avg * np.random.randn(n_samples, self.n_components)
			np.abs(A,out=A)
			np.abs(D,out=D)
			return D,A

		if type == 1:
			avg = np.sqrt(X.mean() / self.n_components)
			A = avg * np.random.randn(self.n_components,n_features) 
			np.abs(A,out=A)
			return A
		
		if type == 0:
			avg = np.sqrt(X.mean() / self.n_components)
			D = avg * np.random.randn(n_samples, self.n_components)
			np.abs(D,out=D)
			return D


	def train(self,X,previousD,previousA, fixedD=False, verbose = False):

		if self.solver == 'cd':
			D,A = grad_desc(X,previousD,previousA, self.alpha, self.max_iter)

		if self.solver == 'mu':
			D,A = mult_update(X,previousD,previousA, self.max_iter)

		return D,A

	def predict(self, D,A):
		return np.dot(D,A)

	def evaluate(self,X_train,X_test,treinedD,treinedA):
		train_error = _beta_divergence(X_train,treinedD,treinedA, beta = 2, square_root=True)
		test_error = _beta_divergence(X_test,treinedD,treinedA, beta = 2, square_root=True)

		return train_error,test_error


