'''
Wesin Alves
2017-11-14
This code implements non negative matrix factorization (NMF) algorithm with sum to k constraint
for energy disaggregation
'''
import numpy as np 

def  normalize_cols(matrix):
	return matrix / np.maximum(np.sum(matrix, 0), 1e-7)

def grad_desc(X,previousD,previousA, alpha, max_iter, eps=1e-7):
	
	D = previousD
	A = previousA
	for i in range(max_iter):
		grad_D = np.dot((X - np.dot(D,A)), A.T)
		grad_A = np.dot(D.T, (X - np.dot(D,A)))
		
		D = D + alpha * grad_D
		D[(grad_D < eps) & (D < eps)] = 0
		D = normalize_cols(D)

		A = A + alpha * grad_A
		A[(grad_A < eps) & (A < eps)] = 0
		A = normalize_cols(A)

	return D,A

def mult_update(X,previousD,previousA, max_iter, eps=1e-7):
	
	D = previousD
	A = previousA
	for i in range(max_iter):
		A = A * np.dot(D.T,X) / np.maximum(np.dot(D.T, np.dot(D,A)), eps)
		D = D * np.dot(X,A.T) / np.maximum(np.dot(D, np.dot(A,A.T)), eps)

	return D,A

def cnmf(X, previousD, previousA, max_iter, alpha, beta, eps=1e-7):
	D = previousD
	A = previousA
	for i in range(max_iter):
		A = A * np.dot(D.T,X) / (np.dot(D.T, np.dot(D,A)) + beta * A + eps)
		D = D * np.dot(X,A.T) / (np.dot(D, np.dot(A,A.T)) + alpha * D + eps)

	return D,A	