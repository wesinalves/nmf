'''
Author: Wesin Alves
year: 2017

ElasticNet for energy disaggregation

'''
from sklearn.decomposition import NMF
from sklearn.decomposition.nmf import _beta_divergence 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.linear_model import ElasticNet

############# 
#load dataset
#############
with open('dataset/Electricity_HPE.csv', 'rb') as f:
	heat_pump = np.loadtxt(f,skiprows=1, delimiter=',')

with open('dataset/Electricity_CDE.csv', 'rb') as f:
	clothes_dryer = np.loadtxt(f,skiprows=1, delimiter=',')

# create input matrix [1440,7]
# suggestion is crete it in a different file
n_day = 7
n_sample = 1440
n_devices = 2
attribute = 5 # power

init = 0
aggregate_signal = np.zeros((n_sample,n_day))
test_signal = np.zeros((n_sample,n_day))
#aggregate_signal[:,0] = heat_pump[init:1440*(0+1),attribute] + clothes_dryer[init:1440*(0+1),attribute]
for k in range(n_day):
	aggregate_signal[:,k] = heat_pump[init:1440*(k+1),attribute] + clothes_dryer[init:1440*(k+1),attribute]
	init = init + 1440

init2 = init
for n in range(n_day):
	test_signal[:,n] = heat_pump[init:init2+1440*(n+1),attribute]
	init = init + 1440



#############
#create model
#############
alpha = 0.1
enet = ElasticNet(alpha=alpha, l1_ratio=0.7)

############
#train step
############

y_pred_enet = enet.fit(aggregate_signal,test_signal).predict(aggregate_signal)
train_error = r2_score(test_signal,y_pred_enet)
print('Train error: ', train_error)


'''
##############
#evaluate step
##############
W_test = train.transform(test_signal)
test_error = _beta_divergence(test_signal, W_test, train.components_, 'frobenius', square_root=True)
print('Test error: ', test_error)
'''