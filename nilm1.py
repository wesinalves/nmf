'''
Author: Wesin Alves
year: 2017

Nonnegative matrix factorization for energy disaggregation

'''
from sklearn.decomposition import NMF
from sklearn.decomposition.nmf import _beta_divergence 
import numpy as np
import matplotlib.pyplot as plt

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
	test_signal[:,n] = heat_pump[init:init2+1440*(n+1),attribute] + clothes_dryer[init:init2+1440*(n+1),attribute]
	init = init + 1440

'''
visualize inputs 

for x in range(10):
	print('aggregate_signal: {0}, heat: {1}, clothes: {2}'.format(aggregate_signal[x,0], heat_pump[x,5], clothes_dryer[x,5]))

print(heat_pump.shape)
print(aggregate_signal.shape)
'''


#############
#create model
#############
model = NMF(n_components = 2, init = 'random')

############
#train step
############

train = model.fit(aggregate_signal)
W_train = train.transform(aggregate_signal)
train_error = _beta_divergence(aggregate_signal,W_train,train.components_, 'frobenius', square_root=True)
print('Train error: ', train_error)

'''
estimated_signal = train.inverse_transform(W_train)

for k in range(n_day):
	eval('plt.subplot({}1{})'.format(n_day,k+1))
	plt.plot(estimated_signal[:,k], label='Estimated')
	plt.plot(aggregate_signal[:,k], label='Aggregate')



plt.legend()
plt.show()
'''

##############
#evaluate step
##############
W_test = train.transform(test_signal)
test_error = _beta_divergence(test_signal, W_test, train.components_, 'frobenius', square_root=True)
print('Test error: ', test_error)
