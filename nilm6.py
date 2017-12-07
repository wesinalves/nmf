'''
Author: Wesin Alves
year: 2017

NMF for energy disaggregation
Dataset Ampd
number of days = 7
number of samples by day= 1440

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
appliances_name = ['north_bedroom', 'mastersouth_bedroom', 'basement_plugsligths', 'clothes_dryer','clothes_washer',
				'dining_roomplugs', 'dishwasher','eletronic_workbench', 'security_equipament', 'kitchen_fridge',
				'hvac', 'garage', 'heat_pump', 'hot_water', 'home_office','outside_plug','entertainment',
				'utility_plug','wall_oven']

appliances_filename = ['Electricity_B1E.tab', 'Electricity_B2E.tab', 'Electricity_BME.tab', 'Electricity_CDE.tab',
'Electricity_CWE.tab', 'Electricity_DNE.tab', 'Electricity_DWE.tab', 'Electricity_EBE.tab', 'Electricity_EQE.tab',
'Electricity_FGE.tab', 'Electricity_FRE.tab', 'Electricity_GRE.tab', 'Electricity_HPE.tab', 'Electricity_HTE.tab',
'Electricity_OFE.tab', 'Electricity_OUE.tab', 'Electricity_TVE.tab', 'Electricity_UTE.tab', 'Electricity_WOE.tab']				

k = 0
appliances_value = list()
total = 0
for appliance in appliances_name:
	with open('dataset/{}'.format(appliances_filename[k]), 'rb') as f:
		appliances_value.append(np.loadtxt(f,skiprows=1, delimiter='\t'))
	k = k +1

# create input matrix [1440,7]
# suggestion is crete it in a different file
n_day = 7
n_sample = 1440
attribute = 5 # power

init = 0
aggregate_signal = np.zeros((n_sample,n_day))
test_signal = np.zeros((n_sample,n_day))
#aggregate_signal[:,0] = heat_pump[init:1440*(0+1),attribute] + clothes_dryer[init:1440*(0+1),attribute]
for k in range(n_day):
	for appliance in appliances_value:
		aggregate_signal[:,k] = aggregate_signal[:,k] + appliance[init:1440*(k+1),attribute]
	init = init + 1440

init2 = init
for n in range(n_day):
	for appliance in appliances_value:
		test_signal[:,n] = test_signal[:,n] + appliance[init:init2+1440*(n+1),attribute]
	init = init + 1440



#############
#create model
#############
alpha = 0.012
model = NMF(n_components = len(appliances_value), init = 'random', alpha=alpha, l1_ratio=.83, max_iter=2000)

############
#train step
############

train = model.fit(aggregate_signal)
W_train = train.transform(aggregate_signal)
train_error = _beta_divergence(aggregate_signal,W_train,train.components_, beta = 2, square_root=True)
print('Train error: ', train_error)


##############
#evaluate step
##############
W_test = train.transform(test_signal)
test_error = _beta_divergence(test_signal, W_test, train.components_, beta = 2, square_root=True)
print('Test error: ', test_error)

##############
#Compute PCEC
##############
pcec = 0
total_aggragate  = np.sum(test_signal[:,0])
k = 0
total_pcec = 0
total_pceci = 0
#ground truth
for appliance in appliances_value:
	pcec = np.sum(appliance[init2:init2+1440,5]) / total_aggragate
	wi = W_test[:,k].reshape(n_sample,1)
	hi = train.components_[k,:].reshape(1,n_day)
	x1 = np.multiply(wi,hi)
	pcec_i = np.sum(x1[:,0]) / total_aggragate
	print('appliance: {}, pcec: {}, pcec_i {}'.format(appliances_name[k],pcec, pcec_i))
	k += 1
	total_pcec += pcec
	total_pceci += pcec_i

print('total pcec shoud be 1, given {}'.format(total_pcec))
print('total pcec_i shoud be 1, given {}'.format(total_pceci))