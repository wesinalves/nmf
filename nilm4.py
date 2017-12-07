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
import sklearn.decomposition as decomp
from sklearn.metrics import mean_squared_error

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
n_devices = 2
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


# needs to cretate W_train and W_test
W_train = np.zeros((n_sample,len(appliances_value)*2))
W_test = np.zeros((n_sample,len(appliances_value)*2))
wtr_index = 0
wts_index = 0
for i in range( len(appliances_value) ):
	W_train[:,wtr_index] = np.array([appliances_value[i][0:1440, attribute]])
	W_train[:,wtr_index + 1 ] = np.array([appliances_value[i][1441:2881, attribute]])
	wtr_index += 2
for i in range( len(appliances_value) ):
	W_test[:,wts_index] = np.array([appliances_value[i][0:1440, attribute]])
	W_test[:,wts_index + 1 ] = np.array([appliances_value[i][1441:2881, attribute]])
	wts_index += 2

H_init = W_train.T
aggregate_signal = aggregate_signal.T
test_signal = test_signal.T
#############
#create model
#############

W,H,n_iter = decomp.non_negative_factorization(aggregate_signal, H=H_init,
	regularization='transformation', n_components=len(appliances_value)*2, init='custom', update_H=False, max_iter=2000)
print(W.shape)
print(H.shape)

############
#train step
############

train_error = _beta_divergence(aggregate_signal,W,H, beta=2, square_root=True)/(2*len(appliances_value))
print('Train error: ', train_error)


##############
#evaluate step
##############
test_error = _beta_divergence(test_signal, W, H, beta=2, square_root=True)/(2*len(appliances_value))
print('Test error: ', test_error)
