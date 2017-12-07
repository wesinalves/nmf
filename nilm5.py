'''
Author: Wesin Alves
year: 2017

Elasticnet for energy disaggregation
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

dishwasher_train = np.zeros((n_sample,n_day))
dishwasher_test = np.zeros((n_sample,n_day))
init = 0;
for d in range(n_day):
	dishwasher_train[:,n] = appliances_value[1][init:1440*(d+1),attribute]
	init = init + 1440

init2 = init;
for x in range(n_day):
	dishwasher_test[:,x] = appliances_value[1][init:init2+1440*(x+1),attribute]
	init = init + 1440

#############
#create model
#############

###################################################
#ElasticNet
alpha = 0.012
enet = ElasticNet(alpha=alpha, l1_ratio=0.83, max_iter = 2000)

y_pred_enet = enet.fit(aggregate_signal,dishwasher_train).predict(test_signal)
r2_score_enet = r2_score(dishwasher_test,y_pred_enet)
print(enet)
print("r^2 on test data: %f" %r2_score_enet)

disag_error_enet = 0.5*mean_squared_error(dishwasher_test, y_pred_enet)
print("mae on test data: %f" %disag_error_enet)