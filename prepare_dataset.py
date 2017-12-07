#prepare dataset for nilm task
import numpy as np


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
for k in range(n_day):
	for appliance in appliances_value:
		aggregate_signal[:,k] = aggregate_signal[:,k] + appliance[init:1440*(k+1),attribute]
	init = init + 1440

init2 = init
for n in range(n_day):
	for appliance in appliances_value:
		test_signal[:,n] = test_signal[:,n] + appliance[init:init2+1440*(n+1),attribute]
	init = init + 1440

pcec = 0
total_aggragate  = np.sum(aggregate_signal[0:1440,1])
k = 0
total_pcec = 0
for appliance in appliances_value:
	pcec = np.sum(appliance[1441:2881,5]) / total_aggragate
	print('appliance: {}, pcec: {}'.format(appliances_name[k],pcec))
	k += 1
	total_pcec += pcec

print('total pcec shoud be 1, given {}'.format(total_pcec))


'''
str_appliance = ""
for i in range(19):
	str_appliance += "[appliances_value[{0}][0:1440, attribute], appliances_value[{0}][1441:2880, attribute],".format(i)
	print("[appliances_value[{0}][0:1440, attribute], appliances_value[{0}][1441:2880, attribute],".format(i))
'''
W_train = np.zeros((n_sample,38))
w_index = 0
for i in range( len(appliances_value) ):
	W_train[:,w_index] = np.array([appliances_value[i][0:1440, attribute]])
	W_train[:,w_index + 1 ] = np.array([appliances_value[i][1441:2881, attribute]])
	w_index += 2 

print(W_train.shape)
print(W_train)