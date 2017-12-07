#load dataset
import numpy as np
import matplotlib.pyplot as plt
with open('dataset/Electricity_WHE.csv', 'rb') as f:
	aggregate_signal = np.loadtxt(f,skiprows=1, delimiter=',')

with open('dataset/Electricity_HPE.csv', 'rb') as f:
	heat_pump = np.loadtxt(f,skiprows=1, delimiter=',')

with open('dataset/Electricity_CDE.csv', 'rb') as f:
	clothes_dryer = np.loadtxt(f,skiprows=1, delimiter=',')

print(aggregate_signal.shape)

total_aggragate  = np.sum(aggregate_signal[1441:2880,5])
total_heat = np.sum(heat_pump[1441:2880,5])
total_clothes = np.sum(clothes_dryer[1441:2880,5])

pcec_heat = total_heat / total_aggragate
pcec_clothes = total_clothes / total_aggragate

print('pcec heat {0} - pcec clothes {1}'.format(pcec_heat,pcec_clothes))

'''
init = 1
for k in range(10):
	#plt.plot(aggregate_signal[init:1440*(k+1),5])
	print(np.sum(aggregate_signal[init:1440*(k+1),5]))
	print('{0} - columns {1}'.format(init,1440*(k+1)))
	init = init + 1440
'''
	

