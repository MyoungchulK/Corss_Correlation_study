import numpy as np
import os, sys
from matplotlib import pyplot as plt

# custom lib
import correlator_def as corr_def

print('Different correlation method checking')
print('It will generate,')
print('1) example 2 WF')
print('2) numpy correlation')
print('3) numpy correlation w/ bias normalization')
print('4) numpy correlation w/ unbias normalization')
print('5) unbias normalization factor')

#load data
wf1 = np.loadtxt('example_interpolated_wf_A2_R1449_E452_Ch0.txt')
wf2 = np.loadtxt('example_interpolated_wf_A2_R1449_E452_Ch3.txt')

# shape info
# wf[:,0] -> time(ns)
# wf[:,1] -> volt(mV)

# wf
p_wf1 = wf1[:,1]
p_wf2 = wf2[:,1]

# wf length
p_time_len = len(p_wf1)

# time width
t_width = wf1[1,0] - wf1[0,0]

#lag
lag = corr_def.lag_pad_maker(p_time_len, t_width)

# normal correlation
corr = corr_def.cross_correlation(p_wf1, p_wf2)

# correlation with bias normalization
corr_nor = corr_def.cross_correlation_w_bias_normalization(p_wf1, p_wf2, p_time_len)

# correaltion with unbias normalization
corr_new_nor, corr01 = corr_def.cross_correlation_w_unbias_normalization(p_wf1, p_wf2, p_time_len)

# checking by plot
# wf1 and wf2
fig, ax = plt.subplots(figsize=(12, 6))
plt.ylabel(r'Amplitude [ $mV$ ]', fontsize=25)
plt.xlabel(r'Time [ $ns$ ]', fontsize=25)
plt.grid()
plt.tick_params(axis='x', labelsize=20)
plt.tick_params(axis='y', labelsize=20)
plt.title(r'WF', y=1.02,fontsize=15)

plt.plot(wf1[:,0],wf1[:,1],'-',lw=2,color='red',alpha=0.7,label = 'Ch.0')
plt.plot(wf2[:,0],wf2[:,1],'-',lw=2,color='blue',alpha=0.7, label = 'Ch.3')

plt.legend(loc='best',numpoints = 1 ,fontsize=18)
#plt.show()
fig.savefig('wf.png',bbox_inches='tight')#,dpi=100)
plt.close()

#correlation
fig, ax = plt.subplots(figsize=(12, 6))
plt.ylabel(r'Correlation', fontsize=25)
plt.xlabel(r'Lag [ $ns$ ]', fontsize=25)
plt.grid()
plt.tick_params(axis='x', labelsize=20)
plt.tick_params(axis='y', labelsize=20)
plt.title(r'Correlation', y=1.02,fontsize=15)

plt.plot(lag,corr,'-',lw=2,color='red',alpha=0.7)

#plt.show()
fig.savefig('corr.png',bbox_inches='tight')#,dpi=100)
plt.close()

#normalized correlation
fig, ax = plt.subplots(figsize=(12, 6))
plt.ylabel(r'Correlation', fontsize=25)
plt.xlabel(r'Lag [ $ns$ ]', fontsize=25)
plt.grid()
plt.tick_params(axis='x', labelsize=20)
plt.tick_params(axis='y', labelsize=20)
plt.title(r'Normalizaed(bias) Correlation', y=1.02,fontsize=15)

plt.plot(lag,corr_nor,'-',lw=2,color='red',alpha=0.7)

#plt.show()
fig.savefig('corr_nor.png',bbox_inches='tight')#,dpi=100)
plt.close()

#new normalized correlation
fig, ax = plt.subplots(figsize=(12, 6))
plt.ylabel(r'Correlation', fontsize=25)
plt.xlabel(r'Lag [ $ns$ ]', fontsize=25)
plt.grid()
plt.tick_params(axis='x', labelsize=20)
plt.tick_params(axis='y', labelsize=20)
plt.title(r'Normalizaed(unbias) Correlation', y=1.02,fontsize=15)

plt.plot(lag,corr_new_nor,'-',lw=2,color='red',alpha=0.7)

#plt.show()
fig.savefig('corr_new_nor.png',bbox_inches='tight')#,dpi=100)
plt.close()

#new normalization factor
fig, ax = plt.subplots(figsize=(12, 6))
plt.ylabel(r'Correlation', fontsize=25)
plt.xlabel(r'Lag [ $ns$ ]', fontsize=25)
plt.grid()
plt.tick_params(axis='x', labelsize=20)
plt.tick_params(axis='y', labelsize=20)
plt.title(r'Unbias Normalization Factor', y=1.02,fontsize=15)

plt.plot(lag,corr01,'-',lw=2,color='red',alpha=0.7)

#plt.show()
fig.savefig('new_nor_factor.png',bbox_inches='tight')#,dpi=100)
plt.close()

print('Done!')













