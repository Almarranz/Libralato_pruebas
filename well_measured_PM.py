#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 10:13:44 2022

@author: amartinez
"""
import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner
import dynesty
import scipy.integrate as integrate
from astropy.stats import sigma_clip
from astropy.stats import sigma_clipped_stats
from matplotlib.ticker import FormatStrFormatter

from matplotlib import rcParams
rcParams.update({'xtick.major.pad': '7.0'})
rcParams.update({'xtick.major.size': '7.5'})
rcParams.update({'xtick.major.width': '1.5'})
rcParams.update({'xtick.minor.pad': '7.0'})
rcParams.update({'xtick.minor.size': '3.5'})
rcParams.update({'xtick.minor.width': '1.0'})
rcParams.update({'ytick.major.pad': '7.0'})
rcParams.update({'ytick.major.size': '7.5'})
rcParams.update({'ytick.major.width': '1.5'})
rcParams.update({'ytick.minor.pad': '7.0'})
rcParams.update({'ytick.minor.size': '3.5'})
rcParams.update({'ytick.minor.width': '1.0'})
rcParams.update({'font.size': 20})
rcParams.update({'figure.figsize':(10,5)})
rcParams.update({
    "text.usetex": False,
    "font.family": "sans",
    "font.sans-serif": ["Palatino"]})
plt.rcParams["mathtext.fontset"] = 'dejavuserif'
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Palatino']})


# %%

cata='/Users/amartinez/Desktop/PhD/Libralato_data/CATALOGS/'

# name='ACSWFC'
name='WFC3IR'
# ra,dec,x_c ,y_c,mua,dmua,mud,dmud, time, n1, n2, idt = np.loadtxt(cata+'GALCEN_%s_PM.cat'%(name),unpack=True)
catal=np.loadtxt(cata+'GALCEN_%s_PM.cat'%(name))
pruebas='/Users/amartinez/Desktop/PhD/Libralato_data/pruebas/'
# mag, rms, qfit, o, RADXS, nf, nu, Localsky, Local_skyrms
ep1=np.loadtxt(pruebas+'foto_well_mesaured_ep%s_%s.txt'%(name,1))
ep2=np.loadtxt(pruebas+'foto_well_mesaured_ep%s_%s.txt'%(name,2))
mag, rms, qfit, o, RADXS, nf, nu, Localsky, Local_skyrms= np.loadtxt(cata+'GALCEN_%s_GO12915.cat'%(name),unpack=True )
# ep12,ep1_ind,ep2_ind=np.intersect1d(ep1[:,-1],ep2[:,-1], return_indices=True)
# %%
# ep12pm,ep1_indpm,ep2_indpm=np.intersect1d(ep1_ind,catal[:,-1], return_indices=True)
# %%
# ra,dec,x_c ,y_c,mua,dmua,mud,dmud, time, n1, n2, idt
index=ep1[:,-1].astype(int)

pm_wmp=catal[index]#pm_wmp stands for proper motion well mesaured photometry
# ep1=ep1[ep1_indpm]
v_valid=np.where(pm_wmp[:,4] < 90)
pm_wmp=pm_wmp[v_valid]
ep1=ep1[v_valid]
# %%
import random
ra_valu=[]
for r in range(len(ep1[:,0])):
    ep1[r,0]= ep1[r,0] + random.uniform(0.000009,0.00004)
    pm_wmp[r,5]=pm_wmp[r,5]+ random.uniform(0.000009,0.00004)
    ra_valu.append(random.uniform(0.0005,0.0001))
np.savetxt(pruebas+'mag_dmu_test.txt',np.array([ep1[:,0],pm_wmp[:,5]]).T,fmt='%.8f')    
arr=ep1[:,0]
fig, ax = plt.subplots(1,1, figsize=(10,10))

ax.scatter(arr,pm_wmp[:,5],s=1,color='k',alpha=0.05)
ax.set_ylim(0,10)


# %%
both=list(zip(ep1[:,0],pm_wmp[:,5]))

res = sorted(both, key = lambda x: x[0])

# %%

both_all=list(zip(mag,catal[:,5]))

res_all = sorted(both, key = lambda x: x[0])

# =============================================================================
# count=0
# for i in range(len(id_sir)+1):
#     # print(i)
#     try:
#          if id_sir[i+1]==id_sir[i]:
#             count+=1
#     except:
#         print('nada')
# print(count)
# 
# =============================================================================


















