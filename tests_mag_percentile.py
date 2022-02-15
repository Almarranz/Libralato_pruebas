#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 10:31:49 2022

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
# %%

cata='/Users/amartinez/Desktop/PhD/Libralato_data/CATALOGS/'
#R.A. Dec. X Y μαcosδ σμαcosδ μδ σμδ  time n1 n2 ID
# name='ACSWFC'
name='WFC3IR'
# ra,dec,x_c ,y_c,mua,dmua,mud,dmud, time, n1, n2, idt = np.loadtxt(cata+'GALCEN_%s_PM.cat'%(name),unpack=True)
catal=np.loadtxt(cata+'GALCEN_%s_PM.cat'%(name))
pruebas='/Users/amartinez/Desktop/PhD/Libralato_data/pruebas/'

# %%

mag, rms, qfit, o, RADXS, nf, nu, Localsky, Local_skyrms= np.loadtxt(cata+'GALCEN_%s_GO12915.cat'%(name),unpack=True )
all_ep1=np.loadtxt(cata+'GALCEN_%s_GO12915.cat'%(name),unpack=False)
mag2, rms2, qfit2, o2, RADXS2, nf2, nu2, Localsky2, Local_skyrms2= np.loadtxt(cata+'GALCEN_%s_GO13771.cat'%(name),unpack=True )
all_ep2=np.loadtxt(cata+'GALCEN_%s_GO13771.cat'%(name),unpack=False )

# %%
ep1_test=all_ep1
# %%
fig, ax = plt.subplots(1,1,figsize=(10,10))
n,bins_edges,otro=ax.hist(ep1_test[:,0],bins=np.arange(np.round(min(mag)),np.round(max(mag)+1),1),linewidth=2,edgecolor='black') 
mag_b=np.digitize(ep1_test[:,0], np.arange(np.round(min(mag)),np.round(max(mag)+1),1), right=True)
# %%
all_sum=[]
qfit_valid=[]
for i in range(len(np.arange(np.round(min(mag)),np.round(max(mag)+1),1))):
    mag_binned=np.where(mag_b==i)
    qfit_i=qfit[mag_binned]
    print('%.5f'%(np.percentile(qfit_i,85)),i,len(qfit_i),len(mag_binned[0]))
    perc = np.percentile(qfit_i,85)
    for j in range(len(qfit_i)):
        if qfit_i[j] > 0.6:
            if qfit_i[j] >= perc or qfit_i[j] >= 0.975:
                qfit_valid.append(mag_binned[0][j])
                
    all_sum.append(len(qfit_i))
print(sum(all_sum))
# %% 
c=0
for i in range(len(mag_binned[0])):
    c =+ i
    print(mag_binned[0][i],c)

















