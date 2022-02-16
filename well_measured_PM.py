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
# mag, rms, qfit, o, RADXS, nf, nu, Localsky, Local_skyrms = np.loadtxt(cata+'GALCEN_%s_GO12915.cat'%(name),unpack=True)
ep1_fix=np.loadtxt(pruebas+'foto_well_mesaured_ep%s_%s.txt'%(name,1))
ep2_fix=np.loadtxt(pruebas+'foto_well_mesaured_ep%s_%s.txt'%(name,2))
mag, rms, qfit, o, RADXS, nf, nu, Localsky, Local_skyrms= np.loadtxt(cata+'GALCEN_%s_GO12915.cat'%(name),unpack=True )



# %%
ep12,ep12_ind,ep21_ind=np.intersect1d(ep1_fix[:,-1],ep2_fix[:,-1], return_indices=True)
# I can not make this work. something look bad when I use  the index for ep12 for selecting stars 
# %%
# ra,dec,x_c ,y_c,mua,dmua,mud,dmud, time, n1, n2, idt
ep1=ep1_fix
ep2=ep2_fix
index1=ep1[:,-1].astype(int)
pm_wmp=catal[index1]#pm_wmp stands for proper motion well mesaured photometry
# ep1=ep1[ep12_ind]
# ep1=ep1[ep1_indpm]
v_valid=np.where((pm_wmp[:,5]<90) & (pm_wmp[:,4]<70) )
pm_wmp=pm_wmp[v_valid]
ep1=ep1[v_valid]

fig, ax = plt.subplots(1,1, figsize=(10,10))



ax.scatter(ep1[:,0],pm_wmp[:,5],s=1,color='red',alpha=1)
# ax.scatter(mag,catal[:,5],s=1,color='k',alpha=1)
ax.set_ylim(0,10)
ax.set_xlim(12,24)
# %%
index2=ep2[:,-1].astype(int)
pm_wmp=catal[index2]#pm_wmp stands for proper motion well mesaured photometry
# ep1=ep1[ep12_ind]
# ep1=ep1[ep1_indpm]
v_valid=np.where((pm_wmp[:,5]<90) & (pm_wmp[:,4]<70) )
pm_wmp=pm_wmp[v_valid]
ep2=ep2[v_valid]

fig, ax = plt.subplots(1,1, figsize=(10,10))



ax.scatter(ep2[:,0],pm_wmp[:,5],s=1,color='red',alpha=1)
# ax.scatter(mag,catal[:,5],s=1,color='k',alpha=1)
ax.set_ylim(0,10)
ax.set_xlim(12,24)

# %%

ind=ep21_ind.astype(int)

ep2=ep2_fix[ind]
pm_wmp=catal[ind]


fig, ax = plt.subplots(1,1, figsize=(10,10))



ax.scatter(ep2[:,0],pm_wmp[:,5],s=1,color='red',alpha=1)
# ax.scatter(mag,catal[:,5],s=1,color='k',alpha=1)
ax.set_ylim(0,10)
ax.set_xlim(12,24)

# %%
ep12_s=sorted(ep12_ind)


both = set(ep1_fix[:,-1]).intersection(ep2_fix[:,-1])
# %%
indices_A = [list(ep1_fix[:,-1]).index(x) for x in both]






