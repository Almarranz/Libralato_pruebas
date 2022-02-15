#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 13:42:16 2022

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
idt=catal[:,-1]

fit_g=np.percentile(qfit,85)#(a)
rms_g=np.percentile(rms,85)#(b)
ratio=nu/nf#(c)
# o<1(d)
rds_abs=np.absolute(RADXS)#(e)
# (f) their flux within the PSF fitting radius is at least 3σ above the local sky. For
# dont know how to implement f

good1=np.where((qfit > 0.975) & (rms<0.4244) & (ratio>0.5) & 
               (ratio > 0.5) & (o<1) & (rds_abs<0.1))

# %%
idt2=catal[:,-1]

fit_g2=np.percentile(qfit2,85)#(a)
rms_g2=np.percentile(rms2,85)#(b)
nf2=nf2[np.where(nf2 != 0)]
nu2=nu2[np.where(nf2 != 0)]

ratio2=nu2/nf2#(c)
# o<1(d)
rds_abs2=np.absolute(RADXS2)#(e)
# (f) their flux within the PSF fitting radius is at least 3σ above the local sky. For
# dont know how to implement f

good2=np.where((qfit2[np.where(nf2 != 0)] > 0.975) & (rms2[np.where(nf2 != 0)]<0.4142) & (ratio2[np.where(nf2 != 0)]>0.5) & 
               (ratio2[np.where(nf2 != 0)] > 0.5) & (o2[np.where(nf2 != 0)] < 1) & (rds_abs2[np.where(nf2 != 0)] < 0.1))


# %%
idt=idt[good1]
idt2=idt2[good2]
good_both=[]

# %%
ep12,ep1_ind,ep2_ind=np.intersect1d(idt,idt2, return_indices=True)
# %% 
np.savetxt(pruebas+'well_mesaured_ep1_%s.txt'%(name),ep1_ind,header='index for the stars that fullfil the well_mesaured critreia from Libralato et al. 2021')
np.savetxt(pruebas+'well_mesaured_ep2_%s.txt'%(name),ep2_ind,header='index for the stars that fullfil the well_mesaured critreia from Libralato et al. 2021')

# %%
# ra,dec,x_c ,y_c,mua,dmua,mud,dmud, time, n1, n2, idt = np.loadtxt(cata+'GALCEN_%s_PM.cat'%(name),unpack=True)
catal1=catal#PM catalog
catal1=catal1[ep1_ind]
good_pm=np.where((catal1[:,5]<5))
fig,ax=plt.subplots(1,1,figsize=(10,10))


# bright=np.where((mag>16)&(mag<20))
ax.scatter(mag[ep1_ind][good_pm],catal1[:,5][good_pm],s=0.1,color='k')
# ax.scatter(mag[ep1_ind][bright],catal[ep1_ind][:,5][bright],zorder=3)
# ax.scatter(mag,catal[:,5],s=0.1)
ax.grid()
plt.ylabel(r'$\mathrm{\mu_{l} (mas\ yr^{-1})}$') 
plt.xlabel('$m_{F139}$') 

ax.set_ylim(0,10)

# %%
# ra,dec,x_c ,y_c,mua,dmua,mud,dmud, time, n1, n2, idt = np.loadtxt(cata+'GALCEN_%s_PM.cat'%(name),unpack=True)
catal1=catal#PM catalog
catal1=catal1[ep2_ind]
good_pm=np.where((catal1[:,5]<5))
fig,ax=plt.subplots(1,1,figsize=(10,10))


# bright=np.where((mag>16)&(mag<20))
ax.scatter(mag[ep2_ind][good_pm],catal1[:,5][good_pm],s=0.1,color='k')
# ax.scatter(mag[ep1_ind][bright],catal[ep1_ind][:,5][bright],zorder=3)
# ax.scatter(mag,catal[:,5],s=0.1)
ax.grid()
plt.ylabel(r'$\mathrm{\mu_{l} (mas\ yr^{-1})}$') 
plt.xlabel('$m_{F139}$') 

ax.set_ylim(0,10)



per=np.percentile(catal1[:,5][good_pm],85)


















