#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 13:52:00 2022

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
cata='/Users/amartinez/Desktop/PhD/Libralato_data/CATALOGS/'


#R.A. Dec. X Y μαcosδ σμαcosδ μδ σμδ  time n1 n2 ID

# name='ACSWFC'
name='WFC3IR'
ra,dec,x ,y,mua,dmua,mud,dmud, time, n1, n2, idt = np.loadtxt(cata+'GALCEN_%s_PM.cat'%(name),unpack=True)
# VEGAmag, rmsmag, QFIT, o, RADXS, nf, nu, Localsky, Local-skyrms
mag, rms, qfit, o, RADXS, nf, nu, Localsky, Local_skyrms= np.loadtxt(cata+'GALCEN_%s_GO12915.cat'%(name),unpack=True )

#%%
good=np.where((dmua<90))
ra=ra[good]
dec=dec[good]
mua=mua[good]
dmua=dmua[good]
mud=mud[good]
dmud=dmud[good]
time=time[good]
n1=n1[good]
n2=n2[good]
idt=idt[good]

#%%
# Here where are transforming the coordinates fron equatorial to galactic
# I am following the paper  https://arxiv.org/pdf/1306.2945.pdf
#  alpha_G = 192.85948,  delta_G = 27.12825, lNGP = 122.93192, according to Perryman & ESA 1997
alpha_g=192.85948
delta_g = 27.12825
tr=np.deg2rad

C1=np.sin(tr(delta_g))*np.cos(tr(dec))-np.cos(tr(delta_g))*np.sin(tr(dec))*np.cos(tr(ra)-tr(alpha_g))
C2=np.cos(tr(delta_g))*np.sin(tr(ra)-tr(alpha_g))
cosb=np.sqrt(C1**2+C2**2)

mul,mub =zip(*[(1/cosb[i])*np.matmul([[C1[i],C2[i]],[-C2[i],C1[i]]],[mua[i],mud[i]]) for i in range(len(ra))])#zip with the* unzips things
mul=np.array(mul)
mub=np.array(mub)
# =============================================================================
# #Im not sure about if I have to transfr¡orm the uncertainties also in the same way....
# dmul,dmub =zip(*[cosb[i]*np.matmul([[C1[i],C2[i]],[-C2[i],C1[i]]],[dmua[i],dmud[i]]) for i in range(len(ra))])#zip with the* unzips things
# dmul=np.array(dmul)
# dmub=np.array(dmub)
# =============================================================================
# for now Ill just leave the like they are
dmul=dmua
dmub=dmud
#%%
good=np.where((mul<70) & (mul>-70))
ra=ra[good]
dec=dec[good]
mul=mul[good]
dmul=dmul[good]
mub=mub[good]
dmub=dmub[good]
time=time[good]
n1=n1[good]
n2=n2[good]
idt=idt[good]
#%%
perc_dmul= np.percentile(dmul,85)
print(perc_dmul,'yomama')
# lim_dmul=perc_dmul
lim_dmul=1
accu=np.where((abs(dmul)<lim_dmul) & (abs(dmub)<lim_dmul))
#%%
mul=mul[accu]
mub=mub[accu]
dmul=dmul[accu]
dmub=dmub[accu]
time=time[accu]


#%%

fig, ax = plt.subplots(1,1, figsize=(10,10))

# sig_h=sigma_clip(mul,sigma=500,maxiters=20,cenfunc='mean',masked=True)
# mul=mul[sig_h.mask==False]

h=ax.hist(mul,bins='auto',linewidth=2,density=True)


x=[h[1][i]+(h[1][1]-h[1][0])/2 for i in range(len(h[0]))]#middle value for each bin
ax.axvline(np.mean(mul), color='r', linestyle='dashed', linewidth=3)
ax.legend(['List=%s, %s, mean= %.2f, std=%.2f'
                  %(name,len(mul),np.mean(mul),np.std(mul))],fontsize=12,markerscale=0,shadow=True,loc=1,handlelength=-0.0)
ax.set_ylabel('N')
ax.set_xlim(-13,3)
ax.set_xlabel(r'$\mathrm{\mu_{l} (mas\ yr^{-1})}$') 
ax.invert_xaxis()
y=h[0]#height for each bin

#%%
fig, ax = plt.subplots(1,1, figsize=(10,10))

# sig_hb=sigma_clip(mub,sigma=500,maxiters=20,cenfunc='mean',masked=True)
# mub=mub[sig_hb.mask==False]

hb=ax.hist(mub,bins='auto',color='orange',linewidth=2,density=True)


xb=[hb[1][i]+(hb[1][1]-hb[1][0])/2 for i in range(len(hb[0]))]#middle value for each bin
ax.axvline(np.mean(mub), color='r', linestyle='dashed', linewidth=3)
ax.legend(['List=%s, %s, mean= %.2f, std=%.2f'
                  %(name,len(mub),np.mean(mub),np.std(mub))],fontsize=12,markerscale=0,shadow=True,loc=1,handlelength=-0.0)
ax.set_ylabel('N')
ax.set_xlim(-10,10)
ax.set_xlabel(r'$\mathrm{\mu_{b} (mas\ yr^{-1})}$') 
yb=hb[0]#height for each bin
#%%

fig, ax =plt.subplots(1,1,figsize=(10,10))
ax.scatter(mul,mub,color='k',s=1,alpha=0.05)
ax.set_xlim(-13,2)
ax.set_ylim(-10,10)
ax.axvline(0)
ax.axhline(0)
ax.axhline(-0.22)
ax.invert_xaxis()







