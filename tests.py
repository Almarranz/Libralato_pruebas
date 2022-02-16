#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 12:25:39 2022

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


#%%
# Here where are transforming the coordinates fron equatorial to galactic
# I am following the paper  https://arxiv.org/pdf/1306.2945.pdf
#  alpha_G = 192.85948,  delta_G = 27.12825, lNGP = 122.93192, according to Perryman & ESA 1997
# =============================================================================
# alpha_g=192.85948
# delta_g = 27.12825
# tr=np.deg2rad
# ra=266.46036
# dec=-28.82440
# 
# mua=-1.45
# mud=-2.68
# 
# C1=np.sin(tr(delta_g))*np.cos(tr(dec))-np.cos(tr(delta_g))*np.sin(tr(dec))*np.cos(tr(ra)-tr(alpha_g))
# C2=np.cos(tr(delta_g))*np.sin(tr(ra)-tr(alpha_g))
# cosb=np.sqrt(C1**2+C2**2)
# 
# i=0
# mul,mub =(1/cosb)*np.matmul([[C1,C2],[-C2,C1]],[mua,mud])
# 
# print(mul,mub)
# =============================================================================
# %%

# =============================================================================
# alpha_g=192.86
# delta_g = 27.13
# tr=np.deg2rad
#  
# ra=266.56
# dec=-28.83
# 
# mua=-1.19
# mud=-2.66
# 
# C1=np.sin(tr(delta_g))*np.cos(tr(dec))-np.cos(tr(delta_g))*np.sin(tr(dec))*np.cos(tr(ra)-tr(alpha_g))
# C2=np.cos(tr(delta_g))*np.sin(tr(ra)-tr(alpha_g))
# cosb=np.sqrt(C1**2+C2**2)
# 
# i=0
# mul,mub =(1/cosb)*np.matmul([[C1,C2],[-C2,C1]],[mua,mud])
# 
# print(mul,mub)
# =============================================================================
# %%
# ra,dec,x_c ,y_c,mua,dmua,mud,dmud, time, n1, n2, idt
cata='/Users/amartinez/Desktop/PhD/Libralato_data/CATALOGS/'
#R.A. Dec. X Y μαcosδ σμαcosδ μδ σμδ  time n1 n2 ID
# name='ACSWFC'
name='WFC3IR'
# ra,dec,x_c ,y_c,mua,dmua,mud,dmud, time, n1, n2, idt = np.loadtxt(cata+'GALCEN_%s_PM.cat'%(name),unpack=True)
catal=np.loadtxt(cata+'GALCEN_%s_PM.cat'%(name))
pruebas='/Users/amartinez/Desktop/PhD/Libralato_data/pruebas/'


mua=catal[:,4]
mud=catal[:,6]
# %%
# mat=np.array([catal[:,4],catal[:,6]])
mat_ad=np.array([mua,mud])

cov_ad=np.cov(mat_ad)
# %%
print(catal[:,3]) 
# %%
# P=np.array([[C1,C2],[-C2,C1]])
# Pt=P.T

# %%
alpha_g=192.86
delta_g = 27.13
tr=np.deg2rad
ra=catal[:,0]
dec=catal[:,1]
C1=np.sin(tr(delta_g))*np.cos(tr(dec))-np.cos(tr(delta_g))*np.sin(tr(dec))*np.cos(tr(ra)-tr(alpha_g))
C2=np.cos(tr(delta_g))*np.sin(tr(ra)-tr(alpha_g))
cosb=np.sqrt(C1**2+C2**2)
cov_gl_all=np.empty([len(ra),2,2])

for i in range(len(ra)):
    P=(1/cosb[i])*np.array([[C1[i],C2[i]],[-C2[i],C1[i]]])
    cov_gl=np.linalg.multi_dot([P,cov_ad,P.T]) 
    cov_gl_all[i,:,:]=cov_gl
    

# %% 
# =============================================================================
# Selction of well mesuared star for WFC3 according with the criteria 
# followed by libralato 2021.
# Note (1): Stars are ordered as in GALCEN_WFC3IR_PM.cat
#      (2): Stars measured in only image have a magnitude rms equal to 9.99 mag
#      (3): Stars with rms_mag, qfit, o, radxs, n_f and n_u equal to 0 are saturated in the majoirity of the images in which it was found
# =============================================================================
name='WFC3IR'
mag, rms, qfit, o, RADXS, nf, nu, Localsky, Local_skyrms= np.loadtxt(cata+'GALCEN_%s_GO12915.cat'%(name),unpack=True )
all_ep1=np.loadtxt(cata+'GALCEN_%s_GO12915.cat'%(name),unpack=False)
mag2, rms2, qfit2, o2, RADXS2, nf2, nu2, Localsky2, Local_skyrms2= np.loadtxt(cata+'GALCEN_%s_GO13771.cat'%(name),unpack=True )
all_ep2=np.loadtxt(cata+'GALCEN_%s_GO13771.cat'%(name),unpack=False )

# %% We are follow here the criteria for well measured stars from libralato et all 2021 (section 2)


fig, ax = plt.subplots(1,1, figsize=(10,10))

ax.scatter(mag,catal[:,5],s=0.1,color='k',alpha=0.1)
plt.xlim(21.75,22.25)
ax.set_ylim(0,10)

# %%

# %%
m,d=np.loadtxt(pruebas+'mag_dmu_test.txt',unpack=True)
fig, ax = plt.subplots(1,1, figsize=(10,10))

ax.scatter(m,d,s=1,color='k',alpha=0.05)
ax.set_ylim(0,10)

#%%
print(m[1000])

plt.scatter(m,d,s=0.01,zorder=3)
plt.scatter(mag,catal[:,5],s=0.01)

plt.ylim(0,10)
plt.xlim(21.75,22.25)

# %%
res=list(zip(m,d))
res_s=sorted(res, key = lambda x: x[0])
# %%
for  i in range(len(res_s)):
    if res_s[i][0]>22 and res_s[i][0]<22.1:
        print(res_s[i])
# %%
print(res_s[0][0])
# %%
a1=[8,3,6,7,10,9.233333]
a2=[1,19,6,9.233,8,11]
a1=np.array(a1)

aa,a12,a21=np.intersect1d(a1, a2,return_indices=True)

# %%
a12=a12.astype(int)
print(a1[a12])











