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
alpha_g=192.85948
delta_g = 27.12825
tr=np.deg2rad
ra=266.46036
dec=-28.82440

mua=-1.45
mud=-2.68

C1=np.sin(tr(delta_g))*np.cos(tr(dec))-np.cos(tr(delta_g))*np.sin(tr(dec))*np.cos(tr(ra)-tr(alpha_g))
C2=np.cos(tr(delta_g))*np.sin(tr(ra)-tr(alpha_g))
cosb=np.sqrt(C1**2+C2**2)

i=0
mul,mub =(1/cosb)*np.matmul([[C1,C2],[-C2,C1]],[mua,mud])

print(mul,mub)
# %%

alpha_g=192.86
delta_g = 27.13
tr=np.deg2rad
 
ra=266.56
dec=-28.83

mua=-1.19
mud=-2.66

C1=np.sin(tr(delta_g))*np.cos(tr(dec))-np.cos(tr(delta_g))*np.sin(tr(dec))*np.cos(tr(ra)-tr(alpha_g))
C2=np.cos(tr(delta_g))*np.sin(tr(ra)-tr(alpha_g))
cosb=np.sqrt(C1**2+C2**2)

i=0
mul,mub =(1/cosb)*np.matmul([[C1,C2],[-C2,C1]],[mua,mud])

print(mul,mub)
# %%

cata='/Users/amartinez/Desktop/PhD/Libralato_data/CATALOGS/'
#R.A. Dec. X Y μαcosδ σμαcosδ μδ σμδ  time n1 n2 ID
# name='ACSWFC'
name='WFC3IR'
# ra,dec,x_c ,y_c,mua,dmua,mud,dmud, time, n1, n2, idt = np.loadtxt(cata+'GALCEN_%s_PM.cat'%(name),unpack=True)
catal=np.loadtxt(cata+'GALCEN_%s_PM.cat'%(name))

# %%
print(mua)
mua=np.array(mua)
mub=np.array(mub)

mua=mua.astype(float)
mub=mub.astype(float)

# %%
mat=np.array([[mua],[mub]])
cov_ad=np.cov(mat.astype(float))


# %%
x = np.array([[0.2, 2], [1, 1], [2, 0]]).T

xcov=np.cov(x)
print(xcov)
# %%
mat=np.array([catal[:,4],catal[:,6]])
cov_ad=np.cov(mat)
# %%
print(catal[:,3])




