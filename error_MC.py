#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 11:45:48 2022

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
import pandas as pd
import random
ata='/Users/amartinez/Desktop/PhD/Libralato_data/CATALOGS/'
pruebas='/Users/amartinez/Desktop/PhD/Libralato_data/pruebas/'


# 'RA_gns','DE_gns','Jmag','Hmag','Ksmag','ra','dec','x_c','y_c','mua','dmua','mud','dmud','time','n1','n2','idt','m139','Separation'
# name='ACSWFC'
name='WFC3IR'
df = pd.read_csv(pruebas+'match_GNS_and_%s_refined.txt'%(name),sep=',',names=['RA_gns','DE_gns','Jmag','Hmag','Ksmag','ra','dec','x_c','y_c','mua','dmua','mud','dmud','time','n1','n2','idt','m139','Separation'])
df_np=df.to_numpy()
# %%
ra=df_np[:,5]
dec=df_np[:,6]
mua=df_np[:,9]
dmua=df_np[:,10]
mud=df_np[:,11]
dmud=df_np[:,12]
# %%
print(mua[0])
#%%
# Here where are transforming the coordinates fron equatorial to galactic
# I am following the paper  https://arxiv.org/pdf/1306.2945.pdf
#  alpha_G = 192.85948,  delta_G = 27.12825, lNGP = 122.93192, according to Perryman & ESA 1997
# mua=mua[0:2]
# mud=mud[0:2]
# dmua=dmua[0:2]
# dmud=dmud[0:2]
# ra=ra[0:2]
# dec=dec[0:2]

# %%
alpha_g=192.85948
delta_g = 27.12825
tr=np.deg2rad
mul_mc=[]
mub_mc=[]
dmul_mc=[]
dmub_mc=[]
# C1=np.sin(tr(delta_g))*np.cos(tr(dec))-np.cos(tr(delta_g))*np.sin(tr(dec))*np.cos(tr(ra)-tr(alpha_g))
# C2=np.cos(tr(delta_g))*np.sin(tr(ra)-tr(alpha_g))
# cosb=np.sqrt(C1**2+C2**2)
for i in range(len(ra)):
    mul_mean=[]
    mub_mean=[]
    for j in range(10000):
        # print('Originlas:',mua[i],mud[i])
        mua_r=random.uniform(mua[i]-dmua[i],mua[i]+dmua[i])
        mud_r=random.uniform(mud[i]-dmud[i],mud[i]+dmud[i])
        # print('Plus minus',mua_r,mud_r)
        C1=np.sin(tr(delta_g))*np.cos(tr(dec[i]))-np.cos(tr(delta_g))*np.sin(tr(dec[i]))*np.cos(tr(ra[i])-tr(alpha_g))
        C2=np.cos(tr(delta_g))*np.sin(tr(ra[i])-tr(alpha_g))
        cosb=np.sqrt(C1**2+C2**2)
        mul_i,mub_i =(1/cosb)*np.matmul([[C1,C2],[-C2,C1]],[mua_r,mud_r])#zip with the* unzips things
        mul_mean.append(mul_i)
        mub_mean.append(mul_i)
    if i%10000 == 0:
        print('just did star #%s'%(i))
    # print('Originlas:',mua[i],dmua[i])
    # print(np.mean(mul_mean),np.std(mul_mean))  
    mul_mc.append(np.mean(mul_mean))
    dmul_mc.append(np.std(mul_mean))
    mub_mc.append(np.mean(mub_mean))
    dmub_mc.append(np.std(mub_mean))
# for r in range(len(ep1[:,0])):
#     ep1[r,0]= ep1[r,0] + random.uniform(0.000009,0.00004)
#     pm_wmp[r,5]=pm_wmp[r,5]+ random.uniform(0.000009,0.00004)
#     ra_valu.append(random.uniform(0.0005,0.0001))

# %%


fig, ax= plt.subplots(1,1,figsize=(10,10))
ax.hist(mul_mean,bins='auto')
















