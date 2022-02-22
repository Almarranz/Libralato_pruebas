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
from datetime import datetime
cata='/Users/amartinez/Desktop/PhD/Libralato_data/CATALOGS/'
pruebas='/Users/amartinez/Desktop/PhD/Libralato_data/pruebas/'
results='/Users/amartinez/Desktop/PhD/Libralato_data/results/'

# 'RA_gns','DE_gns','Jmag','Hmag','Ksmag','ra','dec','x_c','y_c','mua','dmua','mud','dmud','time','n1','n2','idt','m139','Separation'
# name='ACSWFC'
name='WFC3IR'
option=2
if option==1:
    #Option 1:  transformation of selected stars
    df = pd.read_csv(results+'match_GNS_and_%s_refined.txt'%(name),sep=',',names=['RA_gns','DE_gns','Jmag','Hmag','Ksmag','ra','dec','x_c','y_c','mua','dmua','mud','dmud','time','n1','n2','idt','m139','Separation'])
    df_np=df.to_numpy()
    
    ra=df_np[:,5]
    dec=df_np[:,6]
    mua=df_np[:,9]
    dmua=df_np[:,10]
    mud=df_np[:,11]
    dmud=df_np[:,12]
    
elif option==2:
    #Option 2: transformation of the whole catalogue
    # ra,dec,x_c ,y_c,mua,dmua,mud,dmud, time, n1, n2, idt
    ra,dec,x_c ,y_c,mua,dmua,mud,dmud, time, n1, n2, idt = np.loadtxt(cata+'GALCEN_%s_PM.cat'%(name),unpack=True)

# %%
alpha_g=192.85948
delta_g = 27.12825
tr=np.deg2rad
mul_mc=[]
mub_mc=[]
dmul_mc=[]
dmub_mc=[]

for i in range(10):
    mul_mean=[]
    mub_mean=[]
    if dmua[i]<90:
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
            mub_mean.append(mub_i)
        if i%10000 == 0:
            print(datetime.now())
            print('just did star #%s'%(i))
        # print('Originlas:',mua[i],dmua[i])
        # print(np.mean(mul_mean),np.std(mul_mean))  
        mul_mc.append(np.mean(mul_mean))
        dmul_mc.append(np.std(mul_mean))
        mub_mc.append(np.mean(mub_mean))
        dmub_mc.append(np.std(mub_mean))
    else:
        C1=np.sin(tr(delta_g))*np.cos(tr(dec[i]))-np.cos(tr(delta_g))*np.sin(tr(dec[i]))*np.cos(tr(ra[i])-tr(alpha_g))
        C2=np.cos(tr(delta_g))*np.sin(tr(ra[i])-tr(alpha_g))
        cosb=np.sqrt(C1**2+C2**2)
        mul_i,mub_i =(1/cosb)*np.matmul([[C1,C2],[-C2,C1]],[mua[i],mud[i]])#zip with the* unzips things
        mul_mc.append(mul_i)
        dmul_mc.append(dmua[i])
        mub_mc.append(mub_i)
        dmub_mc.append(dmua[i])
        if i%10000 == 0:
            print(datetime.now())
            print('just did star #%s'%(i))


# %%

if option==1:
    np.savetxt(pruebas+'match_GNS_and_%s_refined_galactic.txt'%(name),np.array([mul_mc,mub_mc,dmul_mc,dmub_mc]).T,fmt='%.7f',header='mul_mc,mub_mc,dmul_mc,dmub_mc')
elif option==2:
    np.savetxt(pruebas+'GALCEN_%s_PM_galactic.txt'%(name),np.array([mul_mc,mub_mc,dmul_mc,dmub_mc]).T,fmt='%.7f',header='mul_mc,mub_mc,dmul_mc,dmub_mc')














