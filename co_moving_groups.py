#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 11:02:37 2022

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
pruebas='/Users/amartinez/Desktop/PhD/Libralato_data/pruebas/'
results='/Users/amartinez/Desktop/PhD/Libralato_data/results/'
name='WFC3IR'
#ra, dec, ID(in ACSWFC_PM or WFC3IR_PM),Original list, Altervative Id
yso_ra,yso_dec,yso_ID=np.loadtxt(cata+'GALCEN_TABLE_D.cat',unpack=True, usecols=(0,1,2))
tipo=np.loadtxt(cata+'GALCEN_TABLE_D.cat',unpack=True, usecols=(3),dtype='str')
# yso_df=pd.read_csv(cata+'GALCEN_TABLE_D.cat', sep=' ')
# yso=yso_df.to_numpy()


# ra,dec,x_c ,y_c,mua,dmua,mud,dmud, time, n1, n2, idt = np.loadtxt(cata+'GALCEN_%s_PM.cat'%(name),unpack=True)
# catal=np.loadtxt(cata+'GALCEN_%s_PM.cat'%(name))
# catal=np.loadtxt(results+'refined_%s_PM.txt'%(name))
catal_df=pd.read_csv(results+'%s_refined_with GNS_partner_mag_K_H.txt'%(name),sep=',',names=['ra','dec','x_c','y_c','mua','dmua','mud','dmud','time','n1','n2','idt','m139','Separation','Ks','H'])

# mul_mc,mub_mc,dmul_mc,dmub_mc
gal_coor=np.loadtxt(results+'match_GNS_and_WFC3IR_refined_galactic.txt')

catal=catal_df.to_numpy()
valid=np.where(np.isnan(catal[:,14])==False)
catal=catal[valid]
gal_coor=gal_coor[valid]
# =============================================================================
# no_fg=np.where(catal[:,12]-catal[:,14]>2.5)
# catal=catal[no_fg]
# gal_coor=gal_coor[no_fg]
# 
# =============================================================================
# %%
distance=0.009
found=0
missing=0
for i in range(len(yso_ra)):
# for i in range(1,2):    
    print(yso_ra[i])
    index=np.where((catal[:,0]==yso_ra[i]) & (catal[:,1]==yso_dec[i]) )
    if len(index[0]>0):
        print(index[0])
        print(catal[index[0],0],catal[index[0],1])
        group=np.where(np.sqrt((catal[:,0]-catal[index[0],0])**2 + (catal[:,1]-catal[index[0],1])**2)< distance)
        print(len(group[0]))
        fig, ax = plt.subplots(1,1,figsize=(10,10))
        ax.scatter(catal[index[0],0],catal[index[0],1],color='red',s=100)
        ax.scatter(catal[group[0],0],catal[group[0],1])
        # ax.quiver(catal[index[0],0],catal[index[0],1],[catal[index[0],4]],[catal[index[0],6]],alpha=0.2)#this is for the vector on the Ms object in ecuatorial
        # ax.quiver(catal[index[0],0],catal[index[0],1],[gal_coor[index[0],0]],[gal_coor[index[0],1]])#this is for the vector on the Ms object in galactic
        ax.quiver([catal[group[0],0]],[catal[group[0],1]],np.array([catal[group[0],4]])+3.16,np.array([catal[group[0],6]])+5.6,alpha=0.2)
        ax.quiver([catal[group[0],0]],[catal[group[0],1]],np.array([gal_coor[group[0],0]])+6.4,np.array([gal_coor[group[0],1]])+0.22)

        ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        ax.legend(['yso #%s, %s'%(i,tipo[i])],markerscale=1,loc=1,handlelength=1)
        ax.set_xlabel(r'$\mathrm{ra}$') 
        ax.set_ylabel(r'$\mathrm{dec}$') 
        np.savetxt(pruebas+'group_%s_%s.txt'%(i,name),catal[group],fmt='%.7f')
        found +=1
# =============================================================================
#         fig, ax = plt.subplots(1,1,figsize=(10,10))
#         ax.hist(catal[group[0],4],bins='auto') 
#         ax.hist(catal[group[0],6],alpha=0.5,bins='auto') 
#         ax.legend(['mua (yso #%s)'%(i),'mub'],markerscale=1,loc=1,handlelength=1)
# =============================================================================
        
    else:
        print('No mach in %s catalog'%(name))
        missing +=1
    # plt.xlabel(r'$\mathrm{\mu_{a}cosb (mas\ yr^{-1})}$') 
    # plt.ylabel(r'$\mathrm{\mu_{d} (mas\ yr^{-1})}$') 
    
print('Found %s , missing %s'%(found, missing))
# %%

