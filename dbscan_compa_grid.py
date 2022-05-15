#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 14 11:43:50 2022

@author: amartinez
"""
# =============================================================================
# In this script we will dbscan over the data, but using a rectangula grid
# =============================================================================
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.neighbors import KDTree
from kneed import DataGenerator, KneeLocator
from matplotlib.ticker import FormatStrFormatter
import pandas as pd
import sys
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import QTable
import pylab as p
from random import seed
from random import random
import glob
from sklearn.preprocessing import StandardScaler
import os
import spisea
from spisea import synthetic, evolution, atmospheres, reddening, ifmr
from spisea.imf import imf, multiplicity
import astropy.coordinates as ap_coor

# %%

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
plt.rcParams.update({'figure.max_open_warning': 0})# a warniing for matplot lib pop up because so many plots, this turining it of
# %%
cata='/Users/amartinez/Desktop/PhD/Libralato_data/CATALOGS/'
pruebas='/Users/amartinez/Desktop/PhD/Libralato_data/pruebas/'
results='/Users/amartinez/Desktop/PhD/Libralato_data/results/'
gns_ext = '/Users/amartinez/Desktop/PhD/Libralato_data/extinction_maps/'

name='WFC3IR'
# name='ACSWFC'
trimmed_data='yes'
if trimmed_data=='yes':
    pre=''
elif trimmed_data=='no':
    pre='relaxed_'
    
else:
    sys.exit("Have to set trimmed_data to either 'yes' or 'no'")
    
# %%
# We upload galactic center stars, that we will use in the CMD
# catal=np.loadtxt(results+'refined_%s_PM.txt'%(name))
# catal_df=pd.read_csv(pruebas+'%s_refined_with_GNS_partner_mag_K_H.txt'%(name),sep=',',names=['ra','dec','x_c','y_c','mua','dmua','mud','dmud','time','n1','n2','idt','m139','Separation','Ks','H'])

# "'RA_gns','DE_gns','Jmag','Hmag','Ksmag','ra','dec','x_c','y_c','mua','dmua','mud','dmud','time','n1','n2','ID','mul','mub','dmul','dmub','m139','Separation'",
catal=np.loadtxt(results + '%smatch_GNS_and_%s_refined_galactic.txt'%(pre,name))
# Definition of center can: m139 - Ks(libralato and GNS) or H - Ks(GNS and GNS)
center_definition='G_G'#this variable can be L_G or G_G
if center_definition =='L_G':
    valid=np.where(np.isnan(catal[:,4])==False)# This is for the valus than make Ks magnitude valid, but shouldn´t we do the same with the H magnitudes?
    catal=catal[valid]
    center=np.where(catal[:,-2]-catal[:,4]>2.5) # you can choose the way to make the color cut, as they did in libralato or as it is done in GNS
elif center_definition =='G_G':
    valid=np.where((np.isnan(catal[:,3])==False) & (np.isnan(catal[:,4])==False ))
    catal=catal[valid]
    center=np.where(catal[:,3]-catal[:,4]>1.3)
catal=catal[center]
dmu_lim = 0.5
vel_lim = np.where((catal[:,19]<=dmu_lim) & (catal[:,20]<=dmu_lim))
catal=catal[vel_lim]

# 'ra dec x_c  y_c mua dmua mud dmud time n1 n2 ID mul mub dmul dmub '
catal_all = np.loadtxt(cata + '%s_pm_galactic.txt'%(name))


#mul, mub, mua, mud, ra, dec,dmul,dmub, position in GALCEN_TABLE_D.cat 
Ms_all=np.loadtxt(pruebas +'pm_of_Ms_in_%s.txt'%(name))# this are the information (pm, coordinates and ID) for the Ms that remain in the data after triming it 4
# %%
# ra, dec, other things
#Selecting the massive stars to plotting in the xy plot
Ms_ra, Ms_dec = np.loadtxt(cata + 'GALCEN_TABLE_D.cat',usecols=(0,1),unpack = True)

Ms_xy = [int(np.where((Ms_ra[i]==(catal_all[:,0])) & ((Ms_dec[i]==catal_all[:,1])))[0]) for i in range(len(Ms_ra)) if len(np.where((Ms_ra[i]==(catal_all[:,0])) & ((Ms_dec[i]==catal_all[:,1])))[0]) >0]
# %%

fig, ax = plt.subplots(1,1,figsize=(10,10))
ax.scatter(catal[:,7],catal[:,8],color = 'k', alpha = 0.3)

# %%
ra_=catal[:,5]
dec_=catal[:,6]
# Process needed for the trasnformation to galactic coordinates
gal_c = SkyCoord(ra=ra_*u.degree, dec=dec_*u.degree).galactic#you are using frame 'fk5' but maybe it si J2000, right? becouse this are Paco`s coordinates. Try out different frames


t_gal= QTable([gal_c.l,gal_c.b], names=('l','b')) 
fig, ax = plt.subplots(1,1,figsize=(10,10))
t_gal['l'] = t_gal['l'].wrap_at('180d')
ax.invert_xaxis()
ax.scatter(t_gal['l'].value,t_gal['b'].value,color = 'k', alpha = 0.3)


# %%
# line = np.where((catal[:,7] >11000) & (catal[:,7]<12000))
# # %
# xy_selc = catal[line]
# # %
# xy_selc_sort = np.flip(xy_selc[np.argsort(xy_selc[:,8])],0)

# m = (27667.3063-29465.5753)/(13597.558- 11324.3373)
m=-0.80
fig, ax = plt.subplots(1,1,figsize=(10,10))
ax.scatter(catal[:,7],catal[:,8],color = 'k', alpha = 0.3)
for i in np.arange(16000,44000,4000):
    ax.scatter(np.arange(0,25000,10),i + m*np.arange(0,25000,10),color = 'r', alpha = 0.1,s=1)
m1=1
for i in np.arange(-20000,30000,3000):
    ax.scatter(np.arange(0,25000,10),i + m1*np.arange(0,25000,10),color = 'b', alpha = 0.1,s=1)
ax.set_xlim(0,25000)
ax.set_ylim(0,40000)
#   x_c        y_c     
# 11324.3373 29465.5753
# 13597.558  27667.3063
#%%
x = catal[:,7]
yb1 = 5000 + m1*catal[:,7]
yr1 = 9000 + m1*catal[:,7]
yb = 34000 + m*catal[:,7]
yr = 36000 + m*catal[:,7]


# fr = np.interp(catal[:,7],x,yr)
# fb = np.interp(catal[:,7],x,yb)

# c1 = catal[:,8] < fr
# c2 = catal[:,8] > fb
# good = np.where((catal[:,8]>fb) &(catal[:,8]<fr))
good = np.where((catal[:,8]>yb) & (catal[:,8]<yr) & (catal[:,8]>yb1) &(catal[:,8]<yr1))
fig, ax = plt.subplots(1,1,figsize=(10,10))
ax.scatter(catal[:,7],catal[:,8],color = 'k', alpha = 0.3)
# ax.scatter(catal[:,7],catal[:,8],c=(c1&c2), s=1, cmap="summer_r")
ax.scatter(catal[:,7][good],catal[:,8][good],c='g')
ax.plot(x,yr,color ='orange')
ax.plot(x,yb,color ='yellow')
ax.plot(x,yr1,color ='r')
ax.plot(x,yb1,color ='b')

# %%

chunks =0
vacio = 0
colores =['r', 'b', 'g', 'orange','pink','darkblue','yellow','royalblue']
fig, ax = plt.subplots(1,1,figsize=(10,10))
ax.scatter(catal[:,7],catal[:,8],color = 'k', alpha = 0.3)
step=4000
step1=2000
for i in np.arange(16000,44000,step):
    yb = i + m*catal[:,7]
    yr = i+step + m*catal[:,7]
    for j in np.arange(-20000,30000,step1):
        yb1 = j + m1*catal[:,7]
        yr1 = j+step1 + m1*catal[:,7]
        good = np.where((catal[:,8]>yb) & (catal[:,8]<yr) & (catal[:,8]>yb1) &(catal[:,8]<yr1))
        if len(good[0])<1:
            vacio +=1
        ax.scatter(catal[:,7][good],catal[:,8][good],c=random.choice(colores))
# =============================================================================
#         t_gal['l'] = t_gal['l'].wrap_at('180d')
#         ax.invert_xaxis()
#         ax.scatter(t_gal['l'][good].value,t_gal['b'][good].value,c=random.choice(colores), alpha = 0.3)
#         chunks +=1
# =============================================================================
for i in np.arange(16000,44000,step):
    ax.scatter(np.arange(0,25000,10),i + m*np.arange(0,25000,10),color = 'r', alpha = 0.1,s=1)
m1=1
for i in np.arange(-20000,30000,step1):
    ax.scatter(np.arange(0,25000,10),i + m1*np.arange(0,25000,10),color = 'b', alpha = 0.1,s=1)
ax.set_xlim(0,25000)
ax.set_ylim(0,40000)
print(chunks, vacio)
# %%
print(len(np.arange(16000,41000,2000)),len(np.arange(-20000,30000,2000)))






