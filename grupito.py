#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 13:24:27 2022

@author: amartinez
"""

# in this script we are going to check the proper motions of a little supocious group of MS in libralato catalog of massive stars
#Well, after the whole morning working of this, you just discoverd the Arches cluster...
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import pandas as pd
from astropy.coordinates import match_coordinates_sky
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.table import QTable
from matplotlib import rcParams
import os
import glob
import sys
# %% for plotting
from sklearn.preprocessing import StandardScaler
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
# 'ra dec x_c  y_c mua dmua mud dmud  time  n1  n2 ID mul mub dmul dmub
gr = np.loadtxt(pruebas +'ms_grupito.txt')
gns = pd.read_csv(cata + 'GNS_central.csv')

# %%
gns_coor=SkyCoord(ra = gns['_RAJ2000'].values*u.degree, dec = gns['_DEJ2000'].values*u.degree)
gr_coor= SkyCoord(ra=gr[:,0]*u.degree, dec = gr[:,1]*u.degree)

# %%
idx = gr_coor.match_to_catalog_sky(gns_coor)
# %%
valid=np.where(idx[1] < 0.5*u.arcsec)
idx_v=idx[0][valid]
# %%
gr_valid=gr[valid]
gns_np=gns.to_numpy()
gns_valid= gns_np[idx[0][valid]]

# %%
fix, ax = plt.subplots(1,1,figsize=(10,10))
ax.scatter(gns_valid[:,0],gns_valid[:,1])
ax.scatter(gr_valid[:,0],gr_valid[:,1],s=4)
#%%

# %%
# count=0
# for i in range(len(valid[0])):
#     c1=SkyCoord(ra = gns_valid[i,0]*u.degree, dec=gns_valid[i,1]*u.degree)
#     c2=SkyCoord(ra=gr_valid[i,0]*u.degree,dec=gr_valid[i,1]*u.degree)
#     sep = c1.separation(c2).to('arcsec')
#     count +=1
#     print(sep, count)
# print(count)
# np.savetxt(pruebas + 'grupin_gns.txt',gns_valid[350:352,:],fmt='%.7f')
# np.savetxt(pruebas + 'gruppin_lib.txt',gr[350:352,:],fmt='%.7f')
# %%
gr_all=np.c_[gr_valid,gns_valid[:,18],gns_valid[:,20],gns_valid[:,22]]
center=np.where(gns_valid[:,20]-gns_valid[:,22]>1.3)
gr_center=gr_all[center]
good=np.where(gr_center[:,15]<90)
gr_good=gr_center[good]
# %%This plots the good star remain in the data, most of them happen to be the ones consider as part of the archer cluster
# we can study them to learn what a reminiscence of a cluster looks like
# if this actually a reminiscence of the arches cluster??
fix, ax = plt.subplots(1,1,figsize=(10,10))
# ax.scatter(gns_valid[:,0],gns_valid[:,1])
for c in range(len(gr_good)):
    a=gr_good[c,0]
    b=gr_good[c,1]
    ax.text(a,b+0.002,'%.3f'%(gr_good[c,-2]-gr_good[c,-1]))
    print(a,b)
ax.scatter(gr_good[:,0],gr_good[:,1],s=4)
ax.quiver(gr_good[:,0],gr_good[:,1],gr_good[:,12],gr_good[:,13])
# %%
colors=[]
for i in range(len(gr_good)):
    colors.append(gr_good[i,-2]-gr_good[i,-1])
print(max(colors)-min(colors))
# %%

fix, ax = plt.subplots(1,1,figsize=(10,10))


ax.scatter(gr_center[:,12],gr_center[:,13],s=10)
ax.scatter(gr_good[:,12],gr_good[:,13],s=40,color='red')
ax.set_xlim(-10,10)
ax.set_ylim(-10,10)







