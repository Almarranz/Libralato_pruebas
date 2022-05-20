#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 20 11:28:52 2022

@author: amartinez
"""

# =============================================================================
# Here we are going to divide the section in smalles LxL areas, using an especific
# epsilon for that area.
# =============================================================================
# %%imports
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
# %%plotting parametres
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

section = 'A'#selecting the whole thing
subsec = '/Users/amartinez/Desktop/PhD/Libralato_data/pruebas/subsec_%s/'%(section)

# "'RA_gns','DE_gns','Jmag','Hmag','Ksmag','ra','dec','x_c','y_c','mua','dmua','mud','dmud','time','n1','n2','ID','mul','mub','dmul','dmub','m139','Separation'",
if section == 'All':
    catal=np.loadtxt(results + '%smatch_GNS_and_%s_refined_galactic.txt'%(pre,name))
else:
    catal=np.loadtxt(results + 'sec_%s_%smatch_GNS_and_%s_refined_galactic.txt'%(section,pre,name))
# %%
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
dmu_lim = 1
vel_lim = np.where((catal[:,19]<=dmu_lim) & (catal[:,20]<=dmu_lim))
catal=catal[vel_lim]

# 'ra dec x_c  y_c mua dmua mud dmud time n1 n2 ID mul mub dmul dmub '
# catal_all = np.loadtxt(cata + '%s_pm_galactic.txt'%(name))
# %%
clus_test = np.loadtxt(pruebas + 'dbs_cluster1_of_group89.txt')
m1 = -0.85
m = 1.1
step = 1000

for f_remove in glob.glob(pruebas + 'subsec_%s/subsec*'%(section)):
    os.remove(f_remove)
colores =['b','r','g','orange','fuchsia']
missing =0
fig, ax = plt.subplots(1,1, figsize=(10,10))
ax.scatter(catal[:,7],catal[:,8])
fila =-1
for i in np.arange(26300,17300,-step):
    fila = int((26300 - i)/1000)
    yg_1 =  i + m*catal[:,7]
    yg_2 =  i - step +m*catal[:,7]
    ax.plot(catal[:,7],yg_1, color ='g')
    # ax.scatter(catal[:,7][good],catal[:,8][good],color =np.random.choice(colores))
    for j in np.arange(33000,26000,-step):
        columna =int((33000 - j)/step)
       
        yr_1 = j + m1*catal[:,7]
        yr_2 = j - step +m1*catal[:,7]
        ax.plot(catal[:,7],yr_1, color ='r')
        good = np.where((catal[:,8]<yg_1)&(catal[:,8]>yg_2)
                        & (catal[:,8]<yr_1)&(catal[:,8]>yr_2))
        ax.scatter(catal[:,7][good],catal[:,8][good],color =np.random.choice(colores))
        if len(good[0]>0):
            print(fila,columna)
            np.savetxt(pruebas + 'subsec_%s/subsec_%s_%s_%s.txt'%(section, section, fila, columna)
                       ,catal[good],fmt='%.7f %.7f %.4f %.4f %.4f %.7f %.7f %.4f %.4f %.5f %.5f %.5f %.5f %.0f %.0f %.0f %.0f %.5f %.5f %.5f %.5f %.5f %.3f',
                       header ="'RA_gns','DE_gns','Jmag','Hmag','Ksmag','ra','dec','x_c','y_c','mua','dmua','mud','dmud','time','n1','n2','ID','mul','mub','dmul','dmub','m139','Separation'")
            missing += len(good[0])
print(missing, len(catal))
    # ax.scatter(catal[:,7][good],catal[:,8][good],color =np.random.choice(colores))
    # for j in range(26300 - j*step,26300):
        
ax.scatter(clus_test[:,2],clus_test[:,3])
ax.set_ylim(min(catal[:,8]-500),max(catal[:,8]+500))
# %%
subs = np.loadtxt(subsec +'subsec_A_3_2.txt')
fig, ax = plt.subplots(1,1, figsize=(10,10))
ax.scatter(catal[:,7],catal[:,8])
ax.scatter(subs[:,7],subs[:,8])








