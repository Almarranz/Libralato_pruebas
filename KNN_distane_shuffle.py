#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 30 22:48:05 2022

@author: amartinez
"""

# =============================================================================
# Shuffle the velocities with position adn computing the minimun KNN distace
# in order to find the limit to the rigth epsilon when using 
# dbscan
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
import random
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

section = 'A'
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
# "'RA_gns','DE_gns','Jmag','Hmag','Ksmag','ra','dec','x_c','y_c','mua','dmua','mud','dmud','time','n1','n2','ID','mul','mub','dmul','dmub','m139','Separation'",
# catal=np.loadtxt(pruebas + 'sec_%s_%smatch_GNS_and_%s_refined_galactic.txt'%(section,pre,name))
catal=np.genfromtxt(results + 'sec_%s_%smatch_GNS_and_%s_refined_galactic.txt'%(section,pre,name))



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
# This is if we want to use only stars streaming eastward
# east = np.where(catal[:,-6]>-5.72)
# catal_east=catal
# catal_east=catal[east]


# %
Aks_gns = pd.read_fwf(gns_ext + 'central.txt', sep =' ',header = None)

# %
AKs_np = Aks_gns.to_numpy()#TODO
center = np.where(AKs_np[:,6]-AKs_np[:,8] > 1.3)#TODO
AKs_center =AKs_np[center]#TODO
# %
gns_coord = SkyCoord(ra=AKs_center[:,0]*u.degree, dec=AKs_center[:,2]*u.degree)
# %
# %
AKs_list =  np.arange(1.6,2.11,0.01)

# % This is for shuffling the magnitudes
samples =7
H = catal[:,3]
Ks = catal[:,4]
magns = np.array([H,Ks]).T
np.random.shuffle(magns)
H_shu, Ks_shu = magns[:,0],magns[:,1]
# %
# Here we are goint to try a different aproach, by simple shuffeling the exiting velocities among the exiting positions
veloc = np.array([catal[:,-6],catal[:,-5]]).T
np.random.shuffle(veloc)
mul_sim, mub_sim = veloc[:,0], veloc[:,1]
# # %
# catal_sim = np.c_[catal[:,7],catal[:,8],mul_sim,mub_sim,
#                   catal[:,3],catal[:,4]]
catal_sim = np.c_[catal[:,7],catal[:,8],mul_sim,mub_sim,
                  H_shu,Ks_shu]

X=np.array([cata[:,7],catal[:,8],cata[:,-6],catal[:-5],catal[:,3]-catal[:,4]]).T
X_stad = StandardScaler().fit_transform(X)
tree = KDTree(X_stad, leaf_size=2) 
dist, ind = tree.query(X_stad, k=samples) #DistNnce to the 1,2,3...k neighbour
d_KNN=sorted(dist[:,-1])#distance to the Kth neighbour

X_sim=catal_sim
X_stad_sim = StandardScaler().fit_transform(X_sim)
tree_sim =  KDTree(X_stad_sim, leaf_size=2)

dist_sim, ind_sim = tree_sim.query(X_stad_sim, k=samples) #DistNnce to the 1,2,3...k neighbour
d_KNN_sim=sorted(dist_sim[:,-1])#distance to the Kth neighbou


fig, ax = plt.subplots(1,1,figsize=(20,10))

ax.hist(d_KNN,bins ='auto',histtype ='step',color = 'k',label = 'real',linewidth=5)
ax.hist(d_KNN_sim,bins ='auto',histtype ='step',color = 'r', label = 'simulated',linewidth=5,alpha =0.5)
ax.set_xlabel('%s-NN distance'%(samples))
props = dict(boxstyle='round', facecolor='k', alpha=0.3)
# place a text box in upper left in axes coords
# ax.text(0.55, 0.58, 'The presence of $\it{clustered}$ data \nreduces the KNN distance\nin the real sample', transform=ax.transAxes, fontsize=27,
#         verticalalignment='top', bbox=props)
ax.axvline(min(d_KNN),ls = 'dashed', color ='k', linewidth = 5)
ax.axvline(min(d_KNN_sim),ls = 'dashed', color ='r', linewidth = 5,alpha =0.5)
ax.set_ylabel('N')
ax.set_xlim(0,0.05)
ax.legend()




