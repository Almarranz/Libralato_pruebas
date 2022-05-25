#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 25 12:44:21 2022

@author: amartinez
"""

# =============================================================================
# It runs dbscan on the subsections using the epsilin comute as the average of 
# the minimun KNN from the data and the minimun KNN from a simulated gaussian kernel
# of the same data.
# =============================================================================
import hdbscan
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
from scipy.stats import gaussian_kde
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
# 'ra dec x_c  y_c mua dmua mud dmud time n1 n2 ID mul mub dmul dmub '
name='WFC3IR'
catal_all = np.loadtxt(cata + '%s_pm_galactic.txt'%(name))
# %%

# name='ACSWFC'
trimmed_data='yes'
if trimmed_data=='yes':
    pre=''
elif trimmed_data=='no':
    pre='relaxed_'
    
else:
    sys.exit("Have to set trimmed_data to either 'yes' or 'no'")
    
# %%
section = 'A'#sections from A to D. Maybe make a script for each section...
subsec = '/Users/amartinez/Desktop/PhD/Libralato_data/pruebas/subsec_%s/'%(section)

col =np.arange(2,3,1)
row =np.arange(1,2,1)
area = 2.1
# %    
for c in range(len(col)):
    for r in range(len(row)):
    
        catal = np.loadtxt(subsec + 'subsec_%s_%s_%s_%smin.txt'%(section,col[c],row[r],area))
        
        valid=np.where((np.isnan(catal[:,3])==False) & (np.isnan(catal[:,4])==False ))
        catal=catal[valid]
        center=np.where(catal[:,3]-catal[:,4]>1.3)
        catal=catal[center]
        dmu_lim = 2
        vel_lim = np.where((catal[:,19]<=dmu_lim) & (catal[:,20]<=dmu_lim))
        catal=catal[vel_lim]
        
        datos = catal
        mul,mub = datos[:,-6],datos[:,-5]
        x,y = datos[:,7], datos[:,8]
        color = datos[:,3]-datos[:,4]
        
        mul_kernel, mub_kernel = gaussian_kde(mul), gaussian_kde(mub)
        x_kernel, y_kernel = gaussian_kde(x), gaussian_kde(y)
        mub_sim,  mul_sim = mub_kernel.resample(len(datos)), mul_kernel.resample(len(datos))
        x_sim, y_sim = x_kernel.resample(len(datos)), y_kernel.resample(len(datos))
        color_kernel = gaussian_kde(color)
        color_sim = color_kernel.resample(len(datos))
        
        fig, ax = plt.subplots(1,5, figsize=(40,10))
        ax[0].hist(mul, bins ='auto', histtype ='step',color = 'k',label ='real')
        ax[0].hist(mul_sim[0], bins ='auto', histtype = 'step',label ='sim',color = 'r')
        ax[1].hist(mub, bins ='auto', histtype ='step',color = 'k',label ='real')
        ax[1].hist(mub_sim[0], bins ='auto', histtype = 'step',label ='sim',color = 'r')
        
        ax[2].hist(x, bins ='auto', histtype ='step',color = 'k',label ='real')
        ax[2].hist(x_sim[0], bins ='auto', histtype = 'step',label ='sim',color = 'r')
        ax[3].hist(y, bins ='auto', histtype ='step',color = 'k',label ='real')
        ax[3].hist(y_sim[0], bins ='auto', histtype = 'step',label ='sim',color = 'r')
        
        ax[4].hist(color, bins ='auto', histtype ='step',color = 'k',label ='real')
        ax[4].hist(color_sim[0], bins ='auto', histtype = 'step',label ='sim',color = 'r')
        
        
        ax[0].set_xlabel('mul')
        ax[1].set_xlabel('mub')
        ax[2].set_xlabel('x')
        ax[3].set_xlabel('y')
        ax[4].set_xlabel('H$-$Ks')
        ax[0].legend(loc =2 )
        ax[1].legend(loc =2)
        ax[2].legend(loc =2 )
        ax[3].legend(loc =2)
        ax[4].legend(loc =2)
        
        
        # %
        
        X_sim=np.array([mul_sim[0],mub_sim[0],x_sim[0],y_sim[0],color_sim[0]]).T
        X_stad_sim = StandardScaler().fit_transform(X_sim)
        
        X=np.array([mul,mub,datos[:,7],datos[:,8],color]).T
        X_stad = StandardScaler().fit_transform(X)
        
        tree = KDTree(X_stad, leaf_size=2) 
        tree_sim =  KDTree(X_stad, leaf_size=2)
        
        samples=7# number of minimun objects that defined a cluster
        samples_dist = samples# t
        
        dist, ind = tree.query(X_stad, k=samples_dist) #DistNnce to the 1,2,3...k neighbour
        d_KNN=sorted(dist[:,-1])#distance to the Kth neighbour
        
        
        dist_sim, ind_sim = tree.query(X_stad_sim, k=samples_dist) #DistNnce to the 1,2,3...k neighbour
        d_KNN_sim=sorted(dist_sim[:,-1])#distance to the Kth neighbour
        
        
        fig, ax = plt.subplots(1,2,figsize=(20,10))
        ax[0].set_title('Sub_sec_%s_%s'%(col[c],row[r]))
        ax[0].plot(np.arange(0,len(datos),1),d_KNN,linewidth=1,color ='k')
        ax[0].plot(np.arange(0,len(datos),1),d_KNN_sim, color = 'r')
        
        # ax.legend(['knee=%s, min=%s, eps=%s, Dim.=%s'%(round(kneedle.elbow_y, 3),round(min(d_KNN),2),round(epsilon,2),len(X[0]))])
        ax[0].set_xlabel('Point') 
        ax[0].set_ylabel('%s-NN distance'%(samples)) 
        
        ax[1].hist(d_KNN,bins ='auto',histtype ='step',color = 'k')
        ax[1].hist(d_KNN_sim,bins ='auto',histtype ='step',color = 'r')
        ax[1].set_xlabel('%s-NN distance'%(samples)) 
        
        texto = '\n'.join(('min real d_KNN = %s'%(round(min(d_KNN),3)),'min sim d_KNN =%s'%(round(min(d_KNN_sim),3)),'average = %s'%((round(min(d_KNN),3)+round(min(d_KNN_sim),3))/2)))
        props = dict(boxstyle='round', facecolor='w', alpha=0.5)
        # place a text box in upper left in axes coords
        ax[1].text(0.65, 0.95, texto, transform=ax[1].transAxes, fontsize=14,
            verticalalignment='top', bbox=props)
        
        ax[1].set_ylabel('N') 
        # ax.set_xlim(0,0.5)
       


















































































