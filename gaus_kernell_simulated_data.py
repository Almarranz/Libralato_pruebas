#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 21 09:14:04 2022

@author: amartinez
"""

# =============================================================================
# In here we are going to generates simulates data for each of the subsecciotion
# generated by dividing_sections.py, using Gaussina kernel generated data,
# whit the same distribution of the astrometric parametres.
# =============================================================================
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors, KernelDensity
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
from scipy.stats import norm, gaussian_kde
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
subsec = '/Users/amartinez/Desktop/PhD/Libralato_data/pruebas/subsec_%s/'%(section)
name='WFC3IR'
# name='ACSWFC'
trimmed_data='yes'
if trimmed_data=='yes':
    pre=''
elif trimmed_data=='no':
    pre='relaxed_'
    
else:
    sys.exit("Have to set trimmed_data to either 'yes' or 'no'")
    
col =np.arange(0,8,1)
row =np.arange(0,6,1)
# col =[2]
# row =[2]
# "'RA_gns','DE_gns','Jmag','Hmag','Ksmag','ra','dec','x_c','y_c','mua','dmua','mud','dmud','time','n1','n2','ID','mul','mub','dmul','dmub','m139','Separation'",
catal_all = np.loadtxt(results + '%smatch_GNS_and_%s_refined_galactic.txt'%(pre,name))
catal=np.loadtxt(results + 'sec_%s_%smatch_GNS_and_%s_refined_galactic.txt'%(section,pre,name))
# %%    
for c in range(len(col)):
    for r in range(len(row)):
        try:
            sub_catal = np.loadtxt(subsec + 'subsec_A_%s_%s.txt'%(col[c],row[r]))    
            
            datos = sub_catal
            var = -6
            mul,mub = datos[:,-6],datos[:,-5]
            mul_kernel = gaussian_kde(mul)
            mub_kernel = gaussian_kde(mub)
            mub_sim = mub_kernel.resample(len(datos))
            mul_sim = mul_kernel.resample(len(datos))
            
            
            fig, ax = plt.subplots(1,2, figsize=(20,10))
            ax[0].hist(mul, bins ='auto', histtype ='step',color = 'k',label ='real')
            ax[0].hist(mul_sim[0], bins ='auto', histtype = 'step',label ='sim')
            ax[1].hist(mub, bins ='auto', histtype ='step',color = 'k',label ='real')
            ax[1].hist(mub_sim[0], bins ='auto', histtype = 'step',label ='sim')
            ax[0].set_xlabel('mul')
            ax[1].set_xlabel('mub')
            ax[0].legend(loc =2 )
            ax[1].legend(loc =2)
            
            # %
            
            X_sim=np.array([mul_sim[0],mub_sim[0],datos[:,7],datos[:,8]]).T
            X_stad_sim = StandardScaler().fit_transform(X_sim)
            
            X=np.array([mul,mub,datos[:,7],datos[:,8]]).T
            X_stad = StandardScaler().fit_transform(X)
            
            tree = KDTree(X_stad, leaf_size=2) 
            tree_sim =  KDTree(X_stad, leaf_size=2)
            
            samples=5# number of minimun objects that defined a cluster
            samples_dist = samples# t
            
            dist, ind = tree.query(X_stad, k=samples_dist) #DistNnce to the 1,2,3...k neighbour
            d_KNN=sorted(dist[:,-1])#distance to the Kth neighbour
            
            dist_sim, ind_sim = tree.query(X_stad_sim, k=samples_dist) #DistNnce to the 1,2,3...k neighbour
            d_KNN_sim=sorted(dist_sim[:,-1])#distance to the Kth neighbour
            
            
            fig, ax = plt.subplots(1,2,figsize=(20,10))
            ax[0].set_title('Sub_sec_%s_%s'%(col[c],row[r]))
            ax[0].plot(np.arange(0,len(datos),1),d_KNN,linewidth=1)
            ax[0].plot(np.arange(0,len(datos),1),d_KNN_sim, color = 'r')
            
            # ax.legend(['knee=%s, min=%s, eps=%s, Dim.=%s'%(round(kneedle.elbow_y, 3),round(min(d_KNN),2),round(epsilon,2),len(X[0]))])
            ax[0].set_xlabel('Point') 
            ax[0].set_ylabel('%s-NN distance'%(samples)) 
            
            ax[1].hist(d_KNN,bins ='auto',histtype ='step',color = 'k')
            ax[1].hist(d_KNN_sim,bins ='auto',histtype ='step',color = 'r')
            ax[1].set_xlabel('%s-NN distance'%(samples)) 
            ax[1].set_ylabel('N') 
            # ax.set_xlim(0,0.5)
        except:
            print('there is no sectioin %s_%s'%(col[c],row[r]))


























