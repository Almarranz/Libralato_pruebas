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
    
row_col =[2,2]
# "'RA_gns','DE_gns','Jmag','Hmag','Ksmag','ra','dec','x_c','y_c','mua','dmua','mud','dmud','time','n1','n2','ID','mul','mub','dmul','dmub','m139','Separation'",
catal_all = np.loadtxt(results + '%smatch_GNS_and_%s_refined_galactic.txt'%(pre,name))
catal=np.loadtxt(results + 'sec_%s_%smatch_GNS_and_%s_refined_galactic.txt'%(section,pre,name))
sub_catal = np.loadtxt(subsec + 'subsec_A_%s_%s.txt'%(row_col[0],row_col[1]))    
# %%    

var = -6
mul,mub = sub_catal[:,-6],sub_catal[:,-5]
mul_kernel = gaussian_kde(mul)
mub_kernel = gaussian_kde(mub)
mub_sim = mub_kernel.resample(len(sub_catal))
mul_sim = mul_kernel.resample(len(sub_catal))

fig, ax = plt.subplots(1,2, figsize=(20,10))
ax[0].hist(mul, bins ='auto', histtype ='step',color = 'k',label ='real')
ax[0].hist(mul_sim[0], bins ='auto', histtype = 'step',label ='sim')
ax[1].hist(mub, bins ='auto', histtype ='step',color = 'k',label ='real')
ax[1].hist(mub_sim[0], bins ='auto', histtype = 'step',label ='sim')
ax[0].set_xlabel('mul')
ax[1].set_xlabel('mub')
ax[0].legend(loc =2 )
ax[1].legend(loc =2)








