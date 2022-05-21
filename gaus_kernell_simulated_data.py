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
fig, ax = plt.subplots(1,1,figsize = (10,10))
var = -6
var1 =-5
ax.hist(catal_all[:,var],bins='auto',density = 'True',histtype='step',color='#9370DB')
# ax.hist(catal[:,var],bins='auto',density = 'True',histtype='step')
# ax.hist(sub_catal[:,var],bins='auto',density = 'True',histtype='step')
# fig, ax = plt.subplots(1,1,figsize = (10,10))
# ax.scatter(sub_catal[:,var],sub_catal[:,var1])
# ax.hist2d(sub_catal[:,var],sub_catal[:,var1],density = 'True',bins = 100,alpha = 0.5)
# %%
# lest adjust the data to a gaussian kernel distributiom
# Ajuste del modelo KDE, folloing: https://www.cienciadedatos.net/documentos/pystats02-kernel-density-estimation-kde-python.html
# ==============================================================================
var = -6
# datos = catal[:,var]
# datos = catal_all[:,var]
datos = sub_catal[:,var]

modelo_kde = KernelDensity(kernel='gaussian', bandwidth=0.67444)

modelo_kde.fit(X=datos.reshape(-1,1))
# %
new_X = np.arange(-10,16,1)
log_density_pred = modelo_kde.score_samples(X=new_X.reshape(-1, 1))
#Se aplica el exponente para deshacer el logaritmo
density_pred = np.exp(log_density_pred)
print(density_pred)
# %
fig, ax = plt.subplots(1,1,figsize = (10,10))

ax.hist(datos,bins=len(new_X),histtype='step',color='k',linewidth=10,)
ax.vlines(np.mean(datos),0,160)
ax.plot(new_X,density_pred*300,color='red',linewidth=4)
# %%

# %
# # Validación cruzada para identificar kernel y bandwidth
# # ==============================================================================
from scipy import stats
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
param_grid = {'kernel': ['gaussian'],
              'bandwidth' : np.linspace(0.01, 3, 10)
              }

grid = GridSearchCV(
        estimator  = KernelDensity(),
        param_grid = param_grid,
        n_jobs     = -1,
        cv         = 10, 
        verbose    = 0
      )

# Se asigna el resultado a _ para que no se imprima por pantalla
_ = grid.fit(X = datos.reshape((-1,1)))

# %
print("----------------------------------------")
print("Mejores hiperparámetros encontrados (cv)")
print("----------------------------------------")
print(grid.best_params_, ":", grid.best_score_, grid.scoring)

modelo_kde_final = grid.best_estimator_



# %%
mub_sim_l = np.random.normal(loc=np.mean(datos), scale=1.338, size=len(datos))
fig, ax = plt.subplots(1,1,figsize = (10,10))
ax.hist(datos,bins=len(new_X),histtype='step',color='k',linewidth=10,)

ax.hist(mub_sim_l,bins=len(new_X),histtype='step',color='r',linewidth=10,)





