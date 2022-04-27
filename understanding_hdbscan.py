#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 17:35:08 2022

@author: amartinez
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.datasets as data
import matplotlib.pyplot as plt
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
sns.set_context('poster')
sns.set_style('white')
sns.set_color_codes()
plot_kwds = {'alpha' : 0.5, 's' : 80, 'linewidths':0}

moons, _ = data.make_moons(n_samples=50, noise=0.05)
blobs, _ = data.make_blobs(n_samples=50, centers=[(-0.75,2.25), (1.0, 2.0)], cluster_std=0.25)
test_data = np.vstack([moons, blobs])
plt.scatter(test_data.T[0], test_data.T[1], color='b', **plot_kwds)

import hdbscan

clusterer = hdbscan.HDBSCAN(min_cluster_size=5, gen_min_span_tree=True,approx_min_span_tree=True, leaf_size=40,metric='euclidean')
clusterer.fit(test_data)

# hdbscan(algorithm='best', alpha=1.0, approx_min_span_tree=True,gen_min_span_tree=True, leaf_size=40,metric='euclidean', min_cluster_size=5, min_samples=None, p=None)

clusterer.minimum_spanning_tree_.plot(edge_cmap='viridis',
                                      edge_alpha=0.6,
                                      node_size=80,
                                      edge_linewidth=2)


# %%
cata='/Users/amartinez/Desktop/PhD/Libralato_data/CATALOGS/'
pruebas='/Users/amartinez/Desktop/PhD/Libralato_data/pruebas/'
results='/Users/amartinez/Desktop/PhD/Libralato_data/results/'
name='WFC3IR'
group = 1
# "'RA_gns','DE_gns','Jmag','Hmag','Ksmag','ra','dec','x_c','y_c','mua','dmua','mud','dmud','time','n1','n2','ID','mul','mub','dmul','dmub','m139','Separation'"
data=np.loadtxt(pruebas + 'group_radio%s_%s_%s.txt'%(76,group,name))

# %%
X=np.array([data[:,-6],data[:,-5],data[:,7],data[:,8]]).T
from sklearn.preprocessing import QuantileTransformer, StandardScaler, Normalizer, RobustScaler, PowerTransformer
method = StandardScaler()

X_stad= method.fit_transform(X)
samples_dist=10
clustering = hdbscan.HDBSCAN(min_cluster_size=samples_dist, gen_min_span_tree=True,allow_single_cluster=False).fit(X_stad)

l=clustering.labels_

n_clusters = len(set(l)) - (1 if -1 in l else 0)
n_noise=list(l).count(-1)
# %
u_labels = set(l)
colors=[plt.cm.rainbow(i) for i in np.linspace(0,1,len(set(l)))]# Returns a color for each cluster. Each color consists in four number, RGBA, red, green, blue and alpha. Full opacity black would be then 0,0,0,1


# %
for k in range(len(colors)): #give noise color black with opacity 0.1
    if list(u_labels)[k] == -1:
        colors[k]=[0,0,0,0.1]
# % 
colores_index=[]

for c in u_labels:
    cl_color=np.where(l==c)
    colores_index.append(cl_color)

fig, ax = plt.subplots(1,3,figsize=(30,10))

ax[2].invert_yaxis()


for i in range(len(set(l))-1):
# t_gal['l'] = t_gal['l'].wrap_at('180d')
    ax[0].scatter(X[:,0][colores_index[-1]],X[:,1][colores_index[-1]], color=colors[-1],s=50,zorder=1)
    ax[0].scatter(X[:,0],X[:,1], color=colors[-1],s=50,zorder=1)
    # ax[1].quiver(t_gal['l'][colores_index[-1]].value,t_gal['b'][colores_index[-1]].value, X[:,0][colores_index[-1]]-pms[2], X[:,1][colores_index[-1]]-pms[3], alpha=0.5, color=colors[-1])
    
    ax[0].scatter(X[:,0][colores_index[i]],X[:,1][colores_index[i]], color=colors[i],s=50,zorder=3)
    ax[0].set_xlim(-10,10)
    ax[0].set_ylim(-10,10)
    ax[0].set_xlabel(r'$\mathrm{\mu_{l} (mas\ yr^{-1})}$') 
    ax[0].set_ylabel(r'$\mathrm{\mu_{b} (mas\ yr^{-1})}$') 
    ax[0].invert_xaxis()

fig, ax = plt.subplots(1,1,figsize=(10,10))
clustering.condensed_tree_.plot(select_clusters=True,selection_palette=colors)

# %%
fig, ax = plt.subplots(1,3,figsize=(30,10))

ax[2].invert_yaxis()
i=0
ax[0].scatter(X[:,0][colores_index[-1]],X[:,1][colores_index[-1]], color=colors[-1],s=50,zorder=1)
ax[0].scatter(X[:,0],X[:,1], color=colors[-1],s=50,zorder=1)
# ax[1].quiver(t_gal['l'][colores_index[-1]].value,t_gal['b'][colores_index[-1]].value, X[:,0][colores_index[-1]]-pms[2], X[:,1][colores_index[-1]]-pms[3], alpha=0.5, color=colors[-1])

ax[0].scatter(X[:,0][colores_index[i]],X[:,1][colores_index[i]], color=colors[i],s=50,zorder=3)
ax[0].set_xlim(-10,10)
ax[0].set_ylim(-10,10)
ax[0].set_xlabel(r'$\mathrm{\mu_{l} (mas\ yr^{-1})}$') 
ax[0].set_ylabel(r'$\mathrm{\mu_{b} (mas\ yr^{-1})}$') 
ax[0].invert_xaxis()

fig, ax = plt.subplots(1,1,figsize=(10,10))
clustering.condensed_tree_.plot(select_clusters=True,selection_palette=colors)



