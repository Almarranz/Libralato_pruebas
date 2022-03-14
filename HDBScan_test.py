#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 10:32:45 2022

@author: amartinez
"""

# This is a tutorial from: https://hdbscan.readthedocs.io/en/latest/advanced_hdbscan.html

import hdbscan
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as np

from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import matplotlib.pylab as plt
from sklearn.neighbors import KDTree
from scipy import stats
from kneed import DataGenerator, KneeLocator
import pandas as pd


pruebas='/Users/amartinez/Desktop/PhD/Libralato_data/pruebas/'
data= np.load(pruebas + 'clusterable_data.npy')
# %%
# fig, ax = plt.subplots(1,1, figsize=(8,8))
# ax.scatter(*data.T,alpha=0.25,linewidth=0)
# %%
samples=15
clusterer = hdbscan.HDBSCAN(min_cluster_size=samples).fit(data)

# %%
l=clusterer.labels_

n_clusters = len(set(l)) - (1 if -1 in l else 0)
print('Number of cluster with min_samples=%s: %s'%(samples,n_clusters))
n_noise=list(l).count(-1)

u_labels = set(l)
prob=clusterer.probabilities_
colors=[plt.cm.rainbow(i) for i in np.linspace(0,1,len(set(l)))]# Returns a color for each cluster. Each color consists in four number, RGBA, red, green, blue and alpha. Full opacity black would be then 0,0,0,1
#colors=np.array(colors)
# for i in range(len(colors)):
#     colors[i][-1]=prob[i]



# %%

# This plot each cluster point with it corresponding color and with it corresponding alpha acording to its probability
for k in range(len(colors)):
    if list(u_labels)[k] == -1:
        colors[k]=[0,0,0,0.1]   
prob=clusterer.probabilities_

ind_c=[]
for i in range(len(data)):
    ind_c.append(colors[l[i]])
ind_c=np.array(ind_c)
for i in range(len(data)):
    ind_c[i][-1]=prob[i]
fig, ax = plt.subplots(1,1,figsize=(8,8))
ax.scatter(data[:,0],data[:,1],color=ind_c,)#gor plotting the noise pointins give some value to alpha

# %%
# This plots each cluster poit with its corresponding colour

# =============================================================================
# for k in range(len(colors)):
#     if list(u_labels)[k] == -1:
#         colors[k]=[0,0,0,0.1]
#      
# colores_index=[]
# 
# for c in u_labels:
#     cl_color=np.where(l==c)
#     colores_index.append(cl_color)
# 
# fig, ax = plt.subplots(1,1,figsize=(8,8))
# 
# # for i in range(n_clusters):
# for i in range(len(set(l))):
#     # fig, ax = plt.subplots(1,1,figsize=(10,10))
#     # ax.set_title('Cluster #%s'%(i+1))
#     ax.scatter(data[:,0][colores_index[i]],data[:,1][colores_index[i]], color=colors[i],s=50)
# =============================================================================
# %%
# In this bit I gonna a use the quintuplet data with HDBscan and try to finf Ban cluster without using epsilon
a, b, c, d, e, f =  np.loadtxt('/Users/banafsh/Python/Machine_Learning_clustering_pm_project/pm-mas.txt', unpack =True)

#removing the scale on X for plotting it
ind = np.where( (c > -10) & (c < 10) & (d < 10) & (d > -10))

c = c[ind]
d = d[ind]
a = a[ind]
b = b[ind]


import math

mu_delta = math.cos(math.radians(148.6)) * (c) + math.sin(math.radians(148.6)) * (d)
mu_alpha = -1 * math.sin(math.radians(148.6)) * (c) + math.cos(math.radians(148.6)) * (d)


mu = np.vstack((mu_alpha,mu_delta)).T
# %%
samples = 10
X=np.array([mu_alpha,mu_delta]).T

X_stad = StandardScaler().fit_transform(X)

clustering = hdbscan.HDBSCAN(min_cluster_size=samples, min_samples=samples).fit(X_stad)

































