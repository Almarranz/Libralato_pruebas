
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
from sklearn.cluster import DBSCAN, OPTICS, cluster_optics_dbscan
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import matplotlib.pylab as plt
from sklearn.neighbors import KDTree
from scipy import stats
from kneed import DataGenerator, KneeLocator
import pandas as pd


# In this bit I gonna a use the quintuplet data with HDBscan and try to finf Ban cluster without using epsilon

catal='/Users/amartinez/Desktop/PhD/Libralato_data/CATALOGS/'
a, b, c, d, e, f =  np.loadtxt(catal + 'pm-mas_quitaplet_Ban.txt', unpack =True)



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
samples = 2
# %%
# X=np.array([mu_alpha,mu_delta]).T
# %%This are the valuses for the tutorial
# np.random.seed(0)
# n_points_per_cluster = 3

# C1 = [-5, -2] + 0.8 * np.random.randn(n_points_per_cluster, 2)
# C2 = [4, -1] + 0.1 * np.random.randn(n_points_per_cluster, 2)
# C3 = [1, -2] + 0.2 * np.random.randn(n_points_per_cluster, 2)
# C4 = [-2, 3] + 0.3 * np.random.randn(n_points_per_cluster, 2)
# C5 = [3, -2] + 1.6 * np.random.randn(n_points_per_cluster, 2)
# C6 = [5, 6] + 2 * np.random.randn(n_points_per_cluster, 2)

C1 = [1, 1] 
C2 = [1.1, 1.3]
C3 = [1.4, 1.5] 
C4 = [2, 1] 
C5 = [1.6, 1.5] 
C6 = [20, 20] 

X = np.vstack((C1, C2, C3, C4, C5, C6))
# %%
# X=np.array((1,1),(2.2,1.2),(1.3,1.3),(1.4,1.6),(1.2,2),(20,20))
# clust = OPTICS(min_samples=50, xi=0.05, metric='euclidean').fit(X)
X_stad = StandardScaler().fit_transform(X)
# X_stad = X

# clusterer = hdbscan.HDBSCAN(min_cluster_size=samples, min_samples=min_cor,).fit(X_stad)
clusterer = OPTICS(min_samples=samples, xi=0.05, metric='euclidean').fit(X_stad)


l=clusterer.labels_

n_clusters = len(set(l)) - (1 if -1 in l else 0)
print('Number of cluster with min_samples=%s: %s'%(samples,n_clusters))
n_noise=list(l).count(-1)

u_labels = set(l)

colors=[plt.cm.rainbow(i) for i in np.linspace(0,1,len(set(l)))]# Returns a color for each cluster. Each color consists in four number, RGBA, red, green, blue and alpha. Full opacity black would be then 0,0,0,1
#colors=np.array(colors)
# for i in range(len(colors)):
#     colors[i][-1]=prob[i]






     
# %%
# This plots each cluster poit with its corresponding colour

for k in range(len(colors)):
    if list(u_labels)[k] == -1:
        colors[k]=[0,0,0,0.1]
     
colores_index=[]

for c in u_labels:
    cl_color=np.where(l==c)
    colores_index.append(cl_color)

fig, ax = plt.subplots(1,1,figsize=(8,8))

# for i in range(n_clusters):
for i in range(len(set(l))):
    # fig, ax = plt.subplots(1,1,figsize=(10,10))
    # ax.set_title('Cluster #%s'%(i+1))
    ax.scatter(X[:,0][colores_index[i]],X[:,1][colores_index[i]], color=colors[i],s=50)
ax.set_ylabel(r'$\mu_{\delta}$ [mas/yr]', fontsize=16)
ax.set_xlabel(r'$\mu_{\alpha}$ [mas/yr]', fontsize=16)    

# ax.set_xlim(-10, 10)
# ax.set_ylim(-10,10)    
# ax.invert_xaxis()
# %%
space = np.arange(len(X))
reachability = clusterer.reachability_[clusterer.ordering_]
labels = clusterer.labels_[clusterer.ordering_] # this gives you the ordering of the cluster by labels and reachabilty, that means that withn the labesl of each cluster are included the labels of noise that are close to each cluster
# print(reachability)
# fig, ax = plt.subplots(1,1,figsize=(8,8))
# ax.scatter(np.arange(len(X)),reachability)
# %%
# =============================================================================
# fig, ax = plt.subplots(1,1,figsize=(10,5))
# 
# for klass in range(0, n_clusters+1):
#     print(klass)
#     Xko =space[labels==klass]
#     Xk =space[np.where(labels==klass)]
#     Rk = reachability[np.where(labels == klass)]
#     ax.scatter(Xk, Rk, color=colors[klass], alpha=0.3)
# ax.scatter(space[np.where(labels==-1)], reachability[np.where(labels==-1)],color=colors[-1])
# ax.set_xlabel('Points(sorted by cluster)')
# ax.set_ylabel('Reachabilty(epsilon)')
# =============================================================================


# %%
# we are going to try and get the value for epsilon from the knee point in the reachabilty plot

distances = np.sort(reachability)
fif, ax =plt.subplots(1,1,figsize=(10,5))

# kneedle = KneeLocator(np.arange(0,len(distances)-1,1), distances[0:-1], interp_method = "polynomial",direction="increasing",curve='convex')
# rodilla=round(kneedle.knee_y, 3)

# ax.plot(list(range(len(reachability))),distances)
# # ax.axhline(round(kneedle.elbow_y, 3),linestyle='dashed',color='k')
# ax.legend(['knee=%s, min=%s'%(round(kneedle.elbow_y, 3),round(min(distances),2))])
ax.set_xlabel('Points(sorted by cluster)')
ax.set_ylabel('Reachabilty(epsilon)')
# print(round(kneedle.knee, 3))
# print(round(kneedle.elbow, 3))
# print(round(kneedle.knee_y, 3))
# print(round(kneedle.elbow_y, 3))
# for klass in range(0, n_clusters):
for klass in range(0, 1):
    print(klass)
    Xko =space[labels==klass]
    Xk =space[np.where(labels==klass)]
    Rk = reachability[np.where(labels == klass)]
    ax.scatter(Xko, Rk, color=colors[klass], alpha=0.3)
ax.scatter(space[np.where(labels==-1)], reachability[np.where(labels==-1)],color=colors[-1])

# %%

# fig,ax=plt.subplots(1,1,figsize=(30,10))
# ax.plot(np.arange(0,len(X),1),clusterer.reachability_[clusterer.ordering_])
# re_his=clusterer.reachability_[:-1]
# # ax.hist(clusterer.reachability_[1:],bins=700)
# # %%
# re_his=clusterer.reachability_[:-1]
# # %%

# fig,ax=plt.subplots(1,1,figsize=(30,10))
# ax.scatter(np.arange(0,len(X),1),clusterer.core_distances_[clusterer.ordering_])
# re_his=clusterer.reachability_[:-1]
# # ax.hist(clusterer.reachability_[1:],bins=700)






























