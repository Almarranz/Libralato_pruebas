#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 09:23:55 2022

@author: amartinez
"""

from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt

# %%
# Following the tutorial at https://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html#sphx-glr-auto-examples-cluster-plot-dbscan-py
X = np.array([[1, 2], [2, 2], [2, 3],[8, 7], [80, 7],[8, 8],[10,15],[2,2.5] ])
clustering = DBSCAN(eps=3, min_samples=2).fit(X)

docu=DBSCAN.__doc__

l=clustering.labels_
# %%
fig, ax = plt.subplots(1,1,figsize=(10,10))
ax.scatter(X[:,0],X[:,1])

#%%

# %%
print(clustering.core_sample_indices_)

# %% Dont need these, that are used in the tuturial
# =============================================================================
# core_samples_mask = np.zeros_like(clustering.labels_, dtype=bool)
# 
# core_samples_mask[clustering.core_sample_indices_]=True
# 
# n_clusters = len(set(l)) - (1 if -1 in l else 0)
# 
# n_noise=list(l).count(-1)
# =============================================================================
# %%
u_labels = set(l)
colors=[plt.cm.rainbow(i) for i in np.linspace(0,1,len(set(l)))]# Returns a color for each cluster. Each color consists in four number, RGBA, red, green, blue and alpha. Full opacity blacn would be then 0,0,0,1
# %%

# %%
for k in range(len(colors)):
    if list(u_labels)[k] == -1:
        colors[k]=[0,0,0,1]
        
colores_index=[]

for c in u_labels:
    cl_color=np.where(l==c)
    colores_index.append(cl_color)
# %%
print(colores_index)

fig, ax = plt.subplots(1,1,figsize=(10,10))
for i in range(len(set(l))):
    ax.scatter(X[:,0][colores_index[i]],X[:,1][colores_index[i]], color=colors[i],s=200,edgecolors='k')
    














