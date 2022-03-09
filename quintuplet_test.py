#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 13:47:39 2022

@author: amartinez
"""

# I going to tets my method with the quintuplet data from Ban

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
samples= 10
X=np.array([mu_alpha,mu_delta]).T

X_stad = StandardScaler().fit_transform(X)
print('These are the mean and std of X: %s %s'%(round(np.mean(X_stad),1),round(np.std(X_stad),1)))
# This is how I do it
tree=KDTree(X_stad, leaf_size=2) 

dist, ind = tree.query(X_stad, k=samples) #DistNnce to the 1,2,3...k neighbour
d_KNN=np.sort(dist[:,2])#distance to the indexed neighbour

# This how Ban do it
nn = NearestNeighbors(n_neighbors=samples, algorithm ='kd_tree')
nn.fit(X_stad)# our training is basically our dataset itself
dist_ban, ind_ban = nn.kneighbors(X_stad,samples)
d_KNN_ban = np.sort(dist_ban, axis=0)
d_KNN_ban = d_KNN_ban[:,1]



fig, ax = plt.subplots(1,1,figsize=(15,15))
ax.plot(np.arange(0,len(mu_delta),1),d_KNN)
ax.plot(np.arange(0,len(mu_delta),1),d_KNN_ban)
kneedle = KneeLocator(np.arange(0,len(mu_delta),1), d_KNN, curve='convex', interp_method = "polynomial")
kneedle_ban = KneeLocator(np.arange(0,len(mu_delta),1), d_KNN_ban, curve='convex', interp_method = "polynomial")

# print(round(kneedle.knee, 3))
# print(round(kneedle.elbow, 3))
# print(round(kneedle.knee_y, 3))
# print(round(kneedle.elbow_y, 3))
rodilla=round(kneedle.elbow_y, 3)
rodilla_ban=round(kneedle_ban.elbow_y,3)

ax.text(0,kneedle.elbow_y+0.1,'%s'%(round(kneedle.elbow_y, 3)),color='k')
ax.axhline(round(kneedle.elbow_y, 3),linestyle='dashed',color='k')

ax.text(0,kneedle_ban.elbow_y+0.1,'%s'%(round(kneedle_ban.elbow_y, 3)),color='red')
ax.axhline(round(kneedle_ban.elbow_y, 3),linestyle='dashed',color='red')





# %%
print(dist[:,1])
# =============================================================================
# nn = NearestNeighbors(n_neighbors=10, algorithm ='kd_tree')
# nn.fit(X)# our training is basically our dataset itself
# 
# 
# 
# dist, ind = nn.kneighbors(X,10)
# 
# 
# distances = np.sort(dist, axis=0)
# 
# 
# distances = distances[:,1]
# =============================================================================



























