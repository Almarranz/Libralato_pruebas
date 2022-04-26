#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 12:52:18 2022

@author: amartinez
"""

# =============================================================================
# # THis is from tutorial https://hdbscan.readthedocs.io/en/latest/comparing_clustering_algorithms.html
# =============================================================================


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.cluster as cluster
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KDTree
from kneed import DataGenerator, KneeLocator
from sklearn.cluster import DBSCAN
import hdbscan
# %%
datos= '/Users/amartinez/Desktop/PhD/Libralato_data/practice/'
morralla = '/Users/amartinez/Desktop/morralla/'
data = np.load(datos+'clusterable_data.npy')
section = np.savetxt(morralla + 'cluster_test.txt',data, fmt = '%.7f')
# %%
fig ,ax = plt.subplots(1,1,figsize=(10,10))
ax.scatter(data[:,0],data[:,1], alpha= 0.3)

X=np.array([data[:,0],data[:,1]]).T
X_stad = StandardScaler().fit_transform(X)
# %%
samples_dist = 50
tree=KDTree(X_stad, leaf_size=2) 
dist, ind = tree.query(X_stad, k=samples_dist) #DistNnce to the 1,2,3...k neighbour
d_KNN=sorted(dist[:,-1])
# %

kneedle = KneeLocator(np.arange(0,len(data),1), d_KNN, curve='convex', interp_method = "polynomial",direction="increasing")
elbow = KneeLocator(np.arange(0,len(data),1), d_KNN, curve='concave', interp_method = "polynomial",direction="increasing")
rodilla=round(kneedle.elbow_y, 3)
try:
    codo = round(elbow.elbow_y, 3)
except :
    codo = 0
# %
# epsilon=np.mean(eps_for_mean)
# epsilon=codo
epsilon = round(min(d_KNN),3)
# sys.exit('salida')
# epsilon=rodilla

fig, ax = plt.subplots(1,1,figsize=(8,8))
ax.plot(np.arange(0,len(data),1),d_KNN)
# ax.legend(['knee=%s, min=%s, eps=%s, Dim.=%s'%(round(kneedle.elbow_y, 3),round(min(d_KNN),2),round(epsilon,2),len(X[0]))])
ax.set_xlabel('Point') 
ax.set_ylabel('%s-NN distance'%(samples_dist)) 
# print(round(kneedle.knee, 3))
# print(round(kneedle.elbow, 3))
# print(round(kneedle.knee_y, 3))
# print(round(kneedle.elbow_y, 3))
ax.axhline(rodilla,linestyle='dashed',color='k')
ax.axhline(codo,linestyle='dashed',color='k')
ax.axhline(epsilon,linestyle='dashed',color='red') 
ax.text(len(X)/2,epsilon, '%s'%(round(epsilon,3)),color='red')

ax.text(0,codo, '%s'%(codo))
ax.text(0,rodilla, '%s'%(rodilla))
ax.fill_between(np.arange(0,len(X)), epsilon, rodilla, alpha=0.5, color='grey')

# %

clustering = DBSCAN(eps=epsilon, min_samples=samples_dist).fit(X_stad)
l=clustering.labels_

l_set=len(set(l))
loop = 0
while len(set(l))<7:
    loop +=1
    clustering = DBSCAN(eps=epsilon, min_samples=samples_dist).fit(X_stad)
    
    l=clustering.labels_
    epsilon +=0.05 # if choose epsilon as min d_KNN you loop over epsilon and a "<" simbol goes in the while loop
    # samples_dist +=1 # if you choose epsilon as codo, you loop over the number of sambles and a ">" goes in the  while loop
    print('DBSCAN loop %s. Trying with eps=%s. cluster = %s '%(loop,round(epsilon,3),len(set(l))-1))
    if loop >1000:
        print('breaking out')
        break


n_clusters = len(set(l)) - (1 if -1 in l else 0)
n_noise=list(l).count(-1)
# %
u_labels = set(l)
colors=[plt.cm.rainbow(i) for i in np.linspace(0,1,len(set(l)))]# Returns a color for each cluster. Each color consists in four number, RGBA, red, green, blue and alpha. Full opacity black would be then 0,0,0,1
# %

# %
for k in range(len(colors)): #give noise color black with opacity 0.1
    if list(u_labels)[k] == -1:
        colors[k]=[0,0,0,0.1]
# %       
colores_index=[]

for c in u_labels:
    cl_color=np.where(l==c)
    colores_index.append(cl_color)
    
# %  
fig, ax = plt.subplots(1,1,figsize=(8,8))
ax.scatter(X[:,0],X[:,1], color=colors[-1],s=50,zorder=1,alpha=0.1)
for i in range(len(set(l))-1):
    ax.scatter(X[:,0][colores_index[i]],X[:,1][colores_index[i]], color=colors[i],s=50,zorder=3)


    ax.scatter(X[:,0][colores_index[i]],X[:,1][colores_index[i]], color=colors[i],s=50,zorder=3)
# %% 

# =============================================================================
# HDBSCAN
# =============================================================================
samples_dist=30

clustering = hdbscan.HDBSCAN(min_cluster_size=samples_dist, gen_min_span_tree=True,
                             allow_single_cluster=False).fit(X_stad)

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


fig, ax = plt.subplots(1,1,figsize=(8,8))
ax.scatter(X[:,0],X[:,1], color=colors[-1],s=50,zorder=1,alpha=0.1)
for i in range(len(set(l))-1):
    ax.scatter(X[:,0][colores_index[i]],X[:,1][colores_index[i]], color=colors[i],s=50,zorder=3)


    ax.scatter(X[:,0][colores_index[i]],X[:,1][colores_index[i]], color=colors[i],s=50,zorder=3)


fig, ax = plt.subplots(1,1,figsize=(10,10))
clustering.condensed_tree_.plot(select_clusters=True,selection_palette=colors)
# %%
import random
# Here we are going to create an artificial dense cluster in and mixing with the test data, then we will try to findings with hdbscan
cluster_false =np.empty((100,2))
for i in range(100):
    cluster_false[i][0],cluster_false[i][1]= random.uniform(0.04, 0.044),random.uniform(-0.10, -0.102)


# %%
print(cluster_false.shape)

data1 = np.r_[data,cluster_false]
# %%

fig ,ax = plt.subplots(1,1,figsize=(10,10))
ax.scatter(data1[:,0],data1[:,1], alpha= 0.3)

fig ,ax = plt.subplots(1,1,figsize=(10,10))
ax.scatter(data[:,0],data[:,1], alpha= 0.3)
np.savetxt(morralla + 'cluster_false.txt',data1)

# %%
X=np.array([data1[:,0],data1[:,1]]).T
X_stad = StandardScaler().fit_transform(X)
samples_dist=30

clustering = hdbscan.HDBSCAN(min_cluster_size=samples_dist, gen_min_span_tree=True,
                             allow_single_cluster=False).fit(X_stad)

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


fig, ax = plt.subplots(1,1,figsize=(8,8))
ax.scatter(X[:,0],X[:,1], color=colors[-1],s=50,zorder=1,alpha=0.1)
for i in range(len(set(l))-1):
    ax.scatter(X[:,0][colores_index[i]],X[:,1][colores_index[i]], color=colors[i],s=50,alpha =0.0011,zorder=3)


    


fig, ax = plt.subplots(1,1,figsize=(10,10))
clustering.condensed_tree_.plot(select_clusters=True,selection_palette=colors)


