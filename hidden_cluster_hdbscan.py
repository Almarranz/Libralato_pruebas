#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 15:38:37 2022

@author: amartinez
"""

# =============================================================================
# In this script we are going to hide a super dense cluster in the test data, then we will figure out the way to highlight it
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
test_data= np.load(datos+'clusterable_data.npy')
section = np.savetxt(morralla + 'cluster_test.txt',test_data, fmt = '%.7f')
# %%
fig ,ax = plt.subplots(1,1,figsize=(10,10))
ax.scatter(test_data[:,0],test_data[:,1], alpha= 0.3)
ax.grid()
import random
# Here we are going to create an artificial dense cluster(denser than the rest of the distribution) in and mixing with the test data, then we will try to findings with hdbscan
size_false = 40
cluster_false =np.empty((size_false,2))
cluster_false1 =np.empty((size_false,2))

# density = 'low'
density = 'high'
x,y = 0.15,-0.05
if density == 'high':
    dx = x + x/3
    dy = y + y/3
elif density == 'low':
    dx = x + x/1
    dy = y + y/1
for i in range(size_false):
    cluster_false[i][0],cluster_false[i][1]= random.uniform(x, dx),random.uniform(y, dy)
    # cluster_false1[i][0],cluster_false1[i][1]= random.uniform(0.21, 0.22),random.uniform(0.10, 0.18)
    
    
data1= np.r_[test_data,cluster_false]
fig ,ax = plt.subplots(1,1,figsize=(10,10))
ax.scatter(data1[:,0],data1[:,1], alpha= 0.3)
ax.grid()
# %

clu = 'test+cluster'
# clu = 'test'# clu = test only test data. clu = 'test+cluster', test plus hidden cluster(s)
if clu == 'test':
    data = test_data
elif clu == 'test+cluster':
    data = data1

X=np.array([data[:,0],data[:,1]]).T
X_stad = StandardScaler().fit_transform(X)

m_c_size = 5 # mini size of a cluster(in_cluster_size)
m_core = 20 # number of point within a distance for a point to be core (min_samples)
clustering = hdbscan.HDBSCAN(min_cluster_size=m_c_size, min_samples=m_core, gen_min_span_tree=True,
                             allow_single_cluster=False,cluster_selection_epsilon=0.035,
                             cluster_selection_method = 'eom').fit(X_stad)

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


fig, ax = plt.subplots(1,1,figsize=(10,10))
ax.set_title('Cluster found = %s'%(n_clusters))
ax.scatter(X[:,0],X[:,1], color=colors[-1],s=50,zorder=1,alpha=0.1)
for i in range(len(set(l))-1):
    ax.scatter(X[:,0][colores_index[i]],X[:,1][colores_index[i]], color=colors[i],s=50,zorder=3,alpha=0.5)
ax.grid()

fig, ax = plt.subplots(1,1,figsize=(8,8))
ax.set_title('%s density = %s'%(clu, density))
clustering.condensed_tree_.plot(select_clusters=True,selection_palette=colors)
ax.grid()



# %%
tree=KDTree(cluster_false, leaf_size=2) 
dist, ind = tree.query(cluster_false, k=10) #DistNnce to the 1,2,3...k neighbour
d_KNN=sorted(dist[:,-1])
# %

kneedle = KneeLocator(np.arange(0,len(cluster_false),1), d_KNN, curve='convex', interp_method = "polynomial",direction="increasing")
elbow = KneeLocator(np.arange(0,len(cluster_false),1), d_KNN, curve='concave', interp_method = "polynomial",direction="increasing")
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
ax.set_title('False cluster density = %s'%(density))
ax.plot(np.arange(0,len(cluster_false),1),d_KNN)
ax.axhline(rodilla,linestyle='dashed',color='k')
ax.axhline(codo,linestyle='dashed',color='k')
ax.text(0,codo, '%s'%(codo))
ax.text(0,rodilla, '%s'%(rodilla))
# %%
lim_xy =[0.2,0.25,-0.05,0.2]
selection = np.where((test_data[:,0]>lim_xy[0]) & (test_data[:,0]<lim_xy[1]) &
                     (test_data[:,1]>lim_xy[2]) & (test_data[:,1]<lim_xy[3]))
test_data = test_data[selection]
tree=KDTree(test_data, leaf_size=2) 
dist, ind = tree.query(test_data, k=10) #DistNnce to the 1,2,3...k neighbour
d_KNN=sorted(dist[:,-1])
# %

kneedle = KneeLocator(np.arange(0,len(test_data),1), d_KNN, curve='convex', interp_method = "polynomial",direction="increasing")
elbow = KneeLocator(np.arange(0,len(test_data),1), d_KNN, curve='concave', interp_method = "polynomial",direction="increasing")
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
ax.set_title('tes_data')
ax.plot(np.arange(0,len(test_data),1),d_KNN)
ax.axhline(rodilla,linestyle='dashed',color='k')
ax.axhline(codo,linestyle='dashed',color='k')
ax.text(0,codo, '%s'%(codo))
ax.text(0,rodilla, '%s'%(rodilla))


# %%







