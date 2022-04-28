#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 09:21:39 2022

@author: amartinez
"""

# =============================================================================
# Imn this one we are going to use OPTICS with the test same test date used in hidden_cluster_hdbscan.py
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
from sklearn.cluster import DBSCAN, OPTICS, cluster_optics_dbscan
# %%
datos= '/Users/amartinez/Desktop/PhD/Libralato_data/practice/'
morralla = '/Users/amartinez/Desktop/morralla/'
test_data= np.load(datos+'clusterable_data.npy')
section = np.savetxt(morralla + 'cluster_test.txt',test_data, fmt = '%.7f')
# %%
fig ,ax = plt.subplots(1,1,figsize=(10,10))
ax.scatter(test_data[:,0],test_data[:,1], alpha= 0.3)
ax.grid()

data = test_data

X=np.array([data[:,0],data[:,1]]).T
X_stad = StandardScaler().fit_transform(X)

m_c_size = 90 # minimun cluster size
clusterer = OPTICS(min_samples=m_c_size, cluster_method = 'xi',xi=0.0029, metric='euclidean').fit(X_stad)

l=clusterer.labels_

n_clusters = len(set(l)) - (1 if -1 in l else 0)
print('Number of cluster with min_samples=%s: %s'%(m_c_size,n_clusters))
n_noise=list(l).count(-1)

u_labels = set(l)

colors=[plt.cm.rainbow(i) for i in np.linspace(0,1,len(set(l)))]# Returns a color for each cluster. Each color consists in four number, RGBA, red, green, blue and alpha. Full opacity black would be then 0,0,0,1


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

space = np.arange(len(X))
reachability = clusterer.reachability_[clusterer.ordering_]
labels = clusterer.labels_[clusterer.ordering_] # this gives you the ordering of the cluster by labels and reachabilty, that means that withn the labesl of each cluster are included the labels of noise that are close to each cluster


distances = np.sort(reachability)
# distances=reachability
fif, ax =plt.subplots(1,1,figsize=(10,5))



ax.plot(list(range(len(reachability))),reachability)

ax.set_xlabel('Points(sorted by cluster)')
ax.set_ylabel('Reachabilty(epsilon)')
# def slope(x1, y1, x2, y2):
#     m = round((y2-y1)/(x2-x1),5)
#     return m

def slope(y1, y2):
    m = round((y1)/(y2),1)
    return m

indx_f=10000
indx_i=1
plus = 5
for klass in range(0, n_clusters):
# for klass in range(0, 2):
    
    Xko =space[labels==klass]
    Xk =space[np.where(labels==klass)]
    Rk = reachability[np.where(labels == klass)]
    print(klass,Xk[1],Rk[1])
    ax.scatter(Xk, Rk, color=colors[klass], alpha=0.03)
# =============================================================================
#     try:
#         steep_f = slope(Rk[indx_f],Rk[indx_f+plus])
#         steep_i = slope(Rk[indx_i],Rk[indx_i+plus])
#         ax.annotate('%s'%(steep_i),(Xk[indx_i+plus],Rk[indx_i+plus]),color = colors[klass],weight='bold')
#         ax.annotate('%s'%(steep_f),(Xk[indx_f+plus],Rk[indx_f+plus]),color = colors[klass],weight='bold')
#     except:
#         steep_f = slope(Rk[-2],Rk[-1])
#         ax.annotate('%s'%(steep_f),(Xk[-1],Rk[-1]),color = colors[klass],weight='bold')
# 
# =============================================================================
    # try:
    #     steep_f = slope(Xk[indx_f],Rk[indx_f],Xk[indx_f+plus],Rk[indx_f+plus])
    #     steep_i = slope(Xk[indx_i],Rk[indx_i],Xk[indx_i+plus],Rk[indx_i+plus])
    #     ax.annotate('%s'%(steep_i),(Xk[indx_i+plus],Rk[indx_i+plus]),color = colors[klass],weight='bold')
    #     ax.annotate('%s'%(steep_f),(Xk[indx_f+plus],Rk[indx_f+plus]),color = colors[klass],weight='bold')
    # except:
    #     steep_f = slope(Xk[-2],Rk[-2],Xk[-1],Rk[-1])
    #     ax.annotate('%s'%(steep_f),(Xk[-1],Rk[-1]),color = colors[klass],weight='bold')

    
ax.scatter(space[np.where(labels==-1)], reachability[np.where(labels==-1)],color=colors[-1],alpha=0.3)






