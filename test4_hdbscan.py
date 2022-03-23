#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 14:02:42 2022

@author: amartinez
"""

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


C1 = [1, 1] 
C2 = [1.1, 1.3]
C3 = [1.4, 1.5] 
C4 = [2, 2] 
C5 = [2.1, 2.5] 
C6 = [2.4, 2.3] 
C7 = [1, 2.4]
C8 = [2.4,2]
X = np.vstack((C1, C2, C3, C4, C5, C6,C7,C8))
X_stad = StandardScaler().fit_transform(X)
samples=3
min_cor=2

clusterer = hdbscan.HDBSCAN(min_cluster_size=samples, min_samples=min_cor,gen_min_span_tree=True).fit(X_stad)
# clusterer = hdbscan.HDBSCAN(min_cluster_size=samples,cluster_selection_epsilon=0.1,cluster_selection_method = 'leaf')
clusterer.fit(X_stad)

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





# This plot each cluster point with it corresponding color and with it corresponding alpha acording to its probability
for k in range(len(colors)):
    if list(u_labels)[k] == -1:
        colors[k]=[0,0,0,0.1]   
prob=clusterer.probabilities_

ind_c=[]
for i in range(len(X)):
    ind_c.append(colors[l[i]])
ind_c=np.array(ind_c)
for i in range(len(X)):
    ind_c[i][-1]=prob[i]
fig, ax = plt.subplots(1,1,figsize=(7,7))
ax.set_title('HDBScan')
ax.scatter(X[:,0],X[:,1],color=ind_c,alpha=0.5,s =20)#gor plotting the noise pointins give some value to alpha

     
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
#     ax.scatter(X[:,0][colores_index[i]],X[:,1][colores_index[i]], color=colors[i],s=50)
# =============================================================================
# %%


fig, ax = plt.subplots(1,1,figsize=(7,7))
clusterer.condensed_tree_.plot(select_clusters=True, selection_palette=colors)

# %%
fig, ax = plt.subplots(1,1,figsize=(10,10))
clusterer.condensed_tree_.plot()



# %%
fig, ax = plt.subplots(1,1,figsize=(10,10))
clusterer.minimum_spanning_tree_.plot(edge_cmap='viridis',
                                      edge_alpha=0.6,
                                      node_size=80,
                                      edge_linewidth=2)


#%%
morralla = '/Users/amartinez/Desktop/morralla/'
fig, ax = plt.subplots(1,1,figsize=(10,10))
clusterer.single_linkage_tree_.plot(cmap='viridis', colorbar=True)
plt.savefig(morralla + 'filename.png', dpi=300)














