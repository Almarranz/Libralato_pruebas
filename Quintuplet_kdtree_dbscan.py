#!/usr/bin/env python
# coding: utf-8

# In[1]:


# %matplotlib inline


# In[2]:


print(__doc__)

import numpy as np

from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import matplotlib.pylab as plt

from scipy import stats

import pandas as pd
# #############################################################################

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





print(len(c))

X = np.vstack((c,d)).T  

plt.figure(figsize=(7,7))

plt.xlim(-10, 10)
plt.ylim(-10,10)


plt.ylabel('v_y(mas/yr)', fontsize=13)
plt.xlabel('v_x(mas/yr)', fontsize=13)   


plt.plot(X[:,0], X[:,1], "ko", markersize = 4)
# plt.savefig("/Users/banafsh/Python/Machine_Learning_practice/vxvy_Quintuplet.pdf")


# #############################################################################


plt.show()


Z = np.vstack((a,b)).T


# In[3]:


#########################################


#scaling features
X = StandardScaler().fit_transform(X)


# Compute KD-tree
vecinos=10 # vecinos was originally 10 (from Ban)

nn = NearestNeighbors(n_neighbors=vecinos, algorithm ='kd_tree')
nn.fit(X)# our training is basically our dataset itself



dist, ind = nn.kneighbors(X,vecinos)


distances = np.sort(dist, axis=0)


distances = distances[:,-1]# In here she uses 1, but shouldn't it be -1, that is the distance to the kneism neighbour?

print('This is dis:',dist)
print('This is disances:',distances)


# plt.title('Distance variation at the 10th Neighbour')




epsilon=[]
for i in range(len(dist)):
 
    epsilon.append(np.mean(dist[i]))



#print "np.mean(epsilon)=", np.mean(epsilon)

n =[1, 3, 5, 10, 20, 30, 60, 90]
eps = [0.27,0.31,0.347,0.365,0.354,0.344,0.31,0.29]

#plt.plot(n, eps, "bo")

#plt.ylabel('epsilon', fontsize=13)
#plt.xlabel('n_bins', fontsize=13) 

#plt.savefig("/Users/banafsheh/Python/Quintuplet/epsilon.pdf")
# plt.show()


from kneed import DataGenerator, KneeLocator

kneedle = KneeLocator(list(range(len(c))), distances, curve="convex", direction="increasing", interp_method = "polynomial")


knee_x = round(kneedle.knee, 3)
knee_y = round(kneedle.knee_y, 3)

print(knee_x)
print(knee_y)


plt.figure(figsize=(6,6))
plt.plot(distances, 'darkblue')
plt.xlabel("Points sorted by distance to the 10th nearest neighbour", fontsize = 11)
plt.ylabel("Distance", fontsize = 11)

plt.xlim(0, len(c))
plt.ylim(0, max(distances))


plt.hlines(knee_y, 0, knee_x, colors='k', linestyles='dashed', lw=2)


# plt.savefig("/Users/banafsh/Python/Machine_Learning_practice/Distance_curve_vxvy_Quintuplet.pdf")


plt.show()


# In[4]:


# #############################################################################
# Compute DBSCAN
# db = DBSCAN(eps=0.2, min_samples=10).fit(X)
db = DBSCAN(eps=knee_y, min_samples=10).fit(X)

#idenstifying the points which makes up our core points
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)

# #############################################################################
# Plot result
plt.figure(figsize=(7,7))


# for plotting I use the non-scaled X
X = np.vstack((c,d)).T  


# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = [plt.cm.hsv(each)
          for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)
    
    
    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=7, alpha = 0.7)

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=4, alpha = 0.7)
    
plt.xlim(-10, 10)
plt.ylim(-10,10)


plt.ylabel('v_y(mas/yr)', fontsize=13)
plt.xlabel('v_x(mas/yr)', fontsize=13)    
    
plt.title('DBSCAN')
# plt.savefig("/Users/banafsh/Python/Machine_Learning_practice/DBSCAN_vxvy_Quintuplet.pdf")

plt.show()

# #############################################################################
# Plot result
plt.figure(figsize=(7,7))


# for plotting I use the non-scaled X


# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = [plt.cm.hsv(each)
          for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    mum = mu[class_member_mask & core_samples_mask]
    plt.plot(mum[:, 0], mum[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=7, alpha =0.7)

    mum = mu[class_member_mask & ~core_samples_mask]
    plt.plot(mum[:, 0], mum[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=4, alpha =0.7)

plt.xlim(10, -10)
plt.ylim(-10,10)    
 
    
plt.ylabel(r'$\mu_{\delta}$ [mas/yr]', fontsize=16)
plt.xlabel(r'$\mu_{\alpha}$ [mas/yr]', fontsize=16)       

    
plt.title('DBSCAN')

# plt.savefig("/Users/banafsh/Python/Machine_Learning_practice/DBSCAN_mu_Quintuplet.pdf")

plt.show()


################## plotting x-y based on the values obtained from the above cluster


plt.figure(figsize=(7,7))



for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    ab = Z[class_member_mask & core_samples_mask]
    plt.plot(ab[:, 0], ab[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=7, alpha = 0.7)

    ab = Z[class_member_mask & ~core_samples_mask]
    plt.plot(ab[:, 0], ab[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=4, alpha = 0.7)

 



plt.ylabel('y [pixels]', fontsize=16)
plt.xlabel('x [pixels]', fontsize=16)    
    
# plt.title('Estimated number of clusters: %d' % n_clusters_)
# plt.savefig("/Users/banafsh/Python/Machine_Learning_practice/xy_equivalent_vxvy_Quintuplet.pdf")

plt.show()


# In[17]:


print(dist[0])


# In[6]:


print(distances)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




