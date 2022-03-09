#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 26 20:25:11 2022

@author: amartinez
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import pandas as pd
from sklearn.neighbors import KDTree
from kneed import DataGenerator, KneeLocator


pruebas='/Users/amartinez/Desktop/PhD/Libralato_data/pruebas/'





iris_df=pd.read_csv(pruebas+'iris_no_names.csv', names=['sepallength', 'sepalwidth',	'petallength','petalwidth', 'class'])
                     
iris=iris_df.to_numpy()
iris=iris[:,0:4]
# %%
# print(iris[:1])
# %%

# tree=KDTree(iris[:,0:2], leaf_size=2) 
tree=KDTree(iris, leaf_size=2) 

# dist, ind = tree.query(iris[:,0:2], k=5)
dist, ind = tree.query(iris, k=5)

# %%
a=60
print(ind[a])
# %%

# fig, ax = plt.subplots(1,1,figsize=(15,15))
# ax.scatter(iris[:,0],iris[:,1])
# ax.scatter(iris[a,0],iris[a,1],color='red')
# for i in range(len(iris)):
#     ax.text(iris[i,0],iris[i,1],i)
# %%

four_KNN=sorted(dist[:,-1])

fig, ax = plt.subplots(1,1,figsize=(15,15))
ax.plot(np.arange(0,len(iris),1),four_KNN)

# %%
kneedle = KneeLocator(np.arange(0,len(iris),1), four_KNN, curve='convex')

print(round(kneedle.knee, 3))
print(round(kneedle.elbow, 3))
print(round(kneedle.knee_y, 3))
print(round(kneedle.elbow_y, 3))
