#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 20 11:28:52 2022

@author: amartinez
"""
# =============================================================================
# In BAN catalog (the goal is find some common cluster with Libralato)
# Here we are going to divide section A in smalles LxL areas, thar overlap. Then
# we´ll run dbs with the kernel method over the first of thes boxes, store the cluster 
# if we like in a particular folder called 'cluster1' and continue with the nex box.
# If we found the same (or partially the same) cluster in an overlapping box, we will store it
# in the same folder 'cluster 1', an so on
# =============================================================================
# %%imports
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.neighbors import KDTree
from kneed import DataGenerator, KneeLocator
from matplotlib.ticker import FormatStrFormatter
import pandas as pd
import sys
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import QTable
import pylab as p
from random import seed
from random import random
import glob
from sklearn.preprocessing import StandardScaler
import os
import math
from scipy.stats import gaussian_kde
import shutil
from datetime import datetime
import astropy.coordinates as ap_coor
import spisea
from spisea import synthetic, evolution, atmospheres, reddening, ifmr
from spisea.imf import imf, multiplicity
from astropy.coordinates import FK5
from sklearn.decomposition import PCA
import pdb
from sklearn.preprocessing import RobustScaler
# %%plotting parametres
rcParams.update({'xtick.major.pad': '7.0'})
rcParams.update({'xtick.major.size': '7.5'})
rcParams.update({'xtick.major.width': '1.5'})
rcParams.update({'xtick.minor.pad': '7.0'})
rcParams.update({'xtick.minor.size': '3.5'})
rcParams.update({'xtick.minor.width': '1.0'})
rcParams.update({'ytick.major.pad': '7.0'})
rcParams.update({'ytick.major.size': '7.5'})
rcParams.update({'ytick.major.width': '1.5'})
rcParams.update({'ytick.minor.pad': '7.0'})
rcParams.update({'ytick.minor.size': '3.5'})
rcParams.update({'ytick.minor.width': '1.0'})
rcParams.update({'font.size': 20})
rcParams.update({'figure.figsize':(10,5)})
rcParams.update({
    "text.usetex": False,
    "font.family": "sans",
    "font.sans-serif": ["Palatino"]})
plt.rcParams["mathtext.fontset"] = 'dejavuserif'
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Palatino']})
plt.rcParams.update({'figure.max_open_warning': 0})# a warniing for matplot lib pop up because so many plots, this turining it of
# %%
cata='/Users/amartinez/Desktop/PhD/Libralato_data/CATALOGS/'
pruebas='/Users/amartinez/Desktop/PhD/Libralato_data/pruebas/'
results='/Users/amartinez/Desktop/PhD/Libralato_data/results/'
gns_ext = '/Users/amartinez/Desktop/PhD/Libralato_data/extinction_maps/'

section = 'A'#selecting the whole thing

MS_ra,MS_dec = np.loadtxt(cata + 'MS_section%s.txt'%(section),unpack=True, usecols=(0,1),skiprows=0)
MS_coord = SkyCoord(ra = MS_ra*u.deg, dec = MS_dec*u.deg, frame = FK5,equinox ='J2014.2')
# ra, dra, dec, ddec, j, dj, h, dh, k, dk, v_x, dv_x, v_y, dv_y, mu_alpha, mu_delta, mu_l, dmu_l, mu_b, dmu_b
catal = np.loadtxt(cata + 'BAN_on_sect%s.txt'%(section))
# %%

valid=np.where((np.isnan(catal[:,6])<90) & (np.isnan(catal[:,8])<90))
catal=catal[valid]
center=np.where(catal[:,6]-catal[:,8]>1.3)
catal=catal[center]
dmu_lim = 1
vel_lim = np.where((catal[:,-3]<=dmu_lim) & (catal[:,-1]<=dmu_lim))
catal=catal[vel_lim]

# 'ra dec x_c  y_c mua dmua mud dmud time n1 n2 ID mul mub dmul dmub '
# catal_all = np.loadtxt(cata + '%s_pm_galactic.txt'%(name))

# %%Thi is for the extinction

Aks_gns = pd.read_fwf(gns_ext + 'central.txt', sep =' ',header = None)

# %
AKs_np = Aks_gns.to_numpy()#TODO
center = np.where(AKs_np[:,6]-AKs_np[:,8] > 1.3)#TODO
AKs_center =AKs_np[center]#TODO
# %
gns_coord = SkyCoord(ra=AKs_center[:,0]*u.degree, dec=AKs_center[:,2]*u.degree)
# %
# %
AKs_list1 =  np.arange(1.6,2.11,0.01)
AKs_list = np.append(AKs_list1,0)#I added the 0 for the isochrones without extiction
# %%
color = pd.read_csv('/Users/amartinez/Desktop/PhD/python/colors_html.csv')
strin= color.values.tolist()
indices = np.arange(0,len(strin),1)
# %%

fig, ax = plt.subplots(1,1, figsize= (10,10))
ax.scatter(catal[:,0],catal[:,2])
# ax.set_ylim(-28.94,-28.88)
# ax.set_xlim(266.51,266.53)   
# %
m1 =-1
m = 60/73
yr_1 = 237.578 + m1*catal[:,0]# yg_1 = (lim_pos_up - (ic)*step/np.cos(45*u.deg)) +  m*catal[:,7]
yr_2 = 237.700 + m1*catal[:,0]

yg_1 = - 247.923 + m*catal[:,0]# yg_1 = (lim_pos_up - (ic)*step/np.cos(45*u.deg)) +  m*catal[:,7]
yg_2 = - 248.023 + m*catal[:,0]
                      
# y = 0.45x - 148.813
ax.scatter(catal[:,0],yg_1,color ='g')
ax.scatter(catal[:,0],yg_2,color ='g')
ax.scatter(catal[:,0],yr_1,color = 'r')
ax.scatter(catal[:,0],yr_2,color = 'r')


# %%
lim_pos_up, lim_pos_down = - 247.923, - 248.023 #intersection of the positives slopes lines with y axis,
lim_neg_up, lim_neg_down = 237.700,237.578
# %%

dist_pos = abs((-m*catal[0,0]+ (lim_pos_down + m*catal[0,0])-lim_pos_up)/np.sqrt((-1)**2+(1)**2))
ang = (np.arctan(m))
# distancia entre yg_up e yg_down
dist_neg = abs((-m1*catal[0,0]+ (lim_neg_down + m1*catal[0,0])-lim_neg_up)/np.sqrt((-1)**2+(1)**2))
# ang = math.degrees(np.arctan(m1))
ang1 = (np.arctan(m1))
# %%


clustered_by_list =['all_color','all']
# clustered_by_list =['all']

# x_box_lst = [1,2,3]
# samples_lst =[10, 9, 8,7]
x_box_lst = [1,2]
samples_lst =[10]
for clus_lista in clustered_by_list:
    clustered_by = clus_lista
    for x_box in x_box_lst:
        step = dist_pos /x_box
        step_neg =dist_neg/x_box
        
        for samples_dist in samples_lst:
           
            for ic in range(x_box*2-1):
                # fig, ax = plt.subplots(1,1, figsize= (10,10))
                # ax.scatter(catal[:,0],catal[:,2])
                ic *= 0.5
                yg_1 = (lim_pos_up - (ic)*step/np.cos(ang)) +  m*catal[:,0]
                # yg_2 = (lim_pos_up - (ic+1)*step*np.cos(45*u.deg)) +  m*catal[:,7]
                yg_2 = (lim_pos_up - (ic+1)*step/np.cos(ang)) +  m*catal[:,0]
                # ax.plot(catal[:,0],yg_1, color ='g')
                # ax.plot(catal[:,0],yg_2, color ='g')
                for jr in range(x_box*2-1):
                    fig, ax = plt.subplots(1,1, figsize=(10,10))
                    ax.scatter(catal[:,0],catal[:,2],alpha =0.1)
                    jr *=0.5
                    yr_1 = (lim_neg_up - (jr)*step_neg/np.cos(ang1)) +  m1*catal[:,0]
                    # yg_2 = (lim_pos_up - (i+1)*step*np.cos(45*u.deg)) +  m*catal[:,7]
                    yr_2 = (lim_neg_up - (jr+1)*step_neg/np.cos(ang1)) +  m1*catal[:,0]
                    ax.plot(catal[:,0],yg_1, color ='g')
                    ax.plot(catal[:,0],yg_2, color ='g')
                    ax.plot(catal[:,0],yr_1, color ='r')
                    ax.plot(catal[:,0],yr_2, color ='r')            













                    