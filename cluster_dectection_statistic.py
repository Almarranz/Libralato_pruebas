#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 12:22:09 2022

@author: amartinez
"""

# =============================================================================
# We are a throw a core cluster obtained on Arches_Hosel_kernel_H_Ks.py in a set 
# of simulated random data and then run dbscan on it. The goal is build and statistic 
# about the finding and the fake positives and so on.
# =============================================================================

# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from astropy.coordinates import SkyCoord
import astropy.coordinates as ap_coor
import astropy.units as u
from matplotlib import rcParams
import sys
from astropy.table import Table
from scipy.stats import gaussian_kde
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KDTree
from kneed import DataGenerator, KneeLocator
from sklearn.preprocessing import StandardScaler
import spisea
from spisea import synthetic, evolution, atmospheres, reddening, ifmr
from spisea.imf import imf, multiplicity
import pandas as pd
from astropy.table import Column
from astropy.coordinates import FK5
from astropy.stats import sigma_clip

# %%

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

#%% 
catal='/Users/amartinez/Desktop/PhD/Arches_and_Quintuplet_Hosek/'
pruebas='/Users/amartinez/Desktop/PhD/Arches_and_Quintuplet_Hosek/pruebas/'
gns_ext = '/Users/amartinez/Desktop/PhD/Libralato_data/extinction_maps/'

# =============================================================================
# #Choose Arches or Quintuplet. We neeed the real data in order to build the simulated one
# =============================================================================
choosen_cluster = 'Arches'#TODO
# choosen_cluster = 'Quintuplet'#TODO

center_arc = SkyCoord('17h45m50.4769267s', '-28d49m19.16770s', frame='icrs',obstime ='J2016.0') if choosen_cluster =='Arches' else SkyCoord('17h46m15.13s', '-28d49m34.7s', frame='icrs')#Quintuplet
# names=('Name','F127M','e_F127M','F153M','e_F153M','ra*','e_ra*','dec','e_dec','pm_ra*','e_pm_ra*','pm_dec','e_pm_dec','t0','n_epochs','dof','chi2_ra*','chi2_dec','Orig_name','Pclust')>
arches=Table.read(catal + 'Arches_cat_H22_Pclust.fits') if choosen_cluster =='Arches' else Table.read(catal + 'Quintuplet_cat_H22_Pclust.fits')

# %% Here we are going to trimm the data
# Only data with valid color and uncertainties in pm smaller than 0.4 and excluding foreground stars by color-cut
m127_all, m153_all = arches['F127M']*u.mag,arches['F153M']*u.mag
valid_colors=np.where((np.isnan(m127_all)==False)&(np.isnan(m153_all)==False))
m127,m153=m127_all[valid_colors],m153_all[valid_colors]
arches=arches[valid_colors]

center = np.where((m127.value - m153.value > 1.7) &(m127.value - m153.value < 4))
arches = arches[center]

epm_gal = SkyCoord(ra  = arches['ra*']*u.arcsec+center_arc.ra,dec = arches['dec']*u.arcsec+ center_arc.dec, pm_ra_cosdec =  arches['e_pm_ra*']*u.mas/u.yr, pm_dec = arches['e_pm_dec']*u.mas/u.yr,frame = 'icrs').galactic
pme_lim = 0.4
valid_epm = np.where((epm_gal.pm_l_cosb.value < pme_lim)&(epm_gal.pm_b.value < pme_lim))
arches=arches[valid_epm]

# %%
hos_coord = SkyCoord(ra  = arches['ra*']*u.arcsec+center_arc.ra,dec = arches['dec']*u.arcsec+ center_arc.dec,frame ='icrs',obstime ='J2016.0')
hos_gal = SkyCoord(ra = hos_coord.ra, dec = hos_coord.dec, frame = 'icrs',obstime ='J2016.0').galactic
pmra, pmdec = arches['pm_ra*']*u.mas/u.yr, arches['pm_dec']*u.mas/u.yr

columnas = len(arches.columns)
if columnas < 26:
    arches.add_column(hos_coord.ra,name='ra_abs',index=0)
    arches.add_column(hos_coord.dec,name='dec_abs',index=1)
    arches.add_column(hos_gal.l,name='l_abs',index=2)
    arches.add_column(hos_gal.b,name='b_abs',index=3)
    pm_gal = SkyCoord(ra  = arches['ra_abs'] ,dec = arches['dec_abs'], pm_ra_cosdec = pmra, pm_dec = pmdec,frame = 'icrs').galactic
    pml, pmb = pm_gal.pm_l_cosb, pm_gal.pm_b
    arches.add_column(pml.value,name='pm_l',index=4)
    arches.add_column(pmb.value,name='pm_b',index=5)
elif columnas == 26:
    print('ra and dec already added to Hoseck data: \n',arches.columns)
# %%
print(arches.columns)
# %%


#%%
# clustered_by = 'all_color'#TODO
clustered_by = 'all'#TODO
samples_dist=7
# %%
#here we generate the kernel simulated data 

colorines = arches['F127M']-arches['F153M']
pml_kernel, pmb_kernel = gaussian_kde(arches['pm_l']), gaussian_kde(arches['pm_b'])
l_kernel, b_kernel = gaussian_kde(arches['l_abs'].value), gaussian_kde(arches['b_abs'].value)
color_kernel = gaussian_kde(colorines)
f127_kernel = gaussian_kde( arches['F127M'])
f153_kernel = gaussian_kde( arches['F153M'])

# %%


lst_d_KNN_sim = []
for d in range(1):#here we are the calculate the mean of the smaller value for the NN distance of multiple simulations.
    mub_sim,  mul_sim = pmb_kernel.resample(len(pmb)), pml_kernel.resample(len(pml))
    l_sim, b_sim = l_kernel.resample(len(pml)), b_kernel.resample(len(pmb))
    f127_sim, f153_sim = f127_kernel.resample(len(arches['F127M'])), f153_kernel.resample(len(arches['F153M']))
    color_sim = color_kernel.resample(len(pml))
    if clustered_by == 'all_color':
        X_sim=np.array([mul_sim[0],mub_sim[0],l_sim[0],b_sim[0],color_sim[0]]).T
        X_stad_sim = StandardScaler().fit_transform(X_sim)
        tree_sim =  KDTree(X_stad_sim, leaf_size=2)
        
        dist_sim, ind_sim = tree_sim.query(X_stad_sim, k=samples_dist) #DistNnce to the 1,2,3...k neighbour
        d_KNN_sim=sorted(dist_sim[:,-1])#distance to the Kth neighbour
        
        lst_d_KNN_sim.append(min(d_KNN_sim))
    elif clustered_by =='all':
        X_sim=np.array([mul_sim[0],mub_sim[0],l_sim[0],b_sim[0]]).T
        X_stad_sim = StandardScaler().fit_transform(X_sim)
        tree_sim =  KDTree(X_stad_sim, leaf_size=2)
        
        dist_sim, ind_sim = tree_sim.query(X_stad_sim, k=samples_dist) #DistNnce to the 1,2,3...k neighbour
        d_KNN_sim=sorted(dist_sim[:,-1])#distance to the Kth neighbour
        
        lst_d_KNN_sim.append(min(d_KNN_sim))

d_KNN_sim_av = np.mean(lst_d_KNN_sim)
d_KNN_sim_max = np.max(lst_d_KNN_sim)
d_KNN_sim_min = np.min(lst_d_KNN_sim)



# %%
# =============================================================================
# Simulated data part
# =============================================================================
#Load the core cluster generated in Arches
from_clus = 'Arches'
# header = 'mul, mub, l, b, f127 - f153'
radio = 2
core_cluster = np.loadtxt(pruebas + 'core_cluster_rad%.0f_%s.txt'%(radio, from_clus))

#Data with cluster
data_with = np.r_[X_sim, core_cluster[:,0:-2]]
dbs_data = data_with
dbs_data_stand = StandardScaler().fit_transform(dbs_data)
tree_with =  KDTree(dbs_data_stand, leaf_size=2)
dist_with, ind_with = tree_with.query(dbs_data_stand, k=samples_dist) #DistNnce to the 1,2,3...k neighbour
d_KNN_with=sorted(dist_with[:,-1])#distance to the Kth neighbour


# Simulated from data with cluster
lst_d_KNN_sim_with = []
for d in range(20):
    pml_kernel_with, pmb_kernel_with = gaussian_kde(data_with[:,0]), gaussian_kde(data_with[:,1])
    l_kernel_with, b_kernel_with =  gaussian_kde(data_with[:,2]), gaussian_kde(data_with[:,3])
    mub_sim_with,  mul_sim_with = pmb_kernel.resample(len(data_with)), pml_kernel.resample(len(data_with))
    l_sim_with, b_sim_with = l_kernel.resample(len(data_with)), b_kernel.resample(len(data_with))
    X_sim_with = np.array([mul_sim_with[0],mub_sim_with[0],l_sim_with[0],b_sim_with[0]]).T
    X_sim_with_stand = StandardScaler().fit_transform(X_sim_with)
    tree_sim_with =  KDTree(X_sim_with_stand , leaf_size=2)
    dist_sim_with, ind_sim_with = tree_sim_with.query(X_sim_with_stand, k=samples_dist) #DistNnce to the 1,2,3...k neighbour
    d_KNN_sim_with=sorted(dist_sim_with[:,-1])#distance to the Kth neighbour
    lst_d_KNN_sim_with.append(min(d_KNN_sim_with))

d_KNN_sim_with_av = np.mean(lst_d_KNN_sim_with)

fig, ax = plt.subplots(1,1,figsize=(10,10))
ax.set_title('Number of points = %s '%(len(pml)))

# ax[0].set_title('Sub_sec_%s_%s'%(col[colum],row[ro]))
# ax[0].plot(np.arange(0,len(datos),1),d_KNN,linewidth=1,color ='k')
# ax[0].plot(np.arange(0,len(datos),1),d_KNN_sim, color = 'r')

# # ax.legend(['knee=%s, min=%s, eps=%s, Dim.=%s'%(round(kneedle.elbow_y, 3),round(min(d_KNN),2),round(epsilon,2),len(X[0]))])
# ax[0].set_xlabel('Point') 
# ax[0].set_ylabel('%s-NN distance'%(samples)) 

ax.hist(d_KNN_with,bins ='auto',histtype ='step',color = 'k')
ax.hist(d_KNN_sim_with,bins ='auto',histtype ='step',color = 'r')
ax.set_xlabel('%s-NN distance'%(samples_dist)) 

eps_av_with = round((min(d_KNN_with)+d_KNN_sim_with_av)/2,3)#TODO
# =============================================================================
# # Here we can select a epsilon bigger than the minimun distance in the simlated data
# # to find fake clusters
# eps_av_with = round(d_KNN_sim_with_av + 0.10*d_KNN_sim_with_av,3) #TODO
# =============================================================================
texto = '\n'.join(('min real d_KNN = %s'%(round(min(d_KNN_with),3)),
                    'min sim d_KNN =%s'%(round(d_KNN_sim_with_av,3)),'average = %s'%(eps_av_with)))
props = dict(boxstyle='round', facecolor='w', alpha=0.5)
# place a text box in upper left in axes coords
ax.text(0.55, 0.25, texto, transform=ax.transAxes, fontsize=20,
    verticalalignment='top', bbox=props)

# %%
# =============================================================================
# DBSCAN part
# =============================================================================
#Here to decide to use the data with the core cluster in it or just the pure 
# random data
clustering = DBSCAN(eps = eps_av_with, min_samples=samples_dist).fit(dbs_data_stand)

l_c=clustering.labels_

n_clusters = len(set(l_c)) - (1 if -1 in l_c else 0)
n_noise=list(l_c).count(-1)

u_labels = set(l_c)
colors=[plt.cm.rainbow(i) for i in np.linspace(0,1,len(set(l_c)))]# Returns a color for each cluster. Each color consists in four number, RGBA, red, green, blue and alpha. Full opacity black would be then 0,0,0,1


for k in range(len(colors)): #give noise color black with opacity 0.1
    if list(u_labels)[k] == -1:
        colors[k]=[0,0,0,0.1]
        
colores_index=[]      
for c in u_labels:
    cl_color=np.where(l_c==c)
    colores_index.append(cl_color)

nr_plot = 2
fig, ax = plt.subplots(1,nr_plot,figsize=(nr_plot*10,10))

ax[0].invert_xaxis()
# ax[2].invert_yaxis()
elements_in_cluster=[]
for i in range(len(set(l_c))-1):
    ax[0].scatter(dbs_data[:,0][colores_index[i]], dbs_data[:,1][colores_index[i]],color=colors[i],zorder=3)
    # ax[1].scatter(l[colores_index[i]], b[colores_index[i]],color=colors[i],zorder=3)
    ax[1].scatter(dbs_data[:,2][colores_index[i]],dbs_data[:,3][colores_index[i]],color=colors[i],zorder=3,s=100)
    # ax[1].scatter(gns_match[colores_index[i]][:,0],gns_match[colores_index[i]][:,2],color=colors[i],zorder=3,s=100)
    # ax[2].scatter(arches['F127M'][colores_index[i]]-arches['F153M'][colores_index[i]],arches['F153M'][colores_index[i]],color=colors[i],zorder=13)
    
ax[0].scatter(dbs_data[:,0][colores_index[-1]], dbs_data[:,1][colores_index[-1]],color=colors[-1],zorder=1)
ax[0].set_xlabel(r'$\mathrm{\mu_{l} (mas\ yr^{-1})}$',fontsize =30) 
ax[0].set_ylabel(r'$\mathrm{\mu_{b} (mas\ yr^{-1})}$',fontsize =30) 
ax[1].scatter(dbs_data[:,2][colores_index[-1]],dbs_data[:,3][colores_index[-1]],color=colors[-1],zorder=1)
# ax[1].scatter(arches[colores_index[-1]]['l_abs'],arches[colores_index[-1]]['b_abs'],color=colors[-1],zorder=3,s=100,alpha = 0.01)
ax[1].set_xlabel('ra(deg)',fontsize =30) 
ax[1].set_ylabel('dec(deg)',fontsize =30)

# ax[2].scatter(arches['F127M'][colores_index[-1]]-arches['F153M'][colores_index[-1]],arches['F153M'][colores_index[-1]],color=colors[-1],zorder=1)
# ax[2].set_xlabel('f127m-f153m',fontsize =30) 
# ax[2].set_ylabel('f153m',fontsize =30)  





# %%















