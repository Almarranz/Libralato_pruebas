#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 10 16:57:10 2022

@author: amartinez
"""

# =============================================================================
# Looks for the cluster in Hoseck`s data and then select the very core of it and 
# then matches the cluster with gns survey and using spisea plots a simulated cluster.
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
# #Choose Arches or Quintuplet
# =============================================================================
choosen_cluster = 'Arches'#TODO
# choosen_cluster = 'Quintuplet'#TODO

center_arc = SkyCoord('17h45m50.4769267s', '-28d49m19.16770s', frame='icrs') if choosen_cluster =='Arches' else SkyCoord('17h46m15.13s', '-28d49m34.7s', frame='icrs')#Quintuplet
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
# =============================================================================
# Here we are going to match with GNS
# =============================================================================
Aks_gns = pd.read_fwf(gns_ext + 'central.txt', sep =' ',header = None)

# %
AKs_np = Aks_gns.to_numpy()
center = np.where(AKs_np[:,6]-AKs_np[:,8] > 1.3)
AKs_center =AKs_np[center]
# %
gns_coord = SkyCoord(ra=AKs_center[:,0]*u.degree, dec=AKs_center[:,2]*u.degree,frame = FK5,equinox ='J2000')


# %% I cosider a math if the stars are less than 1 arcsec away 
hos_coor = SkyCoord(ra  = arches['ra*']*u.arcsec+center_arc.ra,dec = arches['dec']*u.arcsec+ center_arc.dec)
idi, d2d, d3d = hos_coor.match_to_catalog_sky(gns_coord)
is_match = np.where(d2d<1*u.arcsec)

hos_match_coord=hos_coor[is_match]
hos_match = arches[is_match[0]]
gns_match = AKs_center[idi[is_match]]
# %%
hos_match_gal = SkyCoord(ra = hos_match_coord.ra, dec = hos_match_coord.dec, frame = 'icrs').galactic
pmra, pmdec = hos_match['pm_ra*']*u.mas/u.yr, hos_match['pm_dec']*u.mas/u.yr

columnas = len(hos_match.columns)
if columnas < 26:
    hos_match.add_column(hos_match_coord.ra,name='ra_abs',index=0)
    hos_match.add_column(hos_match_coord.dec,name='dec_abs',index=1)
    hos_match.add_column(hos_match_gal.l,name='l_abs',index=2)
    hos_match.add_column(hos_match_gal.b,name='b_abs',index=3)
    pm_gal = SkyCoord(ra  = hos_match['ra_abs'] ,dec = hos_match['dec_abs'], pm_ra_cosdec = pmra, pm_dec = pmdec,frame = 'icrs').galactic
    pml, pmb = pm_gal.pm_l_cosb, pm_gal.pm_b
    hos_match.add_column(pml.value,name='pm_l',index=4)
    hos_match.add_column(pmb.value,name='pm_b',index=5)
elif columnas == 26:
    print('ra and dec already added to Hoseck data: \n',hos_match.columns)
# %%
print(hos_match.columns)
# %%
#Checking the matching, you can delete these tree lines
fig, ax = plt.subplots(1,2, figsize = (20,10))
ax[0].scatter(hos_match['ra_abs'], hos_match['dec_abs'])
ax[0].scatter(gns_match[:,0],gns_match[:,2],color = 'r',s=1)
ax[1].scatter(hos_match['F127M']-hos_match['F153M'], hos_match['F153M'])
ax[1].invert_yaxis()
# %%
# =============================================================================
# This is for checking that the stars in both catalog are match-sorted
# you could erase this,  but you know you will check this fact in the future
# =============================================================================
tc =200
print((gns_match[tc,0]-hos_match['ra_abs'][tc])*u.deg.to(u.arcsec))
#%%
clustered_by = 'all_color'#TODO
# clustered_by = 'all'#TODO
samples_dist=7
# %%
#here we generate the kernel simulated data 

colorines = hos_match['F127M']-hos_match['F153M']
pml_kernel, pmb_kernel = gaussian_kde(hos_match['pm_l']), gaussian_kde(hos_match['pm_b'])
l_kernel, b_kernel = gaussian_kde(hos_match['l_abs'].value), gaussian_kde(hos_match['b_abs'].value)
color_kernel = gaussian_kde(colorines)
# %%
pml, pmb = hos_match['pm_l'],hos_match['pm_b']
l,b = hos_match['l_abs'].value,  hos_match['b_abs'].value
if clustered_by == 'all_color':
    X = np.array([pml,pmb,l,b,colorines]).T
    X_stad = StandardScaler().fit_transform(X)
    tree = KDTree(X_stad, leaf_size=2) 
    dist, ind = tree.query(X_stad, k=samples_dist) #DistNnce to the 1,2,3...k neighbour
    d_KNN=sorted(dist[:,-1])#distance to the Kth neighbour
elif clustered_by == 'all':
    X = np.array([pml,pmb,l,b]).T
    X_stad = StandardScaler().fit_transform(X)
    tree = KDTree(X_stad, leaf_size=2) 
    dist, ind = tree.query(X_stad, k=samples_dist) #DistNnce to the 1,2,3...k neighbour
    d_KNN=sorted(dist[:,-1])#distance to the Kth neighbour


lst_d_KNN_sim = []
for d in range(20):
    mub_sim,  mul_sim = pmb_kernel.resample(len(pmb)), pml_kernel.resample(len(pml))
    l_sim, b_sim = l_kernel.resample(len(pml)), b_kernel.resample(len(pmb))
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


fig, ax = plt.subplots(1,1,figsize=(10,10))
ax.set_title('Number of points = %s '%(len(pml)))

# ax[0].set_title('Sub_sec_%s_%s'%(col[colum],row[ro]))
# ax[0].plot(np.arange(0,len(datos),1),d_KNN,linewidth=1,color ='k')
# ax[0].plot(np.arange(0,len(datos),1),d_KNN_sim, color = 'r')

# # ax.legend(['knee=%s, min=%s, eps=%s, Dim.=%s'%(round(kneedle.elbow_y, 3),round(min(d_KNN),2),round(epsilon,2),len(X[0]))])
# ax[0].set_xlabel('Point') 
# ax[0].set_ylabel('%s-NN distance'%(samples)) 

ax.hist(d_KNN,bins ='auto',histtype ='step',color = 'k')
ax.hist(d_KNN_sim,bins ='auto',histtype ='step',color = 'r')
ax.set_xlabel('%s-NN distance'%(samples_dist)) 

eps_av = round((min(d_KNN)+d_KNN_sim_av)/2,3)
texto = '\n'.join(('min real d_KNN = %s'%(round(min(d_KNN),3)),
                    'min sim d_KNN =%s'%(round(d_KNN_sim_av,3)),'average = %s'%(eps_av)))


props = dict(boxstyle='round', facecolor='w', alpha=0.5)
# place a text box in upper left in axes coords
ax.text(0.55, 0.25, texto, transform=ax.transAxes, fontsize=20,
    verticalalignment='top', bbox=props)

ax.set_ylabel('N') 
# %%

# =============================================================================
# DBSCAN part
# =============================================================================
epsilon = eps_av
clustering = DBSCAN(eps = epsilon, min_samples=samples_dist).fit(X_stad)

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
    
fig, ax = plt.subplots(1,3,figsize=(30,10))

ax[0].invert_xaxis()
ax[2].invert_yaxis()
elements_in_cluster=[]
for i in range(len(set(l_c))-1):
    ax[0].scatter(pml[colores_index[i]], pmb[colores_index[i]],color=colors[i],zorder=3)
    # ax[1].scatter(l[colores_index[i]], b[colores_index[i]],color=colors[i],zorder=3)
    ax[1].scatter(hos_match[colores_index[i]]['ra_abs'],hos_match[colores_index[i]]['dec_abs'],color=colors[i],zorder=3)
    ax[2].scatter(hos_match['F127M'][colores_index[i]]-hos_match['F153M'][colores_index[i]],hos_match['F153M'][colores_index[i]],color=colors[i],zorder=13)
    
ax[0].scatter(pml[colores_index[-1]], pmb[colores_index[-1]],color=colors[-1],zorder=1)
# ax[1].scatter(l[colores_index[-1]], b[colores_index[-1]],color=colors[-1],zorder=1)
ax[2].scatter(hos_match['F127M'][colores_index[-1]]-hos_match['F153M'][colores_index[-1]],hos_match['F153M'][colores_index[-1]],color=colors[-1],zorder=1)
# %
# gns_cluster = gns_match[colores_index[0]]

# fix, ax = plt.subplots(1,1, figsize=(10,10))
# ax.scatter(gns_cluster[colores_index[0]][:,0]['ra_abs'],hos_match[colores_index[0]]['dec'],s=20)
# ax.scatter(gns_cluster[colores_index[0]][:,0],gns_cluster[colores_index[0]][:,2],color = 'red',s =5)


# %%
# print(hos_match[colores_index[0]]['dec_abs'])
ind_t=4
fix, ax = plt.subplots(1,1, figsize=(10,10))
for ind_t in range(5):
    
    ax.scatter(hos_match[colores_index[ind_t]]['ra_abs'],hos_match[colores_index[ind_t]]['dec_abs'],s=20)

print(len(set(l_c))-1)







