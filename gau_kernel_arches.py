#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 26 13:57:25 2022

@author: amartinez
"""
# =============================================================================
# Here we are test the kernel gaussian thing for choosing epsiloon
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
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KDTree
from kneed import DataGenerator, KneeLocator

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
rcParams.update({'font.size': 40})
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
# =============================================================================
# #Choose Arches or Quintuplet
# =============================================================================
choosen_cluster = 'Arches'

center_arc = SkyCoord('17h45m50.4769267s', '-28d49m19.16770s', frame='icrs') if choosen_cluster =='Arches' else SkyCoord('17h46m15.13s', '-28d49m34.7s', frame='icrs')#Quintuplet
arches=Table.read(catal + 'Arches_cat_H22_Pclust.fits') if choosen_cluster =='Arches' else Table.read(catal + 'Quintuplet_cat_H22_Pclust.fits')
# %% Here we are going to trimm the data
# Only data with valid color and uncertainties in pm smaller than 0.4
m127_all, m153_all = arches['F127M']*u.mag,arches['F153M']*u.mag
valid_colors=np.where((np.isnan(m127_all)==False)&(np.isnan(m153_all)==False))
m127,m153=m127_all[valid_colors],m153_all[valid_colors]
arches=arches[valid_colors]

center = np.where((m127.value - m153.value > 1.7) &(m127.value - m153.value < 4))
arches = arches[center]

epm_gal = SkyCoord(ra  = arches['ra*']*u.arcsec+center_arc.ra,dec = arches['dec']*u.arcsec+ center_arc.dec, pm_ra_cosdec =  arches['e_pm_ra*']*u.mas/u.yr, pm_dec = arches['e_pm_dec']*u.mas/u.yr,frame = 'icrs').galactic
pme_lim = 2
valid_epm = np.where((epm_gal.pm_l_cosb.value < pme_lim)&(epm_gal.pm_b.value < pme_lim))
arches=arches[valid_epm]


# %%
columnas=str(arches.columns)
arc_coor=SkyCoord(ra=arches['ra*']*u.arcsec+center_arc.ra,dec=arches['dec']*u.arcsec+ center_arc.dec)
# %%
ra, dec =arc_coor.ra, arc_coor.dec
e_ra,e_dec = arches['e_ra*']*u.arcsec, arches['e_dec']*u.arcsec
# %%
pmra, pmdec = arches['pm_ra*']*u.mas/u.yr, arches['pm_dec']*u.mas/u.yr
e_pmra, e_pmdec = arches['e_pm_ra*'].value, arches['e_pm_dec'].value
print(np.std(e_pmra),np.std(e_pmdec))
# %%
m127, m153 = arches['F127M']*u.mag,arches['F153M']*u.mag

# =============================================================================
# np.savetxt(pruebas + 'arches_for_topcat.txt',np.array([ra.value,dec.value,pmra.value,pmdec.value,m127.value,m153.value]).T,header='ra.value,dec.value,pmra.value,pmdec.value,m127.value,m153.value')
# =============================================================================
# %%
arc_gal=arc_coor.galactic
pm_gal = SkyCoord(ra  = ra ,dec = dec, pm_ra_cosdec = pmra, pm_dec = pmdec,frame = 'icrs').galactic


l,b=arc_gal.l, arc_gal.b
pml,pmb=pm_gal.pm_l_cosb, pm_gal.pm_b
# %% Definition section
def plotting(namex,namey,x,y,ind,**kwargs):

    pl=ax[ind].scatter(x,y,**kwargs)
    
    try:
        ax[ind].set_xlabel('%s(%s)'%(namex,x.unit)) # Set the axis label in the form "Variable description [units]"
        ax[ind].set_ylabel('%s(%s)'%(namey, y.unit))
    except:
        ax[ind].set_xlabel('%s'%(namex)) 
        ax[ind].set_ylabel('%s'%(namey))
    if ind ==2:
        ax[ind].invert_yaxis()
    return pl
# %
def plotting_h(namex,namey,x,y,ind,**kwargs):
    try:
        pl=ax[ind].hexbin(x.value,y.value,**kwargs)
    except:
        pl=ax[ind].hexbin(x,y,**kwargs)
    try:
        ax[ind].set_xlabel('%s(%s)'%(namex,x.unit)) # Set the axis label in the form "Variable description [units]"
        ax[ind].set_ylabel('%s(%s)'%(namey, y.unit))
    except:
        ax[ind].set_xlabel('%s'%(namex)) 
        ax[ind].set_ylabel('%s'%(namey))
    if ind ==2:
        ax[ind].invert_yaxis()
    if ind ==1:
        ax[ind].invert_xaxis()
    return pl

# %
def density_plot(a,b,namex, namey, ind, **kwargs):
    
    xy = np.vstack([a,b])
    z = gaussian_kde(xy)(xy)
    pl =ax[ind].scatter(a, b, c=z,**kwargs)
    try:
        ax[ind].set_xlabel('%s(%s)'%(namex,a.unit)) # Set the axis label in the form "Variable description [units]"
        ax[ind].set_ylabel('%s(%s)'%(namey, b.unit))
    except:
        ax[ind].set_xlabel('%s'%(namex)) 
        ax[ind].set_ylabel('%s'%(namey))
    if ind ==2:
        ax[ind].invert_yaxis()
    if ind ==1:
        ax[ind].invert_xaxis()
    return pl
# %%This is the plotting section
# fig, ax = plt.subplots(1,3,figsize=(30,10))
# plotting('ra','dec',ra,dec,0,alpha=0.5)
# plotting('mura','mudec',pmra,pmdec,1,alpha=0.01)
# plotting('m127-m157','m157',m127-m153,m153,2,alpha=0.05)
# # %%
# fig, ax = plt.subplots(1,3,figsize=(30,10))
# plotting_h('ra','dec',ra,dec,0,bins=50,norm=matplotlib.colors.LogNorm())
# plotting_h('mura','mudec',pmra,pmdec,1,bins=50,norm=matplotlib.colors.LogNorm())
# plotting('m127-m157','m157',m127-m153,m153,2,alpha=0.2)

# %%
def plotting(namex,namey,x,y,ind,**kwargs):

    pl=ax[ind].scatter(x,y,**kwargs)
    
    try:
        ax[ind].set_xlabel('%s(%s)'%(namex,x.unit)) # Set the axis label in the form "Variable description [units]"
        ax[ind].set_ylabel('%s(%s)'%(namey, y.unit))
    except:
        ax[ind].set_xlabel('%s'%(namex)) 
        ax[ind].set_ylabel('%s'%(namey))
    if ind ==2:
        ax[ind].invert_yaxis()
    return pl


# =============================================================================
# Here I going to asume that, the closer to the arches center, the more cluster member is a star
# =============================================================================

radio = 80*u.arcsec
id_clus, id_arc, d2d,d3d = ap_coor.search_around_sky(SkyCoord(['17h45m50.4769267s'], ['-28d49m19.16770s'], frame='icrs'),arc_gal, radio) if choosen_cluster =='Arches' else ap_coor.search_around_sky(SkyCoord(['17h46m15.13s'], ['-28d49m34.7s'], frame='icrs'),arc_gal, radio)


fig, ax = plt.subplots(1,3,figsize=(30,10))
ax[1].set_title('Radio = %s, Green = %s'%(radio,len(id_clus)))
plotting('l','b',arc_gal.l, arc_gal.b,1)
plotting('l','b',arc_gal.l[id_arc], arc_gal.b[id_arc],1,alpha=0.9,color='g')

plotting('mul','mub',pm_gal.pm_l_cosb, pm_gal.pm_b,0)
plotting('mul','mub',pml[id_arc], pmb[id_arc],0,alpha=0.1)
ax[0].invert_xaxis()

plotting('m127-m153','m153',m127-m153, m153,2,zorder=1,alpha=0.01)
plotting('m127-m153','m153',m127[id_arc]-m153[id_arc],m153[id_arc],2,alpha=0.8,color='g')
ax[2].invert_yaxis()

# %%
# now the kennel simulations.

mul, mub = pml[id_arc].value, pmb[id_arc].value
x,y = arc_gal.l[id_arc], arc_gal.b[id_arc]
color = m127[id_arc]-m153[id_arc]

mul_kernel, mub_kernel = gaussian_kde(mul), gaussian_kde(mub)
x_kernel, y_kernel = gaussian_kde(x), gaussian_kde(y)
color_kernel = gaussian_kde(color)

# %
mub_sim,  mul_sim = mub_kernel.resample(len(id_arc)), mul_kernel.resample(len(id_arc))
x_sim, y_sim = x_kernel.resample(len(id_arc)), y_kernel.resample(len(id_arc))
color_sim = color_kernel.resample(len(id_arc))
# %
samples = 7

X=np.array([mul,mub,x,y,color]).T
X_stad = StandardScaler().fit_transform(X)
tree = KDTree(X_stad, leaf_size=2) 
dist, ind = tree.query(X_stad, k=samples) #DistNnce to the 1,2,3...k neighbour
d_KNN=sorted(dist[:,-1])#distance to the Kth neighbour

X_sim=np.array([mul_sim[0],mub_sim[0],x_sim[0],y_sim[0],color_sim[0]]).T
X_stad_sim = StandardScaler().fit_transform(X_sim)
tree_sim =  KDTree(X_stad_sim, leaf_size=2)

dist_sim, ind_sim = tree_sim.query(X_stad_sim, k=samples) #DistNnce to the 1,2,3...k neighbour
d_KNN_sim=sorted(dist_sim[:,-1])#distance to the Kth neighbou

# %%Here we work with paramters than can not have any clustered data to prove that the simulated and the real distribtion will be the same
# =============================================================================
# mul, mub = pml[id_arc].value, pmb[id_arc].value
# x,y = arc_gal.l[id_arc], arc_gal.b[id_arc]
# color = m127[id_arc]-m153[id_arc]
# 
# mul_kernel, mub_kernel = gaussian_kde(mul), gaussian_kde(mub)
# x_kernel, y_kernel = gaussian_kde(x), gaussian_kde(y)
# color_kernel = gaussian_kde(color)
# 
# # %
# mub_sim,  mul_sim = mub_kernel.resample(len(id_arc)), mul_kernel.resample(len(id_arc))
# x_sim, y_sim = x_kernel.resample(len(id_arc)), y_kernel.resample(len(id_arc))
# color_sim = color_kernel.resample(len(id_arc))
# # %
# samples = 7
# 
# # X=np.array([x,y]).T
# # X=np.array([mul,mub]).T
# X=np.array([mul,y]).T
# 
# X_stad = StandardScaler().fit_transform(X)
# tree = KDTree(X_stad, leaf_size=2) 
# dist, ind = tree.query(X_stad, k=samples) #DistNnce to the 1,2,3...k neighbour
# d_KNN=sorted(dist[:,-1])#distance to the Kth neighbour
# 
# # X_sim=np.array([x_sim[0],y_sim[0]]).T
# # X_sim=np.array([mul_sim[0],mub_sim[0]]).T
# X_sim=np.array([mul_sim[0],y_sim[0]]).T
# 
# 
# X_stad_sim = StandardScaler().fit_transform(X_sim)
# tree_sim =  KDTree(X_stad_sim, leaf_size=2)
# 
# dist_sim, ind_sim = tree_sim.query(X_stad_sim, k=samples) #DistNnce to the 1,2,3...k neighbour
# d_KNN_sim=sorted(dist_sim[:,-1])#distance to the Kth neighbour
# =============================================================================

# %%

fig, ax = plt.subplots(1,1,figsize=(20,10))

ax.hist(d_KNN,bins ='auto',histtype ='step',color = 'k',label = 'real',linewidth=5)
# ax.hist(d_KNN_sim,bins ='auto',histtype ='step',color = 'r', label = 'simulated',linewidth=5,alpha =0.5)
ax.set_xlabel('%s-NN distance'%(samples))
ax.set_ylabel('N')
ax.set_xlim(0,2)
ax.legend()
plt.savefig('/Users/amartinez/Desktop/PhD/Charlas/Presentaciones/Brno/' + 'arches_kernel0_arc.png', dpi=300,bbox_inches='tight')

fig, ax = plt.subplots(1,1,figsize=(20,10))

ax.hist(d_KNN,bins ='auto',histtype ='step',color = 'k',label = 'real',linewidth=5)
ax.hist(d_KNN_sim,bins ='auto',histtype ='step',color = 'r', label = 'simulated',linewidth=5,alpha =0.5)
ax.set_xlabel('%s-NN distance'%(samples))
ax.set_ylabel('N')
ax.set_xlim(0,0.05)
ax.legend()
# plt.savefig('/Users/amartinez/Desktop/PhD/Charlas/Presentaciones/Brno/' + 'arches_kernel0.png', dpi=300,bbox_inches='tight')
#%%

fig, ax = plt.subplots(1,1,figsize=(20,10))

ax.hist(d_KNN,bins ='auto',histtype ='step',color = 'k',label = 'real',linewidth=5)
ax.hist(d_KNN_sim,bins ='auto',histtype ='step',color = 'r', label = 'simulated',linewidth=5,alpha =0.5)
ax.set_xlabel('%s-NN distance'%(samples))
props = dict(boxstyle='round', facecolor='k', alpha=0.3)
# place a text box in upper left in axes coords
# ax.text(0.55, 0.58, 'The presence of $\it{clustered}$ data \nreduces the KNN distance\nin the real sample', transform=ax.transAxes, fontsize=27,
#         verticalalignment='top', bbox=props)
ax.axvline(min(d_KNN),ls = 'dashed', color ='k', linewidth = 5)
ax.axvline(min(d_KNN_sim),ls = 'dashed', color ='r', linewidth = 5,alpha =0.5)
ax.set_ylabel('N')
ax.set_xlim(0,0.05)
ax.legend()
# plt.savefig('/Users/amartinez/Desktop/PhD/Charlas/Presentaciones/Brno/' + 'arches_kernel.png', dpi=300,bbox_inches='tight')
#%%
fig, ax = plt.subplots(1,1,figsize=(20,10))

eps = np.mean([min(d_KNN),min(d_KNN_sim)])
ax.hist(d_KNN,bins ='auto',histtype ='step',color = 'k',label = 'real',linewidth=5)
ax.hist(d_KNN_sim,bins ='auto',histtype ='step',color = 'r', label = 'simulated',linewidth=5,alpha=0.5)
ax.set_xlabel('%s-NN distance'%(samples)) 
ax.arrow(1.25,125,eps-1.01,-95,head_width =0.2,head_length=25,shape ='full'
         ,color ='green',width =0.05,alpha=0.7)
# ax.axvline(eps,color='green')
props = dict(boxstyle='round', facecolor='g', alpha=0.5)
# place a text box in upper left in axes coords
ax.text(0.51, 0.58, '$\epsilon = ( min7NN_{real} + min7NN_{simultaed})/2$', transform=ax.transAxes, fontsize=27,
        verticalalignment='top', bbox=props)
ax.set_ylabel('N')
ax.set_ylim(0,)
ax.set_xlim(0,2)
ax.legend()
plt.savefig('/Users/amartinez/Desktop/PhD/Charlas/Presentaciones/Brno/' + 'arches_kernel1.png', dpi=300,bbox_inches='tight')









