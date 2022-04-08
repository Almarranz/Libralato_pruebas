#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 17:24:02 2022

@author: amartinez
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import FormatStrFormatter
import pandas as pd
from astropy.coordinates import match_coordinates_sky, SkyOffsetFrame, ICRS,offset_by
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.table import QTable
from matplotlib import rcParams
import os
import glob
import sys
from astropy.table import Table
from scipy.stats import gaussian_kde
# %%
from sklearn.preprocessing import StandardScaler
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
#Arches reference point 
center_arc = SkyCoord('17h45m50.4769267s', '-28d49m19.16770s', frame='icrs')
arches=Table.read(catal + 'Arches_cat_H22_Pclust.fits')
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
m127_all, m153_all = arches['F127M']*u.mag,arches['F153M']*u.mag
valid_colors=np.where((np.isnan(m127_all)==False)&(np.isnan(m153_all)==False))
m127,m153=m127_all[valid_colors],m153_all[valid_colors]

# =============================================================================
# np.savetxt(pruebas + 'arches_for_topcat.txt',np.array([ra.value,dec.value,pmra.value,pmdec.value,m127.value,m153.value]).T,header='ra.value,dec.value,pmra.value,pmdec.value,m127.value,m153.value')
# =============================================================================

# %%
def plotting(namex,namey,x,y,ind,**kwargs):

    pl=ax[ind].scatter(x,y,**kwargs)
    try:
        ax[ind].set_xlabel('%s(%s)'%(namex,x.unit)) # Set the axis label in the form "Variable description [units]"
        ax[ind].set_ylabel('%s(%s)'%(namey, y.unit))
    except:
        ax[ind].set_xlabel('K') # Set the axis label in the form "Variable description [units]"
        ax[ind].set_ylabel('K')
    if ind ==2:
        ax[ind].invert_yaxis()
    return pl
# %%
def plotting_h(namex,namey,x,y,ind,**kwargs):

    pl=ax[ind].hexbin(x.value,y.value,**kwargs)
    try:
        ax[ind].set_xlabel('%s(%s)'%(namex,x.unit)) # Set the axis label in the form "Variable description [units]"
        ax[ind].set_ylabel('%s(%s)'%(namey, y.unit))
    except:
        ax[ind].set_xlabel('K') # Set the axis label in the form "Variable description [units]"
        ax[ind].set_ylabel('K')
    if ind ==2:
        ax[ind].invert_yaxis()
    if ind ==1:
        ax[ind].invert_xaxis()
    return pl
# %%
fig, ax = plt.subplots(1,3,figsize=(30,10))
plotting('ra','dec',ra,dec,0,alpha=0.5)
plotting('mul','mub',pmra,pmdec,1,alpha=0.01)
plotting('m127-m157','m157',m127-m153,m153,2,alpha=0.05)
# %%
fig, ax = plt.subplots(1,3,figsize=(30,10))
plotting_h('ra','dec',ra,dec,0,bins=50,norm=matplotlib.colors.LogNorm())
plotting_h('mul','mub',pmra,pmdec,1,bins=50,norm=matplotlib.colors.LogNorm())
plotting('m127-m157','m157',m127-m153,m153,2,alpha=0.2)
# %%
arc_gal=arc_coor.galactic
pm_gal = SkyCoord(ra  = ra ,dec = dec, pm_ra_cosdec = pmra, pm_dec = pmdec,frame = 'icrs').galactic

# %%
l,b=arc_gal.l, arc_gal.b
pml,pmb=pm_gal.pm_l_cosb, pm_gal.pm_b
#%%
fig, ax = plt.subplots(1,3,figsize=(30,10))
plotting_h('ra','dec',l,b,0,bins=50,norm=matplotlib.colors.LogNorm())
plotting_h('mul','mub',pml,pmb,1,bins=50,norm=matplotlib.colors.LogNorm())
# %%

def density_plot(a,b,namex, namey, ind, **kwargs):
    xy = np.vstack([a,b])
    z = gaussian_kde(xy)(xy)
    pl =ax[ind].scatter(a, b, c=z,**kwargs)
    try:
        ax[ind].set_xlabel('%s(%s)'%(namex,a.unit)) # Set the axis label in the form "Variable description [units]"
        ax[ind].set_ylabel('%s(%s)'%(namey, b.unit))
    except:
        ax[ind].set_xlabel('K') # Set the axis label in the form "Variable description [units]"
        ax[ind].set_ylabel('K')
    if ind ==2:
        ax[ind].invert_yaxis()
    if ind ==1:
        ax[ind].invert_xaxis()
    return pl
# %%
plt_dim=3
fig, ax = plt.subplots(1,plt_dim,figsize=(plt_dim*10,10))
density_plot(l,b,'l','b',0,cmap='inferno')
density_plot(pml,pmb,'mul','mub',1,cmap='viridis')# add this for log scale in the color map:,norm=matplotlib.colors.LogNorm()
density_plot( m127-m153,m153,'m127-m153','m157',2,cmap='viridis')






















