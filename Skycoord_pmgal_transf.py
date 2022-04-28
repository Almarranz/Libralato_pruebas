#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 13:12:04 2022

@author: amartinez
"""

#This is script is to chenck the transfotmation of galactic pm that I did using the matrix vs the trasformation you can do using SkyCoord

import numpy as np
import astropy.units as u
from astropy.table import Table, QTable
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt

name='WFC3IR'
cata='/Users/amartinez/Desktop/PhD/Libralato_data/CATALOGS/'
pruebas='/Users/amartinez/Desktop/PhD/Libralato_data/pruebas/'
results='/Users/amartinez/Desktop/PhD/Libralato_data/results/'
#%%
catal=np.loadtxt(cata+'GALCEN_%s_PM.cat'%(name))
cat_gal=np.loadtxt(cata+'GALCEN_%s_PM_galactic.txt'%(name))
# %%
ra=catal[:,0]*u.degree
dec=catal[:,1]*u.degree
pmra=catal[:,4]*u.mas/u.yr
pmdec=catal[:,6]*u.mas/u.yr
#%%
data1_coord = SkyCoord(ra  = ra,
                        dec = dec,
                        pm_ra_cosdec = pmra,
                        pm_dec = pmdec,
                        frame = 'fk5')
# %% You need ra and dec to make the transformation
# data1_coord = SkyCoord( pm_ra_cosdec = pmra, pm_dec = pmdec,frame = 'fk5')
#%%
data1_galactic = data1_coord.galactic
# %%
for i in range(1000,1010):
    print(round(pmra[i].value,3),round(pmdec[i].value,3))
    print(round(data1_coord.pm_ra_cosdec[i].value,3),round(data1_coord.pm_dec[i].value,3))
    print(round(data1_galactic.pm_l_cosb[i].value,3),round(data1_galactic.pm_b[i].value,3))
    print(round(cat_gal[i][0],3),round(cat_gal[i][1],3))
    print(20*'#')
    
# %%
print(data1_coord[0])