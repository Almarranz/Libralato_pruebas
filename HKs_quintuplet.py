#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 16:17:07 2022

@author: amartinez
"""

import numpy as np
from astropy.coordinates import match_coordinates_sky
import pandas as pd
import astropy.units as u
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt
#in this script we are going extract the quituplet cluster from GNS central region by cross matching it with a list extrated from Simbad
# Then we check the average H-Ks, to have an idea of this value in a cluster
cata='/Users/amartinez/Desktop/PhD/Libralato_data/CATALOGS/'
pruebas='/Users/amartinez/Desktop/PhD/Libralato_data/pruebas/'
# results='/Users/amartinez/Desktop/PhD/Libralato_data/results/'
sim_qui=pd.read_csv(pruebas +'quintuplet_simbad.csv')
# ['_RAJ2000'0, '_DEJ2000'1, 'RAJ2000'2, 'e_RAJ2000'3, 'DEJ2000'4, 'e_DEJ2000'5, 'RAJdeg'6, 
#  'e_RAJdeg'7, 'DEJdeg'8, 'e_DEJdeg'9, 'RAHdeg'10, 'e_RAHdeg'11, 'DEHdeg'12, 
#  'e_DEHdeg'13, 'RAKsdeg'14, 'e_RAKsdeg'15, 'DEKsdeg'16, 'e_DEKsdeg'17, 
#  'Jmag'18, 'e_Jmag'19, 'Hmag'20, 'e_Hmag'21, 'Ksmag'22, 'e_Ksmag'23, 'iJ'24, 'iH'25, 'iKs'26]
# 27 columns
gns= pd.read_csv(cata + 'GNS_central.csv')# tCentral region of GNS
# %%
ra_q=sim_qui['RA_d'].to_numpy()
dec_q=sim_qui['DEC_d'].to_numpy()
ra_g=gns['_RAJ2000'].to_numpy()
dec_g=gns['_DEJ2000'].to_numpy()


gns_coord = SkyCoord(ra=ra_g*u.degree, dec=dec_g*u.degree)
quit_coord =  SkyCoord(ra=ra_q*u.degree, dec=dec_q*u.degree)


idx = gns_coord.match_to_catalog_sky(quit_coord)

# %
valid = np.where(idx[1]<1*u.arcsec)
gns_np=gns.to_numpy()
gns_quit=gns_np[valid]
#%
center=np.where(gns_quit[:,20] - gns_quit[:,22] > 1.3)
gns_quit_c=gns_quit[center]
np.savetxt(pruebas + 'test_quin.txt',gns_quit_c )
#%

fig,ax = plt.subplots(1,1,figsize=(10,10))
ax.invert_yaxis()
ax.scatter((gns_quit_c[:,20] - gns_quit_c[:,22]),gns_quit_c[:,22],c='k',s=1)

#%
fig,ax = plt.subplots(1,1,figsize=(10,10))

ax.hist((gns_quit_c[:,20] - gns_quit_c[:,22]),bins='auto')
# %
print(np.mean(gns_quit_c[:,20] - gns_quit_c[:,22]))
print(np.std(gns_quit_c[:,20] - gns_quit_c[:,22]))

# %%
r_u=5
test=[]
gns_quit_c_c=SkyCoord(ra=gns_quit_c[:,0]*u.degree,dec=gns_quit_c[:,1]*u.degree)
for t in range(1):
    rand=np.random.choice(np.arange(0,len(gns_quit_c)),1)
    gns_rand=gns_quit_c[rand]
    gns_rand_c=SkyCoord(ra = gns_rand[:,0]*u.degree, dec=gns_rand[:,1]*u.degree)
    idxc, indices, d2d,d3d = gns_quit_c_c.search_around_sky(gns_rand_c, r_u*u.arcsec)
    test.append(max(gns_quit_c[indices][:,20]-gns_quit_c[indices][:,22])-min(gns_quit_c[indices][:,20]-gns_quit_c[indices][:,22]))
    fig, ax = plt.subplots(1,1,figsize=(10,10))

    ax.scatter(gns_quit_c[:,0],gns_quit_c[:,1])
    ax.scatter(gns_quit_c[indices][:,0],gns_quit_c[indices][:,1],color='red')
    ax.scatter(gns_rand_c.ra, gns_rand_c.dec,c ='blue')
print(np.mean(test),np.median(test),np.std(test))

# %%
qt= pd.read_csv(pruebas + 'quintuplet_test.csv')# this is the centrAL region of the clustet cropped by hand

qt_np=qt.to_numpy()

cen=np.where(qt_np[:,20] - qt_np[:,22] > 1.3)
qt_np=qt_np[cen]

fig,ax = plt.subplots(1,1,figsize=(10,10))

ax.hist((qt_np[:,20] - qt_np[:,22]),bins='auto')
# %%

r_u=2
test=[]
qt_np_c=SkyCoord(ra=qt_np[:,0]*u.degree,dec=qt_np[:,1]*u.degree)
for t in range(1):
    rand=np.random.choice(np.arange(0,len(qt_np_c)),1)
    gns_rand=qt_np[rand]
    gns_rand_c=SkyCoord(ra = gns_rand[:,0]*u.degree, dec=gns_rand[:,1]*u.degree)
    idxc1, indices1, d2d1,d3d1 = gns_quit_c_c.search_around_sky(gns_rand_c, r_u*u.arcsec)
    test.append(max(gns_quit_c[indices1][:,20]-gns_quit_c[indices1][:,22])-min(gns_quit_c[indices1][:,20]-gns_quit_c[indices1][:,22]))
    fig, ax = plt.subplots(1,1,figsize=(10,10))

    ax.scatter(gns_quit_c[:,0],gns_quit_c[:,1])
    ax.scatter(gns_quit_c[indices1][:,0],gns_quit_c[indices1][:,1],color='red')
    ax.scatter(gns_rand_c.ra, gns_rand_c.dec,c ='blue')
print(max(gns_quit_c[indices1][:,20]-gns_quit_c[indices1][:,22]),min(gns_quit_c[indices1][:,20]-gns_quit_c[indices1][:,22]))
print(np.mean(test),np.median(test),np.std(test))




# =============================================================================
# test1=[]
# for t in range(10000):
#     rand=np.random.choice(np.arange(0,len(qt_np)),10)
#     gns_rand=qt_np[rand]
#     test1.append(max(gns_rand[:,20]-gns_rand[:,22])-min(gns_rand[:,20]-gns_rand[:,22]))
# print(np.mean(test1),np.median(test1),np.std(test1))
# =============================================================================
#%% group=np.where(np.sqrt((catal[:,5]-catal[index[0],5])**2 + (catal[:,6]-catal[index[0],6])**2)< radio)


r_u=0.001
test=[]
qt_np_c=SkyCoord(ra=qt_np[:,0]*u.degree,dec=qt_np[:,1]*u.degree)

for t in range(1):
    rand=np.random.choice(np.arange(0,len(qt_np_c)),1)
    gns_rand=qt_np[rand]
    gns_rand=SkyCoord(ra = gns_rand[:,0]*u.degree, dec=gns_rand[:,1]*u.degree)
    group=np.where(np.sqrt((gns_rand.ra.value-gns_quit_c[:,0])**2 + (gns_rand.dec.value-gns_quit_c[:,1])**2)< r_u)
    test.append(max(gns_quit_c[group][:,20]-gns_quit_c[group][:,22])-min(gns_quit_c[group][:,20]-gns_quit_c[group][:,22]))
    fig, ax = plt.subplots(1,1,figsize=(10,10))

    ax.scatter(gns_quit_c[:,0],gns_quit_c[:,1])
    ax.scatter(gns_quit_c[group][:,0],gns_quit_c[group][:,1],color='red')
    ax.scatter(gns_rand.ra, gns_rand.dec,c ='blue')
    
    fig, ax = plt.subplots(1,1,figsize=(10,10))
    ax.scatter(gns_quit_c[:,20]-gns_quit_c[:,22],gns_quit_c[:,22])
    ax.scatter(gns_quit_c[group][:,20]-gns_quit_c[group][:,22],gns_quit_c[group][:,22], c='red', s=5)
    ax.invert_yaxis()
print(np.mean(test),np.median(test),np.std(test))



#%%
print(max(gns_quit_c[group][:,20]-gns_quit_c[group][:,22]))

print(min(gns_quit_c[group][:,20]-gns_quit_c[group][:,22]))


















