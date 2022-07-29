#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 19:22:28 2022

@author: amartinez
"""
# =============================================================================
# Here we are going to compare a bunch of cluster files with each other and see
# if any of then are closer than a certain distance. The idea is to compare
# cluster from Libralato and Bans catalogs
# 
# ============================================================================


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import glob
import os
import sys
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import QTable
import astropy.coordinates as ap_coor
import pandas as pd
# import spisea
# from spisea import synthetic, evolution, atmospheres, reddening, ifmr
# from spisea.imf import imf, multiplicity
# from astropy.stats import sigma_clip
from astropy.coordinates import FK5
import time
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

# Path to the folder with the folders cointaining the clusters 
lib_files = '/Users/amartinez/Desktop/morralla/Sec_A_dmu1_at_2022-07-28_123/'
ban_files = '/Users/amartinez/Desktop/morralla/BAN_Sec_A_dmu20000_at_2022-07-29/'
tic = time.perf_counter()
for lib_folder in sorted(glob.glob(lib_files + 'cluster_num*'),key = os.path.getmtime):
    # print(30*'-')
    # print(os.path.basename(lib_folder))
    for file in glob.glob(lib_folder + '/cluster*'):
        lib_ra, lib_dec = np.loadtxt(file,unpack=True, usecols = (0,1))
        lib_coord = SkyCoord(ra = lib_ra, dec = lib_dec,unit = 'deg',frame = FK5,equinox ='J2014.2')
        for ban_folder in sorted(glob.glob(ban_files + 'ban_cluster_num*'),key = os.path.getmtime):
            # print(os.path.basename(ban_folder))
            for file1 in glob.glob(ban_folder + '/ban_cluster*'):
                ban_ra, ban_dec = np.loadtxt(file1, unpack = True, usecols = (0,1))
                ban_coord = SkyCoord(ra = ban_ra*u.deg, dec = ban_dec*u.deg,frame = FK5,equinox ='J2000')
                
                idx, d2d, dd3 = ban_coord.match_to_catalog_sky(lib_coord)
                min_dist = min(d2d.value)
                if min_dist < 1/3600:
                    print(50*'+')
                    print(os.path.basename(file),os.path.basename(file1))
                    print(50*'+')
                # sys.exit()
    print('Donde with folder: %s'%(os.path.basename(lib_folder)))            
toc = time.perf_counter()
print('Comparison took: %.1f sec'%(toc-tic))

                    