#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 17:24:02 2022

@author: amartinez
"""

import numpy as np
import matplotlib.pyplot as plt
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
# %%
catal='/Users/amartinez/Desktop/PhD/Arches_and_Quintuplet_Hosek/'
pruebas='/Users/amartinez/Desktop/PhD/Arches_and_Quintuplet_Hosek/pruebas/'

arches=Table.read(catal + 'Arches_cat_H22_Pclust.fits')
arc_coor=SkyCoord(ra=arches['ra*']*u.arcsec,dec=arches['dec']*u.arcsec)
#Arches reference point 
center_arc = SkyCoord('17h45m50.4769267s', '-28d49m19.16770s', frame='icrs')
new_coord=SkyCoord(arc_coor.ra + center_arc.ra,arc_coor.dec + center_arc.dec)
# %%
print(new_coord)
np.savetxt(pruebas + 'another_test.txt',np.array([new_coord.ra,new_coord.dec]).T)





















