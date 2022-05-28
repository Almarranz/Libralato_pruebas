#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 27 10:23:40 2022

@author: amartinez
"""

# =============================================================================
# IN this script we are going to plot some nice cluster for Brno talk
# =============================================================================

import hdbscan
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
import spisea
from spisea import synthetic, evolution, atmospheres, reddening, ifmr
from spisea.imf import imf, multiplicity
import astropy.coordinates as ap_coor
from scipy.stats import gaussian_kde
import shutil
from datetime import datetime


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
# %%
cata='/Users/amartinez/Desktop/PhD/Libralato_data/CATALOGS/'
pruebas='/Users/amartinez/Desktop/PhD/Libralato_data/pruebas/'
results='/Users/amartinez/Desktop/PhD/Libralato_data/results/'
gns_ext = '/Users/amartinez/Desktop/PhD/Libralato_data/extinction_maps/'
morralla = '/Users/amartinez/Desktop/morralla/'
# 'ra dec x_c  y_c mua dmua mud dmud time n1 n2 ID mul mub dmul dmub '
name='WFC3IR'
catal_all = np.loadtxt(cata + '%s_pm_galactic.txt'%(name))
# %%

# name='ACSWFC'
trimmed_data='yes'
if trimmed_data=='yes':
    pre=''
elif trimmed_data=='no':
    pre='relaxed_'
    
else:
    sys.exit("Have to set trimmed_data to either 'yes' or 'no'")

# %%

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
# %% Cluster extratction
subsec = '/Users/amartinez/Desktop/PhD/Libralato_data/pruebas/subsec_%s/'%('A')
name_cluster ='clusters_common_with_cl0_area19.1_0_0_samp7.txt'
# files = open(morralla +'clusters_common_with_cl0_area1.2_2_0_samp5.txt', 'r')
files =  np.genfromtxt(morralla+'dire_2022-05-26/' +name_cluster,dtype='str')

for f in files:
    cluster= np.concatenate([np.loadtxt(morralla+'dire_2022-05-26/' + f)])
# Lines = files.readlines()
# print(Lines)
section = 'A'#sections from A to D. Maybe make a script for each section...
subsec = '/Users/amartinez/Desktop/PhD/Libralato_data/pruebas/subsec_%s/'%(section)
carp_clus = subsec +'/clusters_in_%s/'%(section)
isE = os.path.exists(carp_clus)
if not isE:
    os.makedirs(subsec +'/clusters_in_%s'%(section))

for erase_f in glob.glob(carp_clus+'Sec_A_*'):
    os.remove(erase_f)
    
for erase_txt in glob.glob(carp_clus+'clusters_common_with_cl*'):
    os.remove(erase_txt)
    

clustered_by = 'all_color'# you can choose whether clustering by position, velocity and color, or only velocity and position
# clustered_by = 'all'
col =np.arange(0,1,1)
row =np.arange(0,1,1)
# areas = np.arange(1.3,24.0,0.1)
areas = np.arange(19.1,19.2,0.1)
samples_lst=[10]# number of minimun objects that defined a cluster

# % 
save_clus ='nada'
for samples in samples_lst:  
    samples_dist = samples# t
    print(samples)
    for area in areas:
        area = round(area,1)
        for colum in range(len(col)):
            for ro in range(len(row)):
                if save_clus == 'stop':
                    sys.exit('You stoped it')
    
                try:
                    catal = np.loadtxt(subsec + 'subsec_%s_%s_%s_%smin.txt'%(section,col[colum],row[ro],area))
                    # print(colum,ro,area)
                except:
                    # print('No area section %s_%s_%s area %s '%(section,col[colum],row[ro],area))
                    continue
                colori ='fuchsia'
                
                fig, ax = plt.subplots(1,1,figsize=(30,10))
                ax.set_ylim(0,10)
                ax.text(0.0, 5, 'Area %s'%(area),fontsize= 400,color='r')
                
                valid=np.where((np.isnan(catal[:,3])==False) & (np.isnan(catal[:,4])==False ))
                catal=catal[valid]
                center=np.where(catal[:,3]-catal[:,4]>1.3)
                catal=catal[center]
                dmu_lim = 5
                vel_lim = np.where((catal[:,19]<=dmu_lim) & (catal[:,20]<=dmu_lim))
                catal=catal[vel_lim]
                
                ra_=catal[:,5]
                dec_=catal[:,6]
                # Process needed for the trasnformation to galactic coordinates
                coordenadas = SkyCoord(ra=ra_*u.degree, dec=dec_*u.degree)#
                gal_c=coordenadas.galactic
    
                t_gal= QTable([gal_c.l,gal_c.b], names=('l','b'))
                
                datos = catal
                mul,mub = datos[:,-6],datos[:,-5]
                x,y = datos[:,7], datos[:,8]
                gal_coor = SkyCoord(ra = datos[:,0]*u.deg, dec =datos[:,1]*u.deg ).galactic
                colorines = datos[:,3]-datos[:,4]
                fig, ax = plt.subplots(1,3, figsize =(30,10))
                ax[0].scatter(mul,mub, color = 'k',alpha = 0.3)
                ax[1].scatter(gal_coor.l.value,gal_coor.b.value, color = 'k',alpha = 0.3)
                # ax[1].scatter(x,y,color = 'k',alpha = 0.3 )
                ax[0].scatter(cluster[:,4],cluster[:,5],color = colori,s=100)
                # ax[1].scatter(cluster[:,8],cluster[:,9],color = colori,s=100)
                ax[1].scatter(cluster[:,2],cluster[:,3],color = colori,s=100)
                ax[0].set_xlim(-10,10)
                ax[0].invert_xaxis()
                ax[0].set_xlabel(r'$\mathrm{\mu_{l} (mas\ yr^{-1})}$') 
                ax[0].set_ylabel(r'$\mathrm{\mu_{b} (mas\ yr^{-1})}$') 
                ax[1].set_xlabel('l (deg)') 
                ax[1].set_ylabel('b (deg)') 
                
                ax[2].scatter(colorines, datos[:,4],color = 'k',alpha = 0.05)
                ax[2].scatter(cluster[:,6]-cluster[:,7],cluster[:,7],color=colori, s=100)
                ax[2].invert_yaxis()
                ax[2].set_xlim(1.1,2.5)
                
                mul_sig, mub_sig = np.std(cluster[:,4]), np.std(cluster[:,5])
                mul_mean, mub_mean = np.mean(cluster[:,4]), np.mean(cluster[:,5])
                
                mul_sig_all, mub_sig_all = np.std(mul), np.std(mub)
                mul_mean_all, mub_mean_all = np.mean(mul), np.mean(mub)
            
            
                vel_txt = '\n'.join(('mul = %s, mub = %s'%(round(mul_mean,3), round(mub_mean,3)),
                                     '$\sigma_{mul}$ = %s, $\sigma_{mub}$ = %s'%(round(mul_sig,3), round(mub_sig,3)))) 
                vel_txt_all = '\n'.join(('mul = %s, mub = %s'%(round(mul_mean_all,3), round(mub_mean_all,3)),
                                     '$\sigma_{mul}$ = %s, $\sigma_{mub}$ = %s'%(round(mul_sig_all,3), round(mub_sig_all,3))))
                
                propiedades = dict(boxstyle='round', facecolor=colori, alpha=0.2)
                propiedades_all = dict(boxstyle='round', facecolor='k', alpha=0.1)
                ax[0].text(0.05, 0.95, vel_txt, transform=ax[0].transAxes, fontsize=18,
                    verticalalignment='top', bbox=propiedades)
                ax[0].text(0.05, 0.83, vel_txt_all, transform=ax[0].transAxes, fontsize=18,
                    verticalalignment='top', bbox=propiedades_all)
                
                c2 = SkyCoord(ra = cluster[:,0]*u.deg,dec = cluster[:,1]*u.deg)
                sep = [max(c2[c_mem].separation(c2)) for c_mem in range(len(c2))]
                rad = max(sep)/2
                prop = dict(boxstyle='round', facecolor=colori, alpha=0.2)
                ax[1].text(0.53, 0.95, 'aprox cluster radio = %s"\n cluster stars = %s '%(round(rad.to(u.arcsec).value,2),len(cluster)), 
                           transform=ax[1].transAxes, fontsize=18,
                                        verticalalignment='top', bbox=prop)
                
                ax[2].set_xlabel('H$-$Ks')
                ax[2].set_ylabel('Ks')
                plt.savefig('/Users/amartinez/Desktop/PhD/Charlas/Presentaciones/Brno/' + 'dbscan_%s.png'%(name_cluster), dpi=300,bbox_inches='tight')
                # %
                # This is for plotting the cluster and the isochrone with extiontion
                # here look for the values of extintiction for the cluster stars in the extintion catalog
                # if does not have a value it will not plot the star in the CMD
                H_Ks_yes = []
                Ks_yes = []
                AKs_clus_all =[]
                for star in range(len(cluster)):
                    clus_coord =  SkyCoord(ra=cluster[star,0]*u.degree, dec=cluster[star,1]*u.degree)
                    idx = clus_coord.match_to_catalog_sky(gns_coord)
                    gns_match = AKs_center[idx[0]]
                    # print(type(gns_match[16])) 
                    if gns_match[16] != '-' and gns_match[18] != '-':
                        AKs_clus_all.append(float(gns_match[18]))
                        H_Ks_yes.append(cluster[:,6]-cluster[:,7])
                        Ks_yes.append(cluster[:,7])
                 # fig, ax = plt.subplots(1,1,figsize =(10,10))
                
                AKs_clus, std_AKs = np.mean(AKs_clus_all),np.std(AKs_clus_all)
                absolute_difference_function = lambda list_value : abs(list_value - AKs_clus)
                AKs = min(AKs_list, key=absolute_difference_function)
                
                iso_dir = '/Users/amartinez/Desktop/PhD/Libralato_data/nsd_isochrones/'
                
                dist = 8000 # distance in parsec
                metallicity = 0.3 # Metallicity in [M/H]
                # logAge_600 = np.log10(0.61*10**9.)
                logAge_10 = np.log10(0.010*10**9.)
                logAge_30 = np.log10(0.030*10**9.)
                logAge_40 = np.log10(0.040*10**9.)
                logAge_50 = np.log10(0.050*10**9.)
                logAge_55 = np.log10(0.055*10**9.)
                logAge_60 = np.log10(0.060*10**9.)
                logAge_90 = np.log10(0.090*10**9.)
                evo_model = evolution.MISTv1() 
                atm_func = atmospheres.get_merged_atmosphere
                red_law = reddening.RedLawNoguerasLara18()
                filt_list = ['hawki,J', 'hawki,H', 'hawki,Ks']
                
                iso_10 =  synthetic.IsochronePhot(logAge_10, AKs, dist, metallicity=metallicity,
                                                evo_model=evo_model, atm_func=atm_func,
                                                red_law=red_law, filters=filt_list,
                                                    iso_dir=iso_dir)
                
                iso_30 = synthetic.IsochronePhot(logAge_30, AKs, dist, metallicity=metallicity,
                                                evo_model=evo_model, atm_func=atm_func,
                                                red_law=red_law, filters=filt_list,
                                                    iso_dir=iso_dir)
                iso_40 = synthetic.IsochronePhot(logAge_40, AKs, dist, metallicity=metallicity,
                                                evo_model=evo_model, atm_func=atm_func,
                                                red_law=red_law, filters=filt_list,
                                                    iso_dir=iso_dir)
                iso_50 = synthetic.IsochronePhot(logAge_50, AKs, dist, metallicity=metallicity,
                                                evo_model=evo_model, atm_func=atm_func,
                                                red_law=red_law, filters=filt_list,
                                                    iso_dir=iso_dir)
                iso_55 = synthetic.IsochronePhot(logAge_55, AKs, dist, metallicity=metallicity,
                                                evo_model=evo_model, atm_func=atm_func,
                                                red_law=red_law, filters=filt_list,
                                                    iso_dir=iso_dir)
                iso_60 = synthetic.IsochronePhot(logAge_60, AKs, dist, metallicity=metallicity,
                                                evo_model=evo_model, atm_func=atm_func,
                                                red_law=red_law, filters=filt_list,
                                                    iso_dir=iso_dir)
                
                iso_90 = synthetic.IsochronePhot(logAge_90, AKs, dist, metallicity=metallicity,
                                                evo_model=evo_model, atm_func=atm_func,
                                                red_law=red_law, filters=filt_list,
                                                    iso_dir=iso_dir)
                # #%
                #%
                
                
                imf_multi = multiplicity.MultiplicityUnresolved()
                
                # Make IMF object; we'll use a broken power law with the parameters from Kroupa+01
                
                # NOTE: when defining the power law slope for each segment of the IMF, we define
                # the entire exponent, including the negative sign. For example, if dN/dm $\propto$ m^-alpha,
                # then you would use the value "-2.3" to specify an IMF with alpha = 2.3. 
                
                massLimits = np.array([0.2, 0.5, 1, 120]) # Define boundaries of each mass segement
                powers = np.array([-1.3, -2.3, -2.3]) # Power law slope associated with each mass segment
                # my_imf = imf.IMF_broken_powerlaw(massLimits, powers, imf_multi)
                my_imf = imf.IMF_broken_powerlaw(massLimits, powers,multiplicity = None)
                
                
                #%
                
                iso=iso_50
                logAge = logAge_50
                mass = 2*10**4.
                mass = 1 * mass
                dAks = round(std_AKs*1,3)
                cluster_spi = synthetic.ResolvedClusterDiffRedden(iso, my_imf, mass,dAks)
                cluster_ndiff = synthetic.ResolvedCluster(iso, my_imf, mass)
                clus = cluster_spi.star_systems
                clus_ndiff = cluster_ndiff.star_systems
                # ax.set_title(name_cluster)
                # ax[2].scatter(clus['m_hawki_H']-clus['m_hawki_Ks'],clus['m_hawki_Ks'],color = 'lavender',alpha=1,zorder=2)

                # ax.scatter(clus_ndiff['m_hawki_H']-clus_ndiff['m_hawki_Ks'],clus_ndiff['m_hawki_Ks'],color = 'k',alpha=0.1,s=1)
                
    
                txt_srn = '\n'.join(('metallicity = %s'%(metallicity),'dist = %.1f Kpc'%(dist/1000),'mass =%.0fx$10^{3}$ $M_{\odot}$'%(mass/1000)))#,'age = %.0f Myr'%(10**logAge/10**6)))
                txt_AKs = '\n'.join(('AKs = %.2f'%(AKs_clus),'std_AKs = %.2f'%(std_AKs)))
                props = dict(boxstyle='round', facecolor='lavender', alpha=0.7)
                # place a text box in upper left in axes coords
                ax[2].text(0.65, 0.65, txt_AKs, transform=ax[2].transAxes, fontsize=18,
                    verticalalignment='top', bbox=props)
                ax[2].text(0.65, 0.55, txt_srn, transform=ax[2].transAxes, fontsize=18,
                    verticalalignment='top', bbox=props)
                ax[2].plot(iso.points['m_hawki_H'] - iso.points['m_hawki_Ks'], 
                                  iso.points['m_hawki_Ks'], 'k',  label='age = %.0f Myr'%(10**logAge/10**6),alpha=0.5)
                ax[2].scatter(H_Ks_yes,Ks_yes, color= colori,s=5,zorder=3, alpha=1)

                ax[2].set_xlabel('H$-$Ks')
                ax[2].set_xlim(1,max(colorines)/2)
                ax[2].set_ylabel('Ks')
                ax[2].set_ylim(min(datos[:,4]),max(datos[:,4]))
                ax[2].invert_yaxis()
                ax[2].legend()
                # plt.savefig('/Users/amartinez/Desktop/PhD/Charlas/Presentaciones/Brno/' + '%s.png'%(name_cluster), dpi=300,bbox_inches='tight')

# %%This is only for the spisea cluster
H_Ks_yes = []
Ks_yes = []
AKs_clus_all =[]
for star in range(len(cluster)):
    clus_coord =  SkyCoord(ra=cluster[star,0]*u.degree, dec=cluster[star,1]*u.degree)
    idx = clus_coord.match_to_catalog_sky(gns_coord)
    gns_match = AKs_center[idx[0]]
    # print(type(gns_match[16])) 
    if gns_match[16] != '-' and gns_match[18] != '-':
        AKs_clus_all.append(float(gns_match[18]))
        H_Ks_yes.append(cluster[:,6]-cluster[:,7])
        Ks_yes.append(cluster[:,7])
fig, ax = plt.subplots(1,1,figsize =(10,10))
# ax.scatter(H_Ks_yes,Ks_yes, color= colori,s=5,zorder=3, alpha=1)
AKs_clus, std_AKs = np.mean(AKs_clus_all),np.std(AKs_clus_all)
absolute_difference_function = lambda list_value : abs(list_value - AKs_clus)
AKs = min(AKs_list, key=absolute_difference_function)

iso_dir = '/Users/amartinez/Desktop/PhD/Libralato_data/nsd_isochrones/'

dist = 8000 # distance in parsec
metallicity = 0.3 # Metallicity in [M/H]
# logAge_600 = np.log10(0.61*10**9.)
logAge_10 = np.log10(0.010*10**9.)
logAge_30 = np.log10(0.030*10**9.)
logAge_40 = np.log10(0.040*10**9.)
logAge_50 = np.log10(0.050*10**9.)
logAge_55 = np.log10(0.055*10**9.)
logAge_60 = np.log10(0.060*10**9.)
logAge_90 = np.log10(0.090*10**9.)
evo_model = evolution.MISTv1() 
atm_func = atmospheres.get_merged_atmosphere
red_law = reddening.RedLawNoguerasLara18()
filt_list = ['hawki,J', 'hawki,H', 'hawki,Ks']

iso_10 =  synthetic.IsochronePhot(logAge_10, AKs, dist, metallicity=metallicity,
                                evo_model=evo_model, atm_func=atm_func,
                                red_law=red_law, filters=filt_list,
                                    iso_dir=iso_dir)

iso_30 = synthetic.IsochronePhot(logAge_30, AKs, dist, metallicity=metallicity,
                                evo_model=evo_model, atm_func=atm_func,
                                red_law=red_law, filters=filt_list,
                                    iso_dir=iso_dir)
iso_40 = synthetic.IsochronePhot(logAge_40, AKs, dist, metallicity=metallicity,
                                evo_model=evo_model, atm_func=atm_func,
                                red_law=red_law, filters=filt_list,
                                    iso_dir=iso_dir)
iso_50 = synthetic.IsochronePhot(logAge_50, AKs, dist, metallicity=metallicity,
                                evo_model=evo_model, atm_func=atm_func,
                                red_law=red_law, filters=filt_list,
                                    iso_dir=iso_dir)
iso_55 = synthetic.IsochronePhot(logAge_55, AKs, dist, metallicity=metallicity,
                                evo_model=evo_model, atm_func=atm_func,
                                red_law=red_law, filters=filt_list,
                                    iso_dir=iso_dir)
iso_60 = synthetic.IsochronePhot(logAge_60, AKs, dist, metallicity=metallicity,
                                evo_model=evo_model, atm_func=atm_func,
                                red_law=red_law, filters=filt_list,
                                    iso_dir=iso_dir)

iso_90 = synthetic.IsochronePhot(logAge_90, AKs, dist, metallicity=metallicity,
                                evo_model=evo_model, atm_func=atm_func,
                                red_law=red_law, filters=filt_list,
                                    iso_dir=iso_dir)
# #%
#%


imf_multi = multiplicity.MultiplicityUnresolved()

# Make IMF object; we'll use a broken power law with the parameters from Kroupa+01

# NOTE: when defining the power law slope for each segment of the IMF, we define
# the entire exponent, including the negative sign. For example, if dN/dm $\propto$ m^-alpha,
# then you would use the value "-2.3" to specify an IMF with alpha = 2.3. 

massLimits = np.array([0.2, 0.5, 1, 120]) # Define boundaries of each mass segement
powers = np.array([-1.3, -2.3, -2.3]) # Power law slope associated with each mass segment
# my_imf = imf.IMF_broken_powerlaw(massLimits, powers, imf_multi)
my_imf = imf.IMF_broken_powerlaw(massLimits, powers,multiplicity = None)


#%

iso=iso_10
logAge = logAge_10
mass = 5*10**4.
mass = 1 * mass
dAks = round(std_AKs*1,3)
cluster_spi = synthetic.ResolvedClusterDiffRedden(iso, my_imf, mass,dAks)
cluster_ndiff = synthetic.ResolvedCluster(iso, my_imf, mass)
clus = cluster_spi.star_systems
clus_ndiff = cluster_ndiff.star_systems
# ax.set_title(name_cluster)
ax.scatter(clus['m_hawki_H']-clus['m_hawki_Ks'],clus['m_hawki_Ks'],color = 'red',alpha=1)
ax.scatter(clus_ndiff['m_hawki_H']-clus_ndiff['m_hawki_Ks'],clus_ndiff['m_hawki_Ks'],color = 'k',alpha=1,s=10)


txt_srn = '\n'.join(('metallicity = %s'%(metallicity),'dist = %.1f Kpc'%(dist/1000),'mass =%.0fx$10^{3}$ $M_{\odot}$'%(mass/1000),'age = %.0f Myr'%(10**logAge/10**6)))
txt_AKs = '\n'.join(('AKs = %.2f'%(AKs_clus),'std_AKs = %.2f'%(std_AKs)))
props = dict(boxstyle='round', facecolor='red', alpha=0.3)
# place a text box in upper left in axes coords
ax.text(0.65, 0.95, txt_AKs, transform=ax.transAxes, fontsize=18,
    verticalalignment='top', bbox=props)
ax.text(0.65, 0.85, txt_srn, transform=ax.transAxes, fontsize=18,
    verticalalignment='top', bbox=props)
# ax.plot(iso.points['m_hawki_H'] - iso.points['m_hawki_Ks'], 
#                   iso.points['m_hawki_Ks'], 'k',  label='age = %.0f Myr'%(10**logAge/10**6),alpha=0.5)

ax.invert_yaxis()
ax.set_xlabel('H$-$Ks')
ax.set_xlim(0,)
ax.set_ylabel('Ks')
ax.legend()
# plt.savefig('/Users/amartinez/Desktop/PhD/Charlas/Presentaciones/Brno/' + 'spisea.png', dpi=300,bbox_inches='tight')
# %%
# Whit isochrnes
fig, ax = plt.subplots(1,1,figsize=(30,10))
ax.set_ylim(0,10)
ax.text(0.0, 5, 'Area %s'%(area),fontsize= 400,color='r')

valid=np.where((np.isnan(catal[:,3])==False) & (np.isnan(catal[:,4])==False ))
catal=catal[valid]
center=np.where(catal[:,3]-catal[:,4]>1.3)
catal=catal[center]
dmu_lim = 5
vel_lim = np.where((catal[:,19]<=dmu_lim) & (catal[:,20]<=dmu_lim))
catal=catal[vel_lim]

ra_=catal[:,5]
dec_=catal[:,6]
# Process needed for the trasnformation to galactic coordinates
coordenadas = SkyCoord(ra=ra_*u.degree, dec=dec_*u.degree)#
gal_c=coordenadas.galactic

t_gal= QTable([gal_c.l,gal_c.b], names=('l','b'))

datos = catal
mul,mub = datos[:,-6],datos[:,-5]
x,y = datos[:,7], datos[:,8]
gal_coor = SkyCoord(ra = datos[:,0]*u.deg, dec =datos[:,1]*u.deg ).galactic
colorines = datos[:,3]-datos[:,4]
fig, ax = plt.subplots(1,3, figsize =(30,10))
ax[0].scatter(mul,mub, color = 'k',alpha = 0.3)
ax[1].scatter(gal_coor.l.value,gal_coor.b.value, color = 'k',alpha = 0.3)
# ax[1].scatter(x,y,color = 'k',alpha = 0.3 )
ax[0].scatter(cluster[:,4],cluster[:,5],color = colori,s=100)
# ax[1].scatter(cluster[:,8],cluster[:,9],color = colori,s=100)
ax[1].scatter(cluster[:,2],cluster[:,3],color = colori,s=100)
ax[0].set_xlim(-10,10)
ax[0].invert_xaxis()
ax[0].set_xlabel(r'$\mathrm{\mu_{l} (mas\ yr^{-1})}$') 
ax[0].set_ylabel(r'$\mathrm{\mu_{b} (mas\ yr^{-1})}$') 
ax[1].set_xlabel('l (deg)') 
ax[1].set_ylabel('b (deg)') 

ax[2].scatter(colorines, datos[:,4],color = 'k',alpha = 0.05)
ax[2].scatter(cluster[:,6]-cluster[:,7],cluster[:,7],color=colori, s=100)
ax[2].invert_yaxis()
ax[2].plot(iso_50.points['m_hawki_H'] - iso_50.points['m_hawki_Ks'], 
                  iso_50.points['m_hawki_Ks'], 'orange',  label='30 Myr')
ax[2].set_xlim(1.1,2.5)

mul_sig, mub_sig = np.std(cluster[:,4]), np.std(cluster[:,5])
mul_mean, mub_mean = np.mean(cluster[:,4]), np.mean(cluster[:,5])

mul_sig_all, mub_sig_all = np.std(mul), np.std(mub)
mul_mean_all, mub_mean_all = np.mean(mul), np.mean(mub)


vel_txt = '\n'.join(('mul = %s, mub = %s'%(round(mul_mean,3), round(mub_mean,3)),
                     '$\sigma_{mul}$ = %s, $\sigma_{mub}$ = %s'%(round(mul_sig,3), round(mub_sig,3)))) 
vel_txt_all = '\n'.join(('mul = %s, mub = %s'%(round(mul_mean_all,3), round(mub_mean_all,3)),
                     '$\sigma_{mul}$ = %s, $\sigma_{mub}$ = %s'%(round(mul_sig_all,3), round(mub_sig_all,3))))

propiedades = dict(boxstyle='round', facecolor=colori, alpha=0.2)
propiedades_all = dict(boxstyle='round', facecolor='k', alpha=0.1)
ax[0].text(0.05, 0.95, vel_txt, transform=ax[0].transAxes, fontsize=18,
    verticalalignment='top', bbox=propiedades)
ax[0].text(0.05, 0.83, vel_txt_all, transform=ax[0].transAxes, fontsize=18,
    verticalalignment='top', bbox=propiedades_all)

c2 = SkyCoord(ra = cluster[:,0]*u.deg,dec = cluster[:,1]*u.deg)
sep = [max(c2[c_mem].separation(c2)) for c_mem in range(len(c2))]
rad = max(sep)/2
prop = dict(boxstyle='round', facecolor=colori, alpha=0.2)
ax[1].text(0.53, 0.95, 'aprox cluster radio = %s"\n cluster stars = %s '%(round(rad.to(u.arcsec).value,2),len(cluster)), 
           transform=ax[1].transAxes, fontsize=18,
                        verticalalignment='top', bbox=prop)
ax[2].set_xlim(1.2,3)
ax[2].set_ylim(min(datos[:,4]),max(datos[:,4]))
ax[2].set_xlabel('H$-$Ks')
ax[2].set_ylabel('Ks')
ax[2].invert_yaxis()
# plt.savefig('/Users/amartinez/Desktop/PhD/Charlas/Presentaciones/Brno/' + 'ISO_dbscan_%s.png'%(name_cluster), dpi=300,bbox_inches='tight')




