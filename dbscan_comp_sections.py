#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 16 12:26:28 2022

@author: amartinez
"""

# =============================================================================
# In this script we are going to look for the cluster on each section separately
# The sections (A to B) have been produced in sections_A_to_B.py
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

name='WFC3IR'
# name='ACSWFC'
trimmed_data='yes'
if trimmed_data=='yes':
    pre=''
elif trimmed_data=='no':
    pre='relaxed_'
    
else:
    sys.exit("Have to set trimmed_data to either 'yes' or 'no'")
    
# %%
section = 'A'#sections from A to D. Maybe make a script for each section...
# section = 'A_1_0'#selecting the whole thing
# "'RA_gns','DE_gns','Jmag','Hmag','Ksmag','ra','dec','x_c','y_c','mua','dmua','mud','dmud','time','n1','n2','ID','mul','mub','dmul','dmub','m139','Separation'",
if section == 'All':
    catal=np.loadtxt(results + '%smatch_GNS_and_%s_refined_galactic.txt'%(pre,name))
else:
    catal=np.loadtxt(results + 'sec_%s_%smatch_GNS_and_%s_refined_galactic.txt'%(section,pre,name))
    # catal=np.genfromtxt(pruebas +'subsec_%s/subsec_%s.txt'%('A',section))

# Definition of center can: m139 - Ks(libralato and GNS) or H - Ks(GNS and GNS)
center_definition='G_G'#this variable can be L_G or G_G
if center_definition =='L_G':
    valid=np.where(np.isnan(catal[:,4])==False)# This is for the valus than make Ks magnitude valid, but shouldn´t we do the same with the H magnitudes?
    catal=catal[valid]
    center=np.where(catal[:,-2]-catal[:,4]>2.5) # you can choose the way to make the color cut, as they did in libralato or as it is done in GNS
elif center_definition =='G_G':
    valid=np.where((np.isnan(catal[:,3])==False) & (np.isnan(catal[:,4])==False ))
    catal=catal[valid]
    center=np.where(catal[:,3]-catal[:,4]>1.3)
catal=catal[center]
dmu_lim = 1
vel_lim = np.where((catal[:,19]<=dmu_lim) & (catal[:,20]<=dmu_lim))
catal=catal[vel_lim]

# 'ra dec x_c  y_c mua dmua mud dmud time n1 n2 ID mul mub dmul dmub '
catal_all = np.loadtxt(cata + '%s_pm_galactic.txt'%(name))




# catal=catal_df.to_numpy()
# valid=np.where(np.isnan(catal[:,14])==False)
# mul_mc,mub_mc,dmul_mc,dmub_mc
# gal_coor=catal[:,[17,18,19,20]]


# %%
# %%
# ra, dec, other things
#Selecting the massive stars to plotting in the xy plot
Ms_ra, Ms_dec = np.loadtxt(cata + 'GALCEN_TABLE_D.cat',usecols=(0,1),unpack = True)

Ms_xy = [int(np.where((Ms_ra[i]==(catal_all[:,0])) & ((Ms_dec[i]==catal_all[:,1])))[0]) for i in range(len(Ms_ra)) if len(np.where((Ms_ra[i]==(catal_all[:,0])) & ((Ms_dec[i]==catal_all[:,1])))[0]) >0]

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
AKs_list = np.append(AKs_list1,0)
# %%

for file_to_erase in glob.glob(pruebas + 'Sec_%s_%s_%scluster*_eps*.txt'%(section,name,pre)):
    os.remove(file_to_erase)
# %

pixel = 'yes'
# cluster_by = 'all'
cluster_by = 'all_and_color'
# pms =[0,0,-5.72,-0.17]# galactic pm obtained from the dynesty adjustement 
pms =[0,0,0,0]
data = catal

ra_=data[:,5]
dec_=data[:,6]
# Process needed for the trasnformation to galactic coordinates
coordenadas = SkyCoord(ra=ra_*u.degree, dec=dec_*u.degree)#
gal_c=coordenadas.galactic

t_gal= QTable([gal_c.l,gal_c.b], names=('l','b'))  

if pixel == 'no':
    X=np.array([data[:,-6]-pms[2],data[:,-5]-pms[3],t_gal['l'].value,t_gal['b'].value]).T
elif pixel == 'yes':
    X=np.array([data[:,-6]-pms[2],data[:,-5]-pms[3],data[:,7],data[:,8]]).T
if cluster_by == 'pos':
    X_stad = StandardScaler().fit_transform(X[:,[2,3]])
elif cluster_by == 'pm':
    X_stad = StandardScaler().fit_transform(X[:,[0,1]])
elif cluster_by == 'all':
    X_stad = StandardScaler().fit_transform(X)
elif cluster_by == 'all_and_color':
    X=np.array([data[:,-6]-pms[2],data[:,-5]-pms[3],data[:,7],data[:,8],data[:,3]-data[:,4]]).T
    X_stad = StandardScaler().fit_transform(X)
    

#this aproach below isn´t working...
# =============================================================================
# elif cluster_by == 'vel':
#     X_vel = np.array([np.sqrt(data[:,-6]**2 + data[:,-5]**2),data[:,-6]/data[:,-5] ,np.sqrt(data[:,7]**2 + data[:,8]**2),data[:,7]/data[:,8]]).T
#     X_stad = StandardScaler().fit_transform(X_vel)
#     
# =============================================================================
tree=KDTree(X_stad, leaf_size=2) 

samples=5# number of minimun objects that defined a cluster
samples_dist = samples# t

dist, ind = tree.query(X_stad, k=samples_dist) #DistNnce to the 1,2,3...k neighbour
d_KNN=sorted(dist[:,-1])#distance to the Kth neighbour

kneedle = KneeLocator(np.arange(0,len(data),1), d_KNN, curve='convex', interp_method = "polynomial",direction="increasing")
elbow = KneeLocator(np.arange(0,len(data),1), d_KNN, curve='concave', interp_method = "polynomial",direction="increasing")
rodilla=round(kneedle.elbow_y, 3)
codo = round(elbow.elbow_y, 3)


epsilon = round(min(d_KNN),3)
# sys.exit('salida')
# epsilon=0.08
clus_method = 'dbs'

clustering = DBSCAN(eps=epsilon, min_samples=samples).fit(X_stad)
l=clustering.labels_

save_clus =''
loop=0
# while len(set(l))<15: # min number of cluster to find. It star looking at the min values of the Knn distance plot and increases epsilon until the cluster are found. BE careful cose ALL cluster will be found with the lastest (and biggest) value of eps, so it might lost some clusters, becouse of the conditions.
# while epsilon<rodilla:
while save_clus != 'stop':                    # What I mean is that with a small epsilon it may found a cluster that fulfill the condition (max diff of color), but when increasing epsilon some other stars maybe added to the cluster with a bigger diff in color and break the rule.
                     # This does not seem a problem when 'while <6' but it is when 'while <20' for example...
    loop +=1
    clustering = DBSCAN(eps=epsilon, min_samples=samples).fit(X_stad)
    print('\n'.join(('*'*len('EPSILON = %s'),'EPSILON = %s,loop = %s'%(epsilon,loop))))
    l=clustering.labels_
    epsilon +=0.01 # if choose epsilon as min d_KNN you loop over epsilon and a "<" simbol goes in the while loop
    # samples +=1 # if you choose epsilon as codo, you loop over the number of sambles and a ">" goes in the  while loop
    # print('DBSCAN loop %s. Trying with eps=%s. cluster = %s '%(loop,round(epsilon,3),len(set(l))-1))
    if loop >100:
        print('breaking out')
        break
        print('this many loops %s'%(loop))       
        print('breaking the loop')
    
    l=clustering.labels_    
    # print('clusters = %s, eps=%s'%(len(set(l))-1),round(epsilon,3))
    fig, ax = plt.subplots(1,1,figsize=(8,8))
    ax.plot(np.arange(0,len(data),1),d_KNN)
    # ax.legend(['knee=%s, min=%s, eps=%s, Dim.=%s'%(round(kneedle.elbow_y, 3),round(min(d_KNN),2),round(epsilon,2),len(X[0]))])
    ax.set_xlabel('Point') 
    ax.set_ylabel('%s-NN distance'%(samples)) 
    # print(round(kneedle.knee, 3))
    # print(round(kneedle.elbow, 3))
    # print(round(kneedle.knee_y, 3))
    # print(round(kneedle.elbow_y, 3))
    ax.axhline(rodilla,linestyle='dashed',color='k')
    ax.axhline(codo,linestyle='dashed',color='k')
    ax.axhline(round(min(d_KNN),3),linestyle='dashed',color='k')
    ax.axhline(epsilon,linestyle='dashed',color='red') 
    ax.text(len(X)/2,epsilon, '%s'%(round(epsilon,3)),color='red')
    
    ax.text(0,codo, '%s'%(codo))
    ax.text(0,rodilla, '%s'%(rodilla))
    ax.fill_between(np.arange(0,len(X)), codo, rodilla, alpha=0.5, color='grey')
    
    n_clusters = len(set(l)) - (1 if -1 in l else 0)
    # print('Group %s.Number of cluster, eps=%s and min_sambles=%s: %s'%(group,round(epsilon,2),samples,n_clusters))
    n_noise=list(l).count(-1)
    # %
    u_labels = set(l)
    colors=[plt.cm.rainbow(i) for i in np.linspace(0,1,len(set(l)))]# Returns a color for each cluster. Each color consists in four number, RGBA, red, green, blue and alpha. Full opacity black would be then 0,0,0,1
    # %
    
    # %
    for k in range(len(colors)): #give noise color black with opacity 0.1
        if list(u_labels)[k] == -1:
            colors[k]=[0,0,0,0.1]
    # %      
    colores_index=[]
    
    for c in u_labels:
        cl_color=np.where(l==c)
        colores_index.append(cl_color)
    # %
    # print(colores_index)
    for files_to_erase in glob.glob(pruebas +  'all_%s_%scluster*_eps%s.txt'%(name,pre,epsilon)):
        os.remove(files_to_erase)
    if epsilon >= round(min(d_KNN),3)+0.01:
        if n_clusters > 0:
            for i in range(len(set(l))-1):
                
                min_c=min(data[:,3][colores_index[i]]-data[:,4][colores_index[i]])
                max_c=max(data[:,3][colores_index[i]]-data[:,4][colores_index[i]])
                min_Ks=min(data[:,4][colores_index[i]])
                min_nth = np.sort(data[:,4][colores_index[i]])
                # index1=np.where((catal[:,5]==Ms[0,4]) & (catal[:,6]==Ms[0,5]) ) # looping a picking the stars coord on the Ms catalog
                
                min_c_J=min(data[:,2][colores_index[i]]-data[:,4][colores_index[i]])
                max_c_J=max(data[:,2][colores_index[i]]-data[:,4][colores_index[i]])
            
                if max_c-min_c <0.5 and any(min_nth<14):
                    fig, ax = plt.subplots(1,3,figsize=(30,10))
                    # fig, ax = plt.subplots(1,3,figsize=(30,10))
                    # ax[2].invert_yaxis()
                   
                    ax[0].set_title('Min %s-NN distance = %s. %s '%(samples,round(min(d_KNN),3),clus_method))
                    # t_gal['l'] = t_gal['l'].wrap_at('180d')
                    ax[0].scatter(X[:,0][colores_index[-1]],X[:,1][colores_index[-1]], color=colors[-1],s=50,zorder=1)
                    ax[0].scatter(X[:,0],X[:,1], color=colors[-1],s=50,zorder=1)
                    # ax[1].quiver(t_gal['l'][colores_index[-1]].value,t_gal['b'][colores_index[-1]].value, X[:,0][colores_index[-1]]-pms[2], X[:,1][colores_index[-1]]-pms[3], alpha=0.5, color=colors[-1])
            
                    ax[0].scatter(X[:,0][colores_index[i]],X[:,1][colores_index[i]], color='blueviolet',s=50,zorder=3)
                    ax[0].set_xlim(-10,10)
                    ax[0].set_ylim(-10,10)
                    ax[0].set_xlabel(r'$\mathrm{\mu_{l} (mas\ yr^{-1})}$') 
                    ax[0].set_ylabel(r'$\mathrm{\mu_{b} (mas\ yr^{-1})}$') 
                    ax[0].invert_xaxis()
                    ax[0].hlines(0,-10,10,linestyle = 'dashed', color ='red')
                    mul_sig, mub_sig = np.std(X[:,0][colores_index[i]]), np.std(X[:,1][colores_index[i]])
                    mul_mean, mub_mean = np.mean(X[:,0][colores_index[i]]), np.mean(X[:,1][colores_index[i]])
                    
                    mul_sig_all, mub_sig_all = np.std(X[:,0]), np.std(X[:,1])
                    mul_mean_all, mub_mean_all = np.mean(X[:,0]), np.mean(X[:,1])
                    
                    
                    vel_txt = '\n'.join(('mul = %s, mub = %s'%(round(mul_mean,3), round(mub_mean,3)),
                                         '$\sigma_{mul}$ = %s, $\sigma_{mub}$ = %s'%(round(mul_sig,3), round(mub_sig,3)))) 
                    vel_txt_all = '\n'.join(('mul = %s, mub = %s'%(round(mul_mean_all,3), round(mub_mean_all,3)),
                                         '$\sigma_{mul}$ = %s, $\sigma_{mub}$ = %s'%(round(mul_sig_all,3), round(mub_sig_all,3))))
                    
                    propiedades = dict(boxstyle='round', facecolor=colors[i], alpha=0.2)
                    propiedades_all = dict(boxstyle='round', facecolor=colors[-1], alpha=0.1)
                    ax[0].text(0.05, 0.95, vel_txt, transform=ax[0].transAxes, fontsize=14,
                        verticalalignment='top', bbox=propiedades)
                    ax[0].text(0.05, 0.85, vel_txt_all, transform=ax[0].transAxes, fontsize=14,
                        verticalalignment='top', bbox=propiedades_all)
                    if pixel =='no':
                        t_gal['l'] = t_gal['l'].wrap_at('180d')
                        
                       
                        ax[1].quiver(t_gal['l'][colores_index[-1]].value,t_gal['b'][colores_index[-1]].value, X[:,0][colores_index[-1]]-pms[2], X[:,1][colores_index[-1]]-pms[3], alpha=0.5, color=colors[-1])
                        ax[1].scatter(t_gal['l'][colores_index[i]].value,t_gal['b'][colores_index[i]].value, color=colors[i],s=50,zorder=3)#plots in galactic
                        ax[1].scatter(t_gal['l'].value,t_gal['b'].value, color=colors[-1],s=50,zorder=1,alpha=0.5)#plots in galactic
                        ax[1].quiver(t_gal['l'].value,t_gal['b'].value, X[:,0]-pms[2], X[:,1]-pms[3], alpha=0.5, color=colors[-1])
                        ax[1].scatter(t_gal['l'][colores_index[i]].value,t_gal['b'][colores_index[i]].value, color=colors[i],s=50,zorder=3)#plots in galactic           
                        ax[1].quiver(t_gal['l'][colores_index[i]].value,t_gal['b'][colores_index[i]].value, X[:,0][colores_index[i]]-pms[2], X[:,1][colores_index[i]]-pms[3], alpha=0.5, color=colors[i])
                        ax[1].set_xlabel('l') 
                        ax[1].set_ylabel('b') 
                        ax[1].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
                        ax[1].xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
                
                    else:
                        # ax[1].scatter(X[:,2], X[:,3], color=colors[-1],s=50,zorder=1,alpha=0.01)#plots in galactic
                        # ax[1].quiver(X[:,2], X[:,3], X[:,0]-pms[2], X[:,1]-pms[3], alpha=0.5, color=colors[-1],zorder=1)
                        # ax[1].quiver(X[:,2][colores_index[-1]], X[:,3][colores_index[-1]], X[:,0][colores_index[-1]]-pms[2], X[:,1][colores_index[-1]]-pms[3], alpha=0.5, color=colors[-1])
                       # t_gal['l'].value,t_gal['b'].value
                       
                        radio = 500*u.arcsec
                        ax[1].set_title('Radio = %s, max d$\mu_{l,b}$ = %s'%(radio,dmu_lim))
                        
                        c2 = SkyCoord(ra = data[:,0][colores_index[i]]*u.deg,dec = data[:,1][colores_index[i]]*u.deg)
                        sep = [max(c2[c_mem].separation(c2)) for c_mem in range(len(c2))]
                        rad = max(sep)/2
                         
                        prop = dict(boxstyle='round', facecolor=colors[i], alpha=0.2)
                        ax[1].text(0.65, 0.95, 'aprox cluster radio = %s"\n cluster stars = %s '%(round(rad.to(u.arcsec).value,2),len(colores_index[i][0])), transform=ax[1].transAxes, fontsize=14,
                                                verticalalignment='top', bbox=prop)
                        
                        id_clus, id_arc, d2d,d3d = ap_coor.search_around_sky(SkyCoord([data[:,5][colores_index[i][0][0]]*u.deg], [data[:,6][colores_index[i][0][0]]*u.deg], frame='icrs'),coordenadas, radio)
                        ax[1].scatter(X[:,2][id_arc], X[:,3][id_arc], color=colors[-1],s=50,zorder=1,alpha=0.01)#plots in galactic
                        ax[1].quiver(X[:,2][id_arc], X[:,3][id_arc], X[:,0][id_arc]*-1, X[:,1][id_arc], alpha=0.5, color=colors[-1],zorder=1)
                        
                        ax[1].scatter(X[:,2][colores_index[i]], X[:,3][colores_index[i]], color='blueviolet',s=50,zorder=3)#plots in galactic
                        ax[1].quiver(X[:,2][colores_index[i]], X[:,3][colores_index[i]], X[:,0][colores_index[i]]*-1, X[:,1][colores_index[i]], alpha=0.5, color='blueviolet')#colors[i]
                        ax[1].set_xlabel('x') 
                        ax[1].set_ylabel('y') 
                        ax[1].yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
                        ax[1].xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
                        # ax[1].scatter(catal_all[:,2][Ms_xy],catal_all[:,3][Ms_xy],color = 'red', s = 100)
                
                    # ax[2].scatter(data[:,3]-data[:,4],data[:,4], color=colors[-1],s=50,zorder=1, alpha=0.1)
                    
                    # This is for plotting the cluster and the isochrone with extiontion
                    H_Ks_yes = []
                    Ks_yes = []
                    AKs_clus_all =[]
                    for m in range(len(colores_index[i][0])):
                        clus_coord =  SkyCoord(ra=data[:,5][colores_index[i][0][m]]*u.degree, dec=data[:,6][colores_index[i][0][m]]*u.degree)
                        idx = clus_coord.match_to_catalog_sky(gns_coord)
                        gns_match = AKs_center[idx[0]]
                        # print(type(gns_match[16])) 
                        if gns_match[16] != '-' and gns_match[18] != '-':
                            AKs_clus_all.append(float(gns_match[18]))
                            H_Ks_yes.append(data[:,3][colores_index[i][0][m]]-data[:,4][colores_index[i][0][m]])
                            Ks_yes.append(data[:,4][colores_index[i][0][m]])
                        
                    ax[2].scatter(H_Ks_yes,Ks_yes, color='blueviolet',s=50,zorder=3, alpha=1)
                    ax[2].invert_yaxis()  
                    
                    AKs_clus, std_AKs = np.mean(AKs_clus_all),np.std(AKs_clus_all)
                    absolute_difference_function = lambda list_value : abs(list_value - AKs_clus)
                    AKs = min(AKs_list, key=absolute_difference_function)
                    
                    # clus_coord =  SkyCoord(ra=data[:,5][colores_index[i]]*u.degree, dec=data[:,6][colores_index[i]]*u.degree)
                    # idx = clus_coord.match_to_catalog_sky(gns_coord)
                    # gns_match = AKs_center[idx[0]]
                    # good = np.where(gns_match[:,11] == -1)
                    # if len(good[0]) != len(gns_match[:,11]):
                    #     print('%s foreground stars in this cluster'%(len(gns_match[:,11]) - len(good)))
                    # gns_match_good = gns_match[good]
                    # AKs_clus_all = [float(gns_match_good[i,18]) for i in range(len(gns_match_good[:,18]))  if gns_match_good[i,18] !='-']
                        
                    # AKs_clus, std_AKs = np.mean(AKs_clus_all),np.std(AKs_clus_all)
                    # absolute_difference_function = lambda list_value : abs(list_value - AKs_clus)
                    
                    # AKs = min(AKs_list, key=absolute_difference_function)
                    # frase = 'Diff in extintion bigger than 0.2!'
                    # # print('\n'.join((10*'§','%s %s'%(AKs_clus,AKs),10*'§')))
                    # if abs(AKs - AKs_clus)>0.2:
                    #     print(''*len(frase)+'\n'+frase+'\n'+''*len(frase))
                    # ax[2].scatter(data[:,3][colores_index[i]]-data[:,4][colores_index[i]],data[:,4][colores_index[i]], color='blueviolet',s=50,zorder=3, alpha=1)
                    # ax[2].invert_yaxis()    
                    # Thi subtract the extiontion to Ks and H and plots the isochrone withput extinction
# =============================================================================
#                     H_Ks_yes = []
#                     Ks_yes = []
#                     for m in range(len(colores_index[i][0])):
#                         clus_coord =  SkyCoord(ra=data[:,5][colores_index[i][0][m]]*u.degree, dec=data[:,6][colores_index[i][0][m]]*u.degree)
#                         idx = clus_coord.match_to_catalog_sky(gns_coord)
#                         gns_match = AKs_center[idx[0]]
#                         # print(type(gns_match[16])) 
#                         if gns_match[16] != '-' and gns_match[18] != '-':
#                             H_Ks_yes.append((data[:,3][colores_index[i][0][m]]-float(gns_match[16]))-((data[:,4][colores_index[i][0][m]]-float(gns_match[18]))))
#                             Ks_yes.append(data[:,4][colores_index[i][0][m]]-float(gns_match[18]))
#                         
#                     ax[2].scatter(H_Ks_yes,Ks_yes, color='blueviolet',s=50,zorder=3, alpha=1)
#                     ax[2].invert_yaxis()  
#                     
#                     AKs_clus = np.mean(H_Ks_yes)
#                     std_AKs = np.std(H_Ks_yes)
#                     absolute_difference_function = lambda list_value : abs(list_value - AKs_clus)
#                     AKs = min(AKs_list, key=absolute_difference_function)
# =============================================================================
                    
                    iso_dir = '/Users/amartinez/Desktop/PhD/Libralato_data/nsd_isochrones/'
                    
                    dist = 8000 # distance in parsec
                    metallicity = 0.17 # Metallicity in [M/H]
                    # logAge_600 = np.log10(0.61*10**9.)
                    logAge = np.log10(0.010*10**9.)
                    logAge_30 = np.log10(0.030*10**9.)
                    logAge_60 = np.log10(0.060*10**9.)
                    logAge_90 = np.log10(0.090*10**9.)
                    evo_model = evolution.MISTv1() 
                    atm_func = atmospheres.get_merged_atmosphere
                    red_law = reddening.RedLawNoguerasLara18()
                    filt_list = ['hawki,J', 'hawki,H', 'hawki,Ks']
                    
                    iso =  synthetic.IsochronePhot(logAge, AKs, dist, metallicity=metallicity,
                                                    evo_model=evo_model, atm_func=atm_func,
                                                    red_law=red_law, filters=filt_list,
                                                        iso_dir=iso_dir)
                    
                    iso_30 = synthetic.IsochronePhot(logAge_30, AKs, dist, metallicity=metallicity,
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
                    
                    
                    mass = 0.5*10**4.
                    mass = 1 * mass
                    dAks = round(std_AKs*1,3)
                    cluster = synthetic.ResolvedClusterDiffRedden(iso, my_imf, mass,dAks)
                    cluster_ndiff = synthetic.ResolvedCluster(iso, my_imf, mass)
                    clus = cluster.star_systems
                    clus_ndiff = cluster_ndiff.star_systems
                    ax[2].set_title('Cluster %s, eps = %s'%(i,round(epsilon,3)))
                    ax[2].scatter(clus['m_hawki_H']-clus['m_hawki_Ks'],clus['m_hawki_Ks'],color = 'lavender',alpha=0.1)
                    ax[2].scatter(clus_ndiff['m_hawki_H']-clus_ndiff['m_hawki_Ks'],clus_ndiff['m_hawki_Ks'],color = 'k',alpha=0.1,s=1)
                    
        
                    txt_srn = '\n'.join(('metallicity = %s'%(metallicity),'dist = %.1f Kpc'%(dist/1000),'mass =%.0fx$10^{3}$ $M_{\odot}$'%(mass/1000),
                                         'age = %.0f Myr'%(10**logAge/10**6)))
                    txt_AKs = '\n'.join(('AKs = %.2f'%(AKs_clus),'std_AKs = %.2f'%(std_AKs)))
                    props = dict(boxstyle='round', facecolor='w', alpha=0.5)
                    # place a text box in upper left in axes coords
                    ax[2].text(0.65, 0.95, txt_AKs, transform=ax[2].transAxes, fontsize=14,
                        verticalalignment='top', bbox=props)
                    ax[2].text(0.65, 0.85, txt_srn, transform=ax[2].transAxes, fontsize=14,
                        verticalalignment='top', bbox=props)
                    ax[2].plot(iso.points['m_hawki_H'] - iso.points['m_hawki_Ks'], 
                                      iso.points['m_hawki_Ks'], 'b-',  label='10 Myr')
                    
                    ax[2].plot(iso_30.points['m_hawki_H'] - iso_30.points['m_hawki_Ks'], 
                                      iso_30.points['m_hawki_Ks'], 'orange',  label='30 Myr')
                    ax[2].plot(iso_60.points['m_hawki_H'] - iso_60.points['m_hawki_Ks'], 
                                      iso_60.points['m_hawki_Ks'], 'green' ,label='60 Myr')
                    ax[2].plot(iso_90.points['m_hawki_H'] - iso_90.points['m_hawki_Ks'], 
                                      iso_90.points['m_hawki_Ks'], 'red' ,label='90 Myr')
                    ax[2].set_xlabel('H$-$Ks')
                    ax[2].set_ylabel('Ks')
                    ax[2].legend(loc =3, fontsize = 12)
                    plt.show()
                    if pixel == 'yes':
                        clus_array = np.array([data[:,5][colores_index[i]],data[:,6][colores_index[i]],t_gal['l'][colores_index[i]].value,t_gal['b'][colores_index[i]].value,
                                                                                              X[:,0][colores_index[i]], 
                                                                                              X[:,1][colores_index[i]],
                                                                                              data[:,3][colores_index[i]],data[:,4][colores_index[i]]]).T
                        clus_array1= np.c_[clus_array, np.full((len(X[:,0][colores_index[i]]),1),i)]
                        frase = 'Do you want to save this cluster?\n("yes", "no" or "jump" to a new epsilon)'
                        print('\n'.join((len(frase)*'π',frase,len(frase)*'π')))
                        save_clus = input('Awnser:')
                        print('You said: %s'%(save_clus))
                        if save_clus =='yes' or save_clus =='y':
                            np.savetxt(pruebas + 'Sec_%s_%s_%scluster%s_eps%s.txt'%(section,name,pre,i,round(epsilon,3)),clus_array1,fmt='%.7f '*8 + '%.0f ', header ='ra, dec, l, b, pml, pmb, H, Ks,cluster')
                            for check in glob.glob(pruebas + 'Sec_%s_%s_%scluster*_eps*.txt'%(section,name,pre)):
                                if 'Sec_%s_%s_%scluster%s_eps%s.txt'%(section,name,pre,i,round(epsilon,3)) != os.path.basename(check):
                                    ra_dec_clus = clus_array[:,0:2]
                                    ra_dec = np.loadtxt(check,usecols=(0,1))
                                    aset = set([tuple(x) for x in ra_dec_clus])
                                    bset = set([tuple(x) for x in ra_dec])
                                    intersection = np.array([x for x in aset & bset])
                                    if len(intersection)> 0 and len(intersection) == len(ra_dec_clus):
                                        print('You have the exact same cluster in: %s'%(os.path.basename(check)))
                                        os.remove(pruebas + 'Sec_%s_%s_%scluster%s_eps%s.txt'%(section,name,pre,i,round(epsilon,3)))
                                    elif len(intersection)> 0 and len(intersection) < len(ra_dec_clus):
                                        print('This cluster(%s) has %s more stars than %s. New will be saved old removed'
                                              %('Sec_%s_%s_%scluster%s_eps%s.txt'%(section, name,pre,i,round(epsilon,3)),len(ra_dec_clus) - len(intersection),os.path.basename(check)))
                                        os.remove(check)
                                        # np.savetxt(pruebas + 'all_%s_%scluster%s_eps%s.txt'%(name,pre,i,round(epsilon,3)),clus_array1,fmt='%.7f '*8 + '%.0f ', header ='ra, dec, l, b, pml, pmb, H, Ks,cluster')
                        elif save_clus == 'stop':
                            sys.exit('You stoped it')
                        elif save_clus =='jump':
                            new_epsilon = input('Select the new epsilon:')
                            epsilon = float(new_epsilon) -0.01
                            break
# %%
# for check in glob.glob(pruebas + 'all_%s_%scluster*_eps*.txt'%(name,pre)):
#     ra_dec_clus = clus_array[:,0:2]
#     ra_dec = np.loadtxt(check,usecols=(0,1))
#     aset = set([tuple(x) for x in ra_dec_clus])
#     bset = set([tuple(x) for x in ra_dec])
#     intersection = np.array([x for x in aset & bset])
#     if len(intersection) == len(ra_dec_clus):
#         print('You have the exact same cluster in: %s'%(os.path.basename(check)))
#     elif len(intersection) > len(ra_dec):
#         print('This cluster has %s more stars than %s. It will be saved'
#               %(len(intersection) - len(ra_dec),os.path.basename(check)))
# # %%
# print(len(ra_dec_clus))       
# A = np.array([[1, 4], [2, 5], [3, 6]])
# B = np.array([[1, 4], [3, 6], [7, 8]])
# aset = set([tuple(x) for x in A])
# bset = set([tuple(x) for x in B])
# intersection = np.array([x for x in aset & bset])
# print(intersection)
# %%
H_Ks_yes = []
Ks_yes = []
for m in range(len(colores_index[i][0])):
    clus_coord =  SkyCoord(ra=data[:,5][colores_index[i][0][m]]*u.degree, dec=data[:,6][colores_index[i][0][m]]*u.degree)
    idx = clus_coord.match_to_catalog_sky(gns_coord)
    gns_match = AKs_center[idx[0]]
    # print(type(gns_match[16])) 
    if gns_match[16] != '-' and gns_match[18] != '-':
        H_Ks_yes.append((data[:,3][colores_index[i][0][m]]-float(gns_match[16]))-((data[:,4][colores_index[i][0][m]]-float(gns_match[18]))))
        Ks_yes.append(data[:,4][colores_index[i][0][m]]-float(gns_match[18]))
# %%
print(np.std(H_Ks_yes))

        
