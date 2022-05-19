#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 18 13:57:08 2022

@author: amartinez
"""

# =============================================================================
# Create simulated data based on real positions. Velocities made as normal distribution
# with means and sigmas equals to the NSD.
# =============================================================================

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
import random
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

section = 'A'
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
# "'RA_gns','DE_gns','Jmag','Hmag','Ksmag','ra','dec','x_c','y_c','mua','dmua','mud','dmud','time','n1','n2','ID','mul','mub','dmul','dmub','m139','Separation'",
catal=np.loadtxt(results + 'sec_%s_%smatch_GNS_and_%s_refined_galactic.txt'%(section,pre,name))
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
# This is if we want to use only stars streaming eastward
# east = np.where(catal[:,-6]>-5.72)
# catal_east=catal
# catal_east=catal[east]


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
AKs_list =  np.arange(1.6,2.11,0.01)
# %%
# Generates the normal distributed samples, with paramtres from dynesty adjustment 
# for the whole set of data (central and dmu(l,b)<1)
mul_e, mul_b, mul_w,sig_e, sig_b, sig_w, area_1, area_2, area_3 = np.loadtxt(pruebas + 'gaus_mul_sec_%s.txt'%(section),unpack = 'True')
mub_b, mub_nsd, sig_b, sig_nsd, area_b, area_nsd = np.loadtxt(pruebas + 'gaus_mub_sec_%s.txt'%(section),unpack = 'True')

# he aproximate the number of for each feature of the nsd by calculating the area under each gaussian (from the above lists)
# and timing it by the total len of the data set
gaul_1 =  int(len(catal)*area_1)
gaul_2 =  int(len(catal)*area_2)
gaul_3 =  int(len(catal)*area_3)
if gaul_1 + gaul_2 + gaul_3 != len(catal):
    fix =len(catal) - (gaul_1 + gaul_2 + gaul_3)
    gaul_3 = gaul_3 + fix
print(len(catal),gaul_1 + gaul_2 + gaul_3)
mul_sim_e = np.random.normal(loc=mul_e, scale=sig_e, size=gaul_1)
mul_sim_b = np.random.normal(loc=mul_b, scale=sig_b, size=gaul_2)
mul_sim_w = np.random.normal(loc=mul_w, scale=sig_w, size=gaul_3)

gaub_1 =  int(len(catal)*area_b)
gaub_2 =  int(len(catal)*area_nsd)

if gaub_1 + gaub_2  != len(catal):
    fix =len(catal) - (gaub_1 + gaub_2)
    gaub_2 = gaub_2 + fix

mub_sim_b = np.random.normal(loc=mub_b, scale=sig_b, size=gaub_1)
mub_sim_nsd = np.random.normal(loc=mub_nsd, scale=sig_nsd, size=gaub_2)

# %
H_shu = catal[:,3]
Ks_shu = catal[:,4]
np.random.shuffle(H_shu)
np.random.shuffle(Ks_shu)
# %
mul_sim = np.concatenate((mul_sim_e,mul_sim_b,mul_sim_w))
mub_sim = np.concatenate((mub_sim_b,mub_sim_nsd))
# %
# catal_sim = np.c_[catal[:,7],catal[:,8],mul_sim,mub_sim,
#                   catal[:,3],catal[:,4]]
catal_sim = np.c_[catal[:,7],catal[:,8],mul_sim,mub_sim,
                  H_shu,Ks_shu]
fig, ax = plt.subplots(1,3,figsize=(30,10))
ax[0].scatter(catal_sim[:,0],catal_sim[:,1])
ax[1].scatter(catal[:,-6],catal[:,-5])
ax[2].set_xlim(-20,15)
ax[1].set_xlim(-20,15)
ax[2].set_ylim(min(catal[:,-5]),max(catal[:,-5]))

# ax[1].set_ylim(-4,4)

ax[2].scatter(catal_sim[:,2],catal_sim[:,3])
ax[1].invert_xaxis()
ax[2].invert_xaxis()

# %
pms =[0,0,0,0]
X=np.array([catal_sim[:,2]-pms[2],catal_sim[:,3]-pms[3],catal_sim[:,0],catal_sim[:,1]]).T
X_stad = StandardScaler().fit_transform(X)

tree=KDTree(X_stad, leaf_size=2) 

samples=10# number of minimun objects that defined a cluster
samples_dist = samples# t

dist, ind = tree.query(X_stad, k=samples_dist) #DistNnce to the 1,2,3...k neighbour
d_KNN=sorted(dist[:,-1])#distance to the Kth neighbour

kneedle = KneeLocator(np.arange(0,len(catal_sim),1), d_KNN, curve='convex', interp_method = "polynomial",direction="increasing")
elbow = KneeLocator(np.arange(0,len(catal_sim),1), d_KNN, curve='concave', interp_method = "polynomial",direction="increasing")
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
max_c = 1
min_c = 0
# while len(set(l))<10:# min number of cluster to find. It star looking at the min values of the Knn distance plot and increases epsilon until the cluster are found. BE careful cose ALL cluster will be found with the lastest (and biggest) value of eps, so it might lost some clusters, becouse of the conditions.
while (max_c-min_c) > 0.3:                         # What I mean is that with a small epsilon it may found a cluster that fulfill the condition (max diff of color), but when increasing epsilon some other stars maybe added to the cluster with a bigger diff in color and break the rule.
                         # This does not seem a problem when 'while <6' but it is when 'while <20' for example...
    loop +=1
    clustering = DBSCAN(eps=epsilon, min_samples=samples).fit(X_stad)
    
    l=clustering.labels_
    epsilon +=0.001 # if choose epsilon as min d_KNN you loop over epsilon and a "<" simbol goes in the while loop
    # samples +=1 # if you choose epsilon as codo, you loop over the number of sambles and a ">" goes in the  while loop
    print('DBSCAN loop %s. Trying with eps=%s. cluster = %s '%(loop,round(epsilon,3),len(set(l))-1))
    if loop >1000:
        # print('breaking out')
        break
       
    # print('breaking the loop')
    print('This is the number of clusters: %s'%(len(set(l))-1))
    
    # =============================================================================
    fig, ax = plt.subplots(1,1,figsize=(8,8))
    ax.plot(np.arange(0,len(catal_sim),1),d_KNN)
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
    # =============================================================================
    
    # %Plots the vector poits plots for all the selected stars
    # =============================================================================
    #     fig, ax = plt.subplots(1,1,figsize=(8,8))
    #     # ax.scatter(X[:,0],X[:,1],s=10,alpha=0.5)
    #     ax.scatter(data[:,-4],data[:,-3],s=10,alpha=0.5)
    #     # ax.set_xlim(-15,15)
    #     # ax.set_ylim(-15,15)
    #     ax.set_xlabel(r'$\mathrm{\mu_{l} (mas\ yr^{-1})}$') 
    #     ax.set_ylabel(r'$\mathrm{\mu_{b} (mas\ yr^{-1})}$') 
    #     ax.set_title('Group %s'%(group))
    # =============================================================================
    #%
    
    
    
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
    if n_clusters > 0:
        #This plots the need plot
    # =============================================================================
    #         fig, ax = plt.subplots(1,1,figsize=(8,8))
    #         ax.plot(np.arange(0,len(data),1),d_KNN)
    #         ax.legend(['knee=%s, min=%s, eps=%s, Dim.=%s'%(round(kneedle.elbow_y, 3),round(min(d_KNN),2),round(epsilon,2),len(X[0]))])
    #         ax.set_xlabel('Point') 
    #         ax.set_ylabel('%s-NN distance'%(samples)) 
    #         # print(round(kneedle.knee, 3))
    #         # print(round(kneedle.elbow, 3))
    #         # print(round(kneedle.knee_y, 3))
    #         # print(round(kneedle.elbow_y, 3))
    #         ax.axhline(round(kneedle.elbow_y, 3),linestyle='dashed',color='k')
    # =============================================================================
        
        # print(''*len('RESTORE CONDITIONS')+'\n'+'RESTORE CONDITIONS'+'\n'+''*len('RESTORE CONDITIONS'))
    
        for i in range(len(set(l))-1):
            
                
            min_c=min(catal_sim[:,4][colores_index[i]]-catal_sim[:,5][colores_index[i]])
            max_c=max(catal_sim[:,4][colores_index[i]]-catal_sim[:,5][colores_index[i]])
            min_Ks=min(catal_sim[:,5][colores_index[i]])
            min_nth = np.sort(catal_sim[:,5][colores_index[i]])
            # index1=np.where((catal[:,5]==Ms[0,4]) & (catal[:,6]==Ms[0,5]) ) # looping a picking the stars coord on the Ms catalog
            
            
        
            if max_c-min_c <0.3 and any(min_nth<14.5):
            # if max_c-min_c <0.3:
                
            
                fig, ax = plt.subplots(1,3,figsize=(30,10))
                
                
               
              
                ax[1].set_title('Cluster %s, diff color = %s'%(i,round(max_c-min_c,3)))
                # t_gal['l'] = t_gal['l'].wrap_at('180d')
                ax[0].scatter(X[:,0][colores_index[-1]],X[:,1][colores_index[-1]], color=colors[-1],s=50,zorder=1)
                ax[0].scatter(X[:,0],X[:,1], color=colors[-1],s=50,zorder=1)
                # ax[1].quiver(t_gal['l'][colores_index[-1]].value,t_gal['b'][colores_index[-1]].value, X[:,0][colores_index[-1]]-pms[2], X[:,1][colores_index[-1]]-pms[3], alpha=0.5, color=colors[-1])
        
                ax[0].scatter(X[:,0][colores_index[i]],X[:,1][colores_index[i]], color=colors[i],s=50,zorder=3)
                # ax[0].set_xlim(-10,10)
                # ax[0].set_ylim(-10,10)
                ax[0].set_xlabel(r'$\mathrm{\mu_{l} (mas\ yr^{-1})}$') 
                ax[0].set_ylabel(r'$\mathrm{\mu_{b} (mas\ yr^{-1})}$') 
            
                
                ax[0].invert_xaxis()
                # Here we save the coordenates of the posible cluster coordinates for further anlysis if required
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
        
                
                ax[1].scatter(X[:,2], X[:,3], color='k',s=50,zorder=3)#plots in galactic
                ax[1].quiver(X[:,2], X[:,3], X[:,0], X[:,1], alpha=0.5, color='k')#colors[i]
          
                ax[1].scatter(X[:,2][colores_index[i]], X[:,3][colores_index[i]], color=colors[i],s=50,zorder=3)#plots in galactic
                ax[1].quiver(X[:,2][colores_index[i]], X[:,3][colores_index[i]], X[:,0][colores_index[i]]*-1, X[:,1][colores_index[i]], alpha=0.5, color=colors[i])#colors[i]
                ax[1].set_xlabel('x') 
                ax[1].set_ylabel('y') 
                
                c2 = SkyCoord(ra = catal[:,0][colores_index[i]]*u.deg,dec = catal[:,1][colores_index[i]]*u.deg)
                sep = [max(c2[c_mem].separation(c2)) for c_mem in range(len(c2))]
                rad = max(sep)/2
                 
                prop = dict(boxstyle='round', facecolor=colors[i], alpha=0.2)
                ax[1].text(0.65, 0.95, 'aprox cluster radio = %s"\n cluster stars = %s '%(round(rad.to(u.arcsec).value,2),len(colores_index[i][0])), transform=ax[1].transAxes, fontsize=14,
                                        verticalalignment='top', bbox=prop)
                
                
                ax[2].scatter(catal[:,3][colores_index[i]]-catal[:,4][colores_index[i]],catal[:,4][colores_index[i]], color='blueviolet',s=50,zorder=3, alpha=1)
                ax[2].invert_yaxis()
    
                clus_coord =  SkyCoord(ra=catal[:,5][colores_index[i]]*u.degree, dec=catal[:,6][colores_index[i]]*u.degree)
                idx = clus_coord.match_to_catalog_sky(gns_coord)
                gns_match = AKs_center[idx[0]]
                good = np.where(gns_match[:,11] == -1)
                if len(good[0]) != len(gns_match[:,11]):
                    print('%s foreground stars in this cluster'%(len(gns_match[:,11]) - len(good)))
                gns_match_good = gns_match[good]
                AKs_clus_all = [float(gns_match_good[i,18]) for i in range(len(gns_match_good[:,18]))  if gns_match_good[i,18] !='-']
                    
                AKs_clus, std_AKs = np.mean(AKs_clus_all),np.std(AKs_clus_all)
                absolute_difference_function = lambda list_value : abs(list_value - AKs_clus)
                
                AKs = min(AKs_list, key=absolute_difference_function)
                frase = 'Diff in extintion bigger than 0.2!'
                # print('\n'.join((10*'§','%s %s'%(AKs_clus,AKs),10*'§')))
                if abs(AKs - AKs_clus)>0.2:
                    print(''*len(frase)+'\n'+frase+'\n'+''*len(frase))
                    
                iso_dir = '/Users/amartinez/Desktop/PhD/Libralato_data/nsd_isochrones/'
                
                dist = 8200 # distance in parsec
                metallicity = 0.17 # Metallicity in [M/H]
                # logAge = np.log10(0.61*10**9.)
                logAge = np.log10(0.010*10**9.)
                
                evo_model = evolution.MISTv1() 
                atm_func = atmospheres.get_merged_atmosphere
                red_law = reddening.RedLawNoguerasLara18()
                filt_list = ['hawki,J', 'hawki,H', 'hawki,Ks']
                
                iso =  synthetic.IsochronePhot(logAge, AKs, dist, metallicity=metallicity,
                                                evo_model=evo_model, atm_func=atm_func,
                                                red_law=red_law, filters=filt_list,
                                                    iso_dir=iso_dir)
                
                
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
                ax[2].scatter(clus['m_hawki_H']-clus['m_hawki_Ks'],clus['m_hawki_Ks'],color = 'r',label='With dAKs = %s mag'%(dAks),alpha=0.1)
                ax[2].scatter(clus_ndiff['m_hawki_H']-clus_ndiff['m_hawki_Ks'],clus_ndiff['m_hawki_Ks'],color = 'k',label='With dAKs = %s mag'%(0),alpha=0.3)
                ax[2].legend(loc =3, fontsize = 12)
    
                txt_srn = '\n'.join(('metallicity = %s'%(metallicity),'dist = %.1f Kpc'%(dist/1000),'mass =%.0fx$10^{3}$ $M_{\odot}$'%(mass/1000),
                                     'age = %.0f Myr'%(10**logAge/10**6)))
                txt_AKs = '\n'.join(('AKs = %.2f'%(AKs_clus),'std_AKs = %.2f'%(std_AKs)))
                props = dict(boxstyle='round', facecolor='w', alpha=0.5)
                # place a text box in upper left in axes coords
                ax[2].text(0.65, 0.95, txt_AKs, transform=ax[2].transAxes, fontsize=14,
                    verticalalignment='top', bbox=props)
                ax[2].text(0.65, 0.85, txt_srn, transform=ax[2].transAxes, fontsize=14,
                    verticalalignment='top', bbox=props)
                
                sys.exit('0.3 reach, fucker in loop %s, cluster %s'%(loop,i) )

# %%
fig, ax = plt.subplots(1,1,figsize =(10,10))
ax.hist(np.concatenate((mul_sim_e,mul_sim_b,mul_sim_w)),bins=151)
ax.invert_xaxis()






