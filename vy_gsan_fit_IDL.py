#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 13:52:00 2022

@author: amartinez
"""

import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner
import dynesty
import scipy.integrate as integrate
from astropy.stats import sigma_clip
from astropy.stats import sigma_clipped_stats
from matplotlib.ticker import FormatStrFormatter

from matplotlib import rcParams
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
cata='/Users/amartinez/Desktop/PhD/Libralato_data/CATALOGS/'
pruebas='/Users/amartinez/Desktop/PhD/Libralato_data/pruebas/'



#R.A. Dec. X Y μαcosδ σμαcosδ μδ σμδ  time n1 n2 ID

# name='ACSWFC'
name='WFC3IR'
ra,dec,x_c ,y_c,mua,dmua,mud,dmud, time, n1, n2, idt = np.loadtxt(cata+'GALCEN_%s_PM.cat'%(name),unpack=True)
gl,gb,mul,mub,dmul,dmub=np.loadtxt(pruebas + '%s_ecu_to_gl_IDL.txt'%(name), unpack=True)

#%%
good=np.where((dmua<90)&(dmua<5))
ra=ra[good]
dec=dec[good]

mua=mua[good]
dmua=dmua[good]
mud=mud[good]
dmud=dmud[good]
mul=mul[good]
mub=mub[good]
dmul=dmul[good]
dmub=dmub[good]

time=time[good]
n1=n1[good]
n2=n2[good]
idt=idt[good]

#%%
#%%
# Transform of coordinates has been already transform with ecu_to_gal.pro, also have been trasformed the uncertainties in the same way.
#
# for now Ill just leave the like they are
dmul=dmua
dmub=dmud

#%%
good=np.where((mul<70) & (mul>-70))
ra=ra[good]
dec=dec[good]
mul=mul[good]
dmul=dmul[good]
mub=mub[good]
dmub=dmub[good]
time=time[good]
n1=n1[good]
n2=n2[good]
idt=idt[good]
#%%
perc_dmul= np.percentile(dmul,85)
print(perc_dmul,'yomama')
# lim_dmul=perc_dmul
lim_dmul=1
accu=np.where((abs(dmul)<lim_dmul) & (abs(dmub)<lim_dmul))
#%%
mul=mul[accu]
mub=mub[accu]
dmul=dmul[accu]
dmub=dmub[accu]
time=time[accu]
#%%
print(min(mul),max(mul))

auto='auto'
if auto !='auto':
    auto=np.arange(min(mul),max(mul),0.25)#also works if running each bing width one by one, for some reason...
    # print(auto)

#%%


#%%
fig, ax = plt.subplots(1,1, figsize=(10,10))

# sig_hb=sigma_clip(mub,sigma=500,maxiters=20,cenfunc='mean',masked=True)
# mub=mub[sig_hb.mask==False]

hb=ax.hist(mub,bins=auto,color='orange',linewidth=2,density=True)
hb1=np.histogram(mub,bins=auto,density=False)

xb=[hb[1][i]+(hb[1][1]-hb[1][0])/2 for i in range(len(hb[0]))]#middle value for each bin
ax.axvline(np.mean(mub), color='r', linestyle='dashed', linewidth=3)
ax.legend(['List=%s, %s, mean= %.2f, std=%.2f'
                  %(name,len(mub),np.mean(mub),np.std(mub))],fontsize=12,markerscale=0,shadow=True,loc=1,handlelength=-0.0)
ax.set_ylabel('N')
ax.set_xlim(-10,10)
ax.set_xlabel(r'$\mathrm{\mu_{b} (mas\ yr^{-1})}$') 
yb=hb[0]#height for each bin
#%%

# =============================================================================
# fig, ax =plt.subplots(1,1,figsize=(10,10))
# ax.scatter(mul,mub,color='k',s=1,alpha=0.05)
# ax.set_xlim(-13,2)
# ax.set_ylim(-10,10)
# ax.axvline(0)
# ax.axhline(0)
# ax.axhline(-0.22)
# ax.invert_xaxis()
# 
# =============================================================================
# %%

yerr=[]
yb=np.where(yb==0,0.001,yb)
y1=hb1[0]
y1=np.where(y1==0,0.001,y1)
yerr = yb*np.sqrt(1/y1)
# yerr = y*np.sqrt(1/y1+1/len(v_y))

    
# In[7]:   
def gaussian(x, mu, sig, amp):
    return amp * (1 / (sig * (np.sqrt(2 * np.pi)))) * np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.))) 
#%%
def loglike(theta):
    mu1, sigma1, amp1,mu2,sigma2,amp2 = theta
    model = gaussian(xb, mu1, sigma1, amp1)+gaussian(xb,mu2,sigma2,amp2)
     
    return -0.5 * np.sum(((yb - model)/yerr) ** 2)#chi squared model
#%% 
def prior_transform(utheta):
    """Transforms the uniform random variable `u ~ Unif[0., 1.)`
    to the parameter of interest `x ~ Unif[-10., 10.)`."""
    #x = 2. * u - 1.  # scale and shift to [-1., 1.)
    #x *= 10.  # scale to [-10., 10.)
    umu1, usigma1, uamp1,  umu2, usigma2, uamp2= utheta

#     mu1 = -1. * umu1-8   # scale and shift to [-10., 10.)
    mu1 = 2* umu1-1  # scale and shift to [-10., 10.)
    sigma1 = 5* (usigma1)   
    amp1 = 1 * uamp1 

    
    mu2 = 2 * umu2-1
    sigma2 = 2 * usigma2   
    amp2 = 1* uamp2   
    

    return mu1, sigma1, amp1, mu2, sigma2, amp2
#%% 
sampler = dynesty.NestedSampler(loglike, prior_transform, ndim=6, nlive=500,
                                            bound='multi', sample='rwalk')
sampler.run_nested()
res = sampler.results
#%%
from dynesty import plotting as dyplot
rcParams.update({'font.size': 10})
# truths = [mu1_true, sigma1_true, amp1_true, mu2_true, sigma2_true, amp2_true]
labels = [r'$\mathrm{\mu 1}$', r'$\mathrm{\sigma 1}$', r'$amp1$', r'$\mathrm{\mu 2}$', r'$\mathrm{\sigma 2}$', r'$amp2$']
# fig, axes = dyplot.traceplot(sampler.results, truths=truths, labels=labels,
#      $\mathrm{\mu_{b}}                        fig=plt.subplots(6, 2, figsize=(16, 27)))

fig, axes = dyplot.traceplot(sampler.results,labels=labels,show_titles=True,
                             fig=plt.subplots(6, 2, figsize=(20, 16)))



plt.show()
rcParams.update({'font.size': 20})

#%%
from dynesty import utils as dyfunc
    
samples, weights = res.samples, np.exp(res.logwt - res.logz[-1])
mean, cov = dyfunc.mean_and_cov(samples, weights)
print(mean)
     #%%                              fig=plt.subplots(6, 6, figsize=(28, 28)))
    # This is de corner plot
fig, axes = dyplot.cornerplot(res, color='royalblue', show_titles=True, quantiles=[0.16,0.5,0.68],truths=mean,
                              quantiles_2d=[0.16,0.5,0.68],
                              title_kwargs={'x': 0.65, 'y': 1.05}, labels=labels,
                              fig=plt.subplots(6, 6, figsize=(28, 28)))
plt.legend(['$\mu_{b}$ %s '%(name)],fontsize=70,markerscale=0,shadow=True,bbox_to_anchor=(1,6.5),handlelength=-0.0)

plt.show() 
    
# %%

from matplotlib import rc
rc('font',**{'family':'serif','serif':['Palatino']})


results = sampler.results
print(results['logz'][-1])

fig, ax = plt.subplots(figsize=(8,8))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
h=ax.hist(mub, bins= auto, color='royalblue', alpha = 0.6, density =True, histtype = 'stepfilled')


xplot = np.linspace(min(xb), max(xb), 1000)

# plt.plot(xplot, gaussian(xplot, mean[0], mean[1], mean[2]) , color="darkorange", linewidth=3, alpha=0.6)

plt.plot(xplot, gaussian(xplot, mean[0], mean[1], mean[2]) + gaussian(xplot, mean[3], mean[4], mean[5]), color="darkorange", linewidth=3, alpha=1)
plt.plot(xplot, gaussian(xplot, mean[0], mean[1], mean[2])  , color="red", linestyle='dashed', linewidth=3, alpha=0.6)
plt.plot(xplot, gaussian(xplot, mean[3], mean[4], mean[5])  , color="k", linestyle='dashed', linewidth=3, alpha=0.6)

plt.text(5,max(hb[0]-0.05),'$logz=%.0f$'%(results['logz'][-1]),color='b')

plt.xlim(-15,15)

plt.ylabel('N')
plt.xlabel(r'$\mathrm{\mu_{b} (mas\ yr^{-1})}$')
# %%
samples, weights = res.samples, np.exp(res.logwt - res.logz[-1])
mean, cov = dyfunc.mean_and_cov(samples, weights)
# print(mean)
quantiles = [dyfunc.quantile(samps, [0.16,0.5,0.84], weights=weights)
             for samps in samples.T]

for i in range(6):
    print('medin %.2f -+ %.2f %.2f'%(quantiles[i][1],quantiles[i][1]-quantiles[i][0],quantiles[i][2]-quantiles[i][1]))
    print(' mean %.2f -+ %.2f %.2f'%(mean[i],mean[i]-quantiles[i][0],quantiles[i][2]-mean[i])+'\n'+30*'*')
   








