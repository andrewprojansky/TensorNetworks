#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MPS_Plotting.py: 
    Code for plotting circuits acting on MPS - including gates, and 
    measurements
    
TO-DO
------
    Make more user friendly
    Text on gates?
    Quimb interface?

@author: andrewprojansky
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
#%%
# Number of data points

"""
ASSUMES MEASUREMENTS HAPPEN AFTER GATES
"""

sites = 8; layers = 5
gate_l = [[(0,1),(4,),(5,6)],[(2,3)], [(0,1,2,3,4), (6,7)]]
meas_l = [[0,2,4,5,7], [1,6,7], [3,4,5]]

def sitecirc(sites, ax, l=1, fc='indianred'):
    """
    Generates row of circles representing sites

    Parameters
    ----------
    sites : int
        number of sites
    ax : matplotlib.axes._subplots.AxesSubplot
        subplot objects are added to
    l : integer, optional
        position of sites. The default is 1.
    fc : string, optional
        color string for circles. The default is 'indianred'.

    """
    scirc = [Circle((j,l), 0.25) for j in np.arange(1,sites+1, 1)]
    pcirc = PatchCollection(scirc, facecolor=fc, alpha=0.9, edgecolor='darkred')
    ax.add_collection(pcirc)
    
def vlines(sites, ax, lc = 'black', lays = layers):
    """
    Generates vertical lines representing physical bonds

    Parameters
    ----------
    sites : int
        number of sites
    ax : matplotlib.axes._subplots.AxesSubplot
        subplot objects are added to
    lc : string, optional
        color string for lines. The default is 'black'.
    lays : int, optional
        lnumber of layers line to pass through. The default is layers.

    """
    
    for k in np.arange(1, sites+1, 1):
        ax.plot([k,k], [1+0.25, lays-0.25], color=lc, alpha=0.9, zorder=0)

def gatecirc(ax, gate_l, layer_l = np.arange(0, sites, 1), 
             fc='lightsteelblue'):
    """
    Generates rectangles representing gates

    Parameters
    ----------
    ax : matplotlib.axes._subplots.AxesSubplot
        subplot objects are added to
    gate_l : list
        list of sites that each gate is applied to
    layer_l : array, optional
        array of layers. The default is np.arange(0, sites, 1).
    fc : string, optional
        color string for gates. The default is 'lightsteelblue'.


    """
    
    if len(gate_l) < sites:
        layer_l = np.arange(0, len(gate_l), 1)
    gcirc = []
    for l in layer_l:
        for indpair in gate_l[l]:
            bl = len(indpair)
            st_ind = indpair[0]
            gcirc.append(Rectangle((st_ind+1-0.35, l+2-0.35), bl-0.3, 0.55))
    gatec = PatchCollection(gcirc, facecolor = fc, edgecolor='black')
    ax.add_collection(gatec)
        
def meascirc(ax, meas_l, layer_l = np.arange(0, sites, 1), 
             fc='lightsteelblue'):
    """
    Generates X markers representing measurement

    Parameters
    ----------
    ax : matplotlib.axes._subplots.AxesSubplot
        subplot objects are added to
    meas_l : list
        list of sites measured at each layer
    layer_l : array, optional
        array of layers. The default is np.arange(0, sites, 1).
    fc : string, optional
        color string for gates. The default is 'red'.

    """
    
    if len(meas_l) < sites:
        layer_l = np.arange(0, len(meas_l), 1)
    for l in layer_l:
        for m in meas_l[l]:
            ax.plot(m+1, l+2.425, 'x', color='red', markersize=8)
'''
fig, ax = plt.subplots(1)
vlines(sites, ax, lays=len(gate_l) + 2)
sitecirc(sites, ax, fc = 'firebrick')
gatecirc(ax, gate_l)
meascirc(ax, meas_l)
###
#If doing an inner product...
#sitecirc(sites, ax, l = len(gate_l)+2)
###
ax.set_aspect('equal', adjustable='box')
ax.plot(sites,layers)
ax.axis('off')
plt.show()
'''