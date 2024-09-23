'''
Example data for Europe power grid.
'''

import numpy as np
import matplotlib.pyplot as pl
from matplotlib.image import imread
import networkx as nx
from gridBalance import *

zoneNames = ['SE1','SE2','SE3','SE4','DK1','DK2','FI','NO1','NO2','NO3','NO4','NO5',\
'PL','NL','GER','BE','LV','LT','EE']
nZones = len(zoneNames)
zi = {zoneNames[i] : i for i in range(nZones)}

# Hämtat från https://data.nordpoolgroup.com/auction/day-ahead/capacities?deliveryDate=2024-09-18
# MW, gäller handelstimmen 17:00-18:00, 2024-09-18
capacitiesEur = np.zeros((nZones,nZones))
capacitiesEur[zi['SE4'],zi['DK2']] = 1261
capacitiesEur[zi['DK2'],zi['SE4']] = 1012
capacitiesEur[zi['SE4'],zi['LT' ]] = 650
capacitiesEur[zi['LT' ],zi['SE4']] = 650
capacitiesEur[zi['SE4'],zi['SE3']] = 2737
capacitiesEur[zi['SE3'],zi['SE4']] = 3879
capacitiesEur[zi['SE3'],zi['DK1']] = 200
capacitiesEur[zi['DK1'],zi['SE3']] = 550
capacitiesEur[zi['SE3'],zi['FI' ]] = 600
capacitiesEur[zi['FI' ],zi['SE3']] = 200
capacitiesEur[zi['SE3'],zi['NO1']] = 1248
capacitiesEur[zi['NO1'],zi['SE3']] = 1270
capacitiesEur[zi['SE3'],zi['SE2']] = 7258
capacitiesEur[zi['SE2'],zi['SE3']] = 6300
capacitiesEur[zi['SE2'],zi['NO3']] = 679
capacitiesEur[zi['NO3'],zi['SE2']] = 586
capacitiesEur[zi['SE2'],zi['NO4']] = 150
capacitiesEur[zi['NO4'],zi['SE2']] = 150
capacitiesEur[zi['SE2'],zi['SE1']] = 1060
capacitiesEur[zi['SE1'],zi['SE2']] = 1081
capacitiesEur[zi['SE1'],zi['FI' ]] = 670
capacitiesEur[zi['FI' ],zi['SE1']] = 530
capacitiesEur[zi['SE1'],zi['NO4']] = 200
capacitiesEur[zi['NO4'],zi['SE1']] = 289
capacitiesEur[zi['NO5'],zi['NO1']] = 2367
capacitiesEur[zi['NO1'],zi['NO5']] = 360
capacitiesEur[zi['NO5'],zi['NO2']] = 400
capacitiesEur[zi['NO2'],zi['NO5']] = 100
capacitiesEur[zi['NO5'],zi['NO3']] = 200
capacitiesEur[zi['NO3'],zi['NO5']] = 0
capacitiesEur[zi['NO4'],zi['NO3']] = 812
capacitiesEur[zi['NO3'],zi['NO4']] = 183
capacitiesEur[zi['NO2'],zi['NL' ]] = 506
capacitiesEur[zi['NL' ],zi['NO2']] = 640
capacitiesEur[zi['NO2'],zi['DK1']] = 1680
capacitiesEur[zi['DK1'],zi['NO2']] = 1680
capacitiesEur[zi['NO2'],zi['NO1']] = 2790
capacitiesEur[zi['NO1'],zi['NO2']] = 1555
capacitiesEur[zi['FI' ],zi['EE' ]] = 1016
capacitiesEur[zi['EE' ],zi['FI' ]] = 1016
capacitiesEur[zi['DK2'],zi['DK1']] = 600
capacitiesEur[zi['DK1'],zi['DK2']] = 531
capacitiesEur[zi['DK2'],zi['GER']] = 585
capacitiesEur[zi['GER'],zi['DK2']] = 600
capacitiesEur[zi['DK1'],zi['GER']] = 2500
capacitiesEur[zi['GER'],zi['DK1']] = 2500
capacitiesEur[zi['DK1'],zi['NL' ]] = 490
capacitiesEur[zi['NL' ],zi['DK1']] = 700
capacitiesEur[zi['PL' ],zi['LT' ]] = 492
capacitiesEur[zi['LT' ],zi['PL' ]] = 350
capacitiesEur[zi['GER'],zi['BE' ]] = 1000
capacitiesEur[zi['BE' ],zi['GER']] = 1000
capacitiesEur[zi['LV' ],zi['EE' ]] = 800
capacitiesEur[zi['EE' ],zi['LV' ]] = 1030
capacitiesEur[zi['LV' ],zi['LT' ]] = 1045
capacitiesEur[zi['LT' ],zi['LV' ]] = 1154

# Final demand, MW, 17:00-18:00, 2024-09-18
buy = np.zeros(19)
buy[zi['SE4']] = 2171
buy[zi['SE3']] = 8545
buy[zi['SE2']] = 1591
buy[zi['SE1']] = 1075
buy[zi['NO5']] = 1152
buy[zi['NO4']] = 1741
buy[zi['NO3']] = 2785
buy[zi['NO2']] = 4356
buy[zi['NO1']] = 2890
buy[zi['FI' ]] = 5229
buy[zi['DK2']] = 607
buy[zi['DK1']] = 1352
buy[zi['PL' ]] = 204
buy[zi['NL' ]] = 835
buy[zi['GER']] = 392
buy[zi['BE' ]] = 357
buy[zi['LV' ]] = 815
buy[zi['LT' ]] = 1384
buy[zi['EE' ]] = 851

# Final sell, MW, 17:00-18:00, 2024-09-18
sell = np.zeros(19)
sell[zi['SE4']] = 222
sell[zi['SE3']] = 6288
sell[zi['SE2']] = 6263
sell[zi['SE1']] = 2900
sell[zi['NO5']] = 4568
sell[zi['NO4']] = 2440
sell[zi['NO3']] = 2052
sell[zi['NO2']] = 5052
sell[zi['NO1']] = 2526
sell[zi['FI' ]] = 5195
sell[zi['DK2']] = 610
sell[zi['DK1']] = 943
sell[zi['PL' ]] = 9
sell[zi['NL' ]] = 1825
sell[zi['GER']] = 4003
sell[zi['BE' ]] = 353
sell[zi['LV' ]] = 647
sell[zi['LT' ]] = 471
sell[zi['EE' ]] = 448

# Final price, €/MWh, 17:00-18:00, 2024-09-18
p_ref = np.zeros(19)
p_ref[zi['SE4']] = 42
p_ref[zi['SE3']] = 42
p_ref[zi['SE2']] = 20
p_ref[zi['SE1']] = 17
p_ref[zi['NO5']] = 20
p_ref[zi['NO4']] = 19
p_ref[zi['NO3']] = 20
p_ref[zi['NO2']] = 42
p_ref[zi['NO1']] = 42
p_ref[zi['FI' ]] = 120
p_ref[zi['DK2']] = 88
p_ref[zi['DK1']] = 88
p_ref[zi['PL' ]] = 153
p_ref[zi['NL' ]] = 79
p_ref[zi['GER']] = 88
p_ref[zi['BE' ]] = 74
p_ref[zi['LV' ]] = 153
p_ref[zi['LT' ]] = 153
p_ref[zi['EE' ]] = 153

img = imread('map_small.png')
pos = {
'SE1':(250,130), 'SE2':(200,250), 'SE3':(200,360), 'SE4':(190,430),
'DK1':(100,430), 'DK2':(140,470), 'FI' :(350,250), 'NO1':(120,320),
'NO2':(70,370), 'NO3':(110,250), 'NO4':(170,100), 'NO5':(60,320),
'PL' :(250,530), 'NL' :(50,520), 'GER':(120,550), 'BE' :(50,570),
'LV' :(340,430), 'LT' :(310,480), 'EE' :(340,370) }

def plotMap(capacities):
  f2z = zoneMap2(capacities)
  G = nx.Graph()
  G.add_nodes_from(list(zi.keys()))
  for i,j in f2z:
    G.add_edge(zoneNames[i],zoneNames[j])

  fig,ax = pl.subplots(figsize=(20,12))
  ax.imshow(img)
  nodes = nx.draw_networkx_nodes(G, pos, ax=ax, node_size=800, node_color='w')
  nodes.set_edgecolor('k')
  nx.draw_networkx_edges(G, pos, ax=ax)
  nx.draw_networkx_labels(G, pos, ax=ax)
  pl.show()

def plotFlowMap(x, f2z):
  nFlows = len(f2z)
  nZones = len(x) - nFlows
  G2 = nx.DiGraph()
  G2.add_nodes_from(list(zi.keys()))
  edge_labels = {}
  for f in range(len(f2z)):
    if x[nZones+f] > 0:
      i,j = f2z[f]
    else:
      j,i = f2z[f]
    G2.add_edge(zoneNames[i],zoneNames[j], weight=np.abs(x[nZones+f]))
    edge_labels[(zoneNames[i],zoneNames[j])] = round(np.abs(x[nZones+f]))
  fig,ax = pl.subplots(figsize=(20,15))
  ax.imshow(img)
  cmap = pl.cm.rainbow
  vmin = 0
  vmax = max(x[0:nZones])  #*1.8
  nodes = nx.draw_networkx_nodes(G2, pos, ax=ax, node_size=800, node_color=x[0:nZones],\
              vmin=vmin,vmax=vmax,cmap=cmap)
  nodes.set_edgecolor('k')
  nx.draw_networkx_edges(G2, pos, ax=ax, width=2,arrows=True,arrowsize=20)
  nx.draw_networkx_edge_labels(G2, pos=pos, edge_labels=edge_labels)
  nx.draw_networkx_labels(G2, pos, ax=ax)
  #ax.tick_params(axis="both",which="both",bottom=1,left=1,labelbottom=1,labelleft=1)
  sm = pl.cm.ScalarMappable(cmap=cmap, norm=pl.Normalize(vmin=vmin, vmax=vmax))
  sm._A = []
  pl.colorbar(sm,ax=ax,shrink=0.5,label='Price [€/MWh]')
  pl.show()