'''
Functions for the power grid price balance examples.
'''

from scipy.optimize import fsolve, minimize
from scipy.special import expit
import numpy as np
import matplotlib.pyplot as pl

pl.rcParams['axes.labelsize'] = 12
pl.rcParams['xtick.labelsize'] = 12
pl.rcParams['ytick.labelsize'] = 12

def zoneMap(nZones):
  # Zone connection map
  f2z = []
  for i in range(nZones):
    for j in range(i+1,nZones):
      f2z.append((i,j)) # positive: from i to j (sink in i, source in j)
  return f2z

def zoneMap2(capacities):
  # Zone connection map, ignore 0 capacities
  nZones = np.size(capacities,0)
  f2z = []
  for i in range(nZones):
    for j in range(i+1,nZones):
      if capacities[i,j] > 0:
        f2z.append((i,j)) # positive, from i to j (sink in i, source in j)
  return f2z

def generateBalanceFunction(producers, consumers, capacities, rate1, rate2):
  
  nZones = len(producers)
  f2z    = zoneMap2(capacities)
  nFlows = len(f2z)
  
  def fnVector(x):
    funVals = np.zeros(nZones+nFlows)
    
    # Zone balances (dq/dt)
    for i in range(nZones):      
      funVals[i] = producers[i](x[i]) - consumers[i](x[i])
    for f in range(nFlows):
      z0,z1 = f2z[f]
      funVals[z0] -= x[nZones + f]
      funVals[z1] += x[nZones + f]
    
    # Flow to balance prices (df/dt)
    for f in range(nFlows):
      i,j = f2z[f]
      fi = nZones + f
      lo = - capacities[j,i]
      hi =   capacities[i,j]
      tol = 0.2*(hi-lo)
      if lo < hi:
        f    = x[fi]
        v2  = rate2 * (x[j] - x[i])
        v1  = max(-rate1 * (f-lo) , v2)
        v3  = min(-rate1 * (f-hi) , v2)
        if f < lo:
          dfdt = v1
        elif f < lo+tol:
          r = (f-lo) / tol
          dfdt = r*v2 + (1-r)*v1
        elif f < hi-tol:
          dfdt = v2
        elif f < hi:
          r = (f-hi+tol) / tol
          dfdt = r*v3 + (1-r)*v2
        else:
          dfdt = v3
      else:
        dfdt = -rate1 * x[fi]
      funVals[fi] = dfdt
    
    return funVals
  
  
  # Merge to one objective function to be minimized
  fnScalar = lambda x : np.sum(fnVector(x)**4)
  
  return fnVector, f2z, fnScalar


def solvePriceBalance(producers,consumers,capacities=None,printInfo=None,tol=None):
  
  nZones = len(producers)
  if len(consumers) != nZones:
    raise Exception('Producers and consumers should have the same length')
  if type(capacities) != np.ndarray:
    capacities=np.zeros(nZones)
  else:
    if np.size(capacities,0) != nZones | np.size(capacities,1) != nZones:
      raise Exception('Capacities should be nZones x nZones')
  
  ref = max([c(0) for c in consumers])
  rate1 = 1
  rate2 = ref/20 # ref load/price 1000/20
  vectorFunction, f2z, scalarFunction = \
    generateBalanceFunction(producers,consumers,capacities,rate1,rate2)
  nVar = nZones + len(f2z)
  
  # Solve without circle-flow prevention
  x = np.zeros(nVar)
  #fn = lambda x : [np.sign(v)*v**4 for v in vectorFunction(x)]
  x,info,ier,msg = fsolve(vectorFunction, x, xtol=1e-7, full_output=1)
  if printInfo:
    print('First solution: ', info['nfev'], msg)
  
  # Add the sum of flows to the balance to avoid circle flows, and reduce the term iteratively
  #x = np.zeros(nVar)
  if nZones > 2:
    for factor in [1e-4, 1e-6, 1e-10]: #[1e0, 1, 1e-2, 1e-6, 1e-8]:
      fn = lambda x : scalarFunction(x) + factor/ref*np.abs(np.sum(x[nZones:nVar]))
      sol = minimize(fn, x, tol=tol, method='powell')
      x=sol.x
      if printInfo:
        #print(sol)
        print(sol.success, ' ', sol.nit, ' ', fn(x))
  return x, f2z


def step(x,x0,y1,width=0.4):
  return y1 * (expit(10*(x-x0)/width)) + 0.01*x

