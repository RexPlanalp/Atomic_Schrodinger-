import numpy as np
from scipy.special import roots_legendre

from petsc4py import PETSc
comm = PETSc.COMM_WORLD
rank = comm.rank

import time


class basis:
    def __init__(self,TISEInstance):
        self._createKnots(TISEInstance)

    def _linearKnots(self,TISEInstance):
        n_basis = TISEInstance["splines"]["n_basis"]
        order = TISEInstance["splines"]["order"]
        rmax = TISEInstance["box"]["grid_size"]


        N_knots = n_basis + order
        N_middle = N_knots - 2 * (order - 2)
        knots_middle = np.linspace(0, rmax, N_middle)
        knots_middle[-1] = rmax  

        knots_start = [0] * (order - 2)
        knots_end = [rmax] * (order - 2)

        return np.array(knots_start + knots_middle.tolist() + knots_end)

    def _quadraticKnots(self,TISEInstance):
        n_basis = TISEInstance["splines"]["n_basis"]
        order = TISEInstance["splines"]["order"]
        rmax = TISEInstance["box"]["grid_size"]

        N_knots = n_basis + order 
        N_middle = N_knots - 2 * (order-2)
        
        indices = np.linspace(0, 1, N_middle)
        knots_middle = rmax * indices**1.2
        
        knots_start = [0] * (order-2)
        knots_end = [rmax] * (order-2)
        
        return np.array(knots_start + knots_middle.tolist() + knots_end)

    def _piecewisequadpluslinKnots(self,TISEInstance):
        n_basis = TISEInstance["splines"]["n_basis"]
        order = TISEInstance["splines"]["order"]
        rmax = TISEInstance["box"]["grid_size"]

        N_knots = n_basis + order 
        N_middle = N_knots - 2 * (order-2)
        

        N_core = 100
        N_outer = N_middle - N_core

        core = 200

        core_knots = core*np.linspace(0,1,N_core)**2
        outer_knots = core + np.linspace(0,1,N_outer)*(rmax-core)

        knots_middle = core_knots.tolist() + outer_knots.tolist()

        knots_start = [0] * (order-2)
        knots_end = [rmax] * (order-2)

        return np.array(knots_start + knots_middle + knots_end)
    
    def _linlinKnots(self,TISEInstance):
        n_basis = TISEInstance["splines"]["n_basis"]
        order = TISEInstance["splines"]["order"]
        rmax = TISEInstance["box"]["grid_size"]


        N_knots = n_basis + order 
        N_middle = N_knots - 2 * (order-2)
            
        N_core = 700
        N_outer = N_middle - N_core

        core = 150

        core_knots = core*np.linspace(0,1,N_core)
        outer_knots = core + np.linspace(0,1,N_outer)*(rmax-core)

        knots_middle = core_knots.tolist() + outer_knots.tolist()

        knots_start = [0] * (order-2)
        knots_end = [rmax] * (order-2)

        return np.array(knots_start + knots_middle + knots_end)
    
    def _createKnots(self,TISEInstance):
        R0_input = TISEInstance["box"]["R0"]
        
        
        if TISEInstance["splines"]["knot_spacing"] == "linear":
            self.knots = self._linearKnots(TISEInstance)
        elif TISEInstance["splines"]["knot_spacing"] == "quadratic":
            self.knots = self._quadraticKnots(TISEInstance)
        elif TISEInstance["splines"]["knot_spacing"] == "quadratic+linear":
            self.knots = self._piecewisequadpluslinKnots(TISEInstance)
        elif TISEInstance["splines"]["knot_spacing"] == "linear+linear":
            self.knots = self._linlinKnots(TISEInstance)
        
        self.R0_index = np.argmin(np.abs(self.knots-R0_input))
        self.R0 = self.knots[self.R0_index]

    
        self.eta = TISEInstance["box"]["eta"]
        
    def integrate(self, func, i_fixed, j_fixed,order,knots):
        start = min(i_fixed, j_fixed)
        end = max(i_fixed + order, j_fixed + order)

        a = knots[start:end]
        b = knots[start + 1:end + 1]

        x, w = roots_legendre(8) 
        
        x = x.reshape(1, -1)
        y = np.outer(b - a, (x + 1) / 2) + a[:, np.newaxis]
        
        
        w = w.reshape(1, -1)
      
        knots = knots[:, np.newaxis] 
        
        if self.R0<np.max(knots):
            w = np.where(y<=self.R0,w,w*np.exp(1j * np.pi * self.eta))
            y = self.R(y)
            knots = self.R(knots)
        
        func_eval = func(x=y, i=i_fixed, j=j_fixed, knots=knots, order=order)
        integral_contributions = np.sum(w * func_eval, axis=1) * (b - a) / 2
        total = np.sum(integral_contributions)
        return total
  
    def B(self,i, k, x, knots):
        x = np.asarray(x)
        if k == 1:
            return np.where((knots[i] <= x) & (x < knots[i + 1]), 1.0, 0.0)
        else:
            denom1 = knots[i + k - 1] - knots[i]
            term1 = 0.0
            if denom1 > 0:
                term1 = (x - knots[i]) / denom1 * self.B(i, k - 1, x, knots)

            denom2 = knots[i + k] - knots[i + 1]
            term2 = 0.0
            if denom2 > 0:
                term2 = (knots[i + k] - x) / denom2 * self.B(i + 1, k - 1, x, knots)

        return term1 + term2

    def dB(self, i, k, x, knots):
        if k == 1:
            return np.zeros_like(x)
        else:
            denom1 = knots[i + k - 1] - knots[i]
            term1 = 0.0
            if denom1 > 0:
                term1 = 1.0 / denom1 * self.B(i, k - 1, x, knots) + (x - knots[i]) / denom1 * self.dB(i, k - 1, x, knots)

            denom2 = knots[i + k] - knots[i + 1]
            term2 = 0.0
            if denom2 > 0:
                term2 = -1.0 / denom2 * self.B(i + 1, k - 1, x, knots) + (knots[i + k] - x) / denom2 * self.dB(i + 1, k - 1, x, knots)

        return term1 + term2

    def R(self,x):
        return np.where(x < self.R0, x, self.R0 + (x - self.R0) * np.exp(1j * np.pi * self.eta))
    
    
