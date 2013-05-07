# -*- coding: utf-8 -*-

import numpy as np
import scipy
import math
import scipy.linalg
import scipy.optimize
from scipy.sparse.linalg import eigsh

from base import Minimizer, is_nonzerofinite
from hf import HessianFree
from linesearch import WolfeLineSearch, BackTrack

from scipy.sparse.linalg import eigsh

###TODO: not copy code from LBFGS ###

class NaturalNewtonLR(Minimizer):
    """
    Unstable
    """
    def __init__(self, wrt, f, fprime, blocksizes,
                 initial_hessian_diag=1,
                 n_factors=10, line_search=None,
                 args=None, 
                 countC=8, N=1,
                 gamma=0.995, BC=2,
                 b=50, k=5,
                 logfunc=None):
        super(NaturalNewtonLR, self).__init__(wrt, args=args, logfunc=logfunc)

        self.f = f
        self.fprime = fprime
        self.blocksizes = blocksizes
        self.countC = countC
        self.gamma = gamma
        self.gamma_sqrt = math.sqrt(self.gamma)
        self.gamma_sqrt_m1 = math.sqrt(1-self.gamma)
        self.BC = BC
        self.N = N
        self.b = b #TODO find a meaningful name
        self.k = k

        
        #for LBFGS#
        self.initial_hessian_diag = initial_hessian_diag
        self.n_factors = n_factors
        if line_search is not None:
            self.line_search = line_search
        else:
            self.line_search = WolfeLineSearch(wrt, self.f, self.fprime)
        ##################################################################



    ### LBFGS ###
    def find_direction(self, grad_diffs, steps, grad, hessian_diag, idxs):
        grad = grad.copy()  # We will change this.
        n_current_factors = len(idxs)

        # TODO: find a good name for this variable.
        rho = scipy.empty(n_current_factors)

        # TODO: vectorize this function
        for i in idxs:
            rho[i] = 1 / scipy.inner(grad_diffs[i], steps[i])

        # TODO: find a good name for this variable as well.
        alpha = scipy.empty(n_current_factors)

        for i in idxs[::-1]:
            alpha[i] = rho[i] * scipy.inner(steps[i], grad)
            grad -= alpha[i] * grad_diffs[i]
        z = hessian_diag * grad

        # TODO: find a good name for this variable (surprise!)
        beta = scipy.empty(n_current_factors)

        for i in idxs:
            beta[i] = rho[i] * scipy.inner(grad_diffs[i], z)
            z += steps[i] * (alpha[i] - beta[i])

        return z, {}
    ##################################################################
    
    def __iter__(self):
        args, kwargs = self.args.next()
        grad = self.fprime(self.wrt, *args, **kwargs)
        mu = grad
        mu_norm = (mu**2).sum()
        G_m1 = np.zeros((mu.size, 1))
        

        ### LBFGS ###
        grad_m1 = scipy.zeros(grad.shape)
        factor_shape = self.n_factors, self.wrt.shape[0]
        grad_diffs = scipy.zeros(factor_shape)
        steps = scipy.zeros(factor_shape)
        hessian_diag = self.initial_hessian_diag
        step_length = None
        idxs = []
        ##################################################################

        for i, (next_args, next_kwargs) in enumerate(self.args):
        ### LBFGS ###
            if i == 0:
                direction = -grad
                info = {}
            else:
                sTgd = scipy.inner(step, grad_diff)
                if sTgd > 1E-10:
                    # Don't do an update if this value is too small.
                    # Determine index for the current update. 
                    if not idxs:
                        # First iteration.
                        this_idx = 0
                    elif len(idxs) < self.n_factors:
                        # We are not "full" yet. Thus, append the next idxs.
                        this_idx = idxs[-1] + 1
                    else:
                        # we are full and discard the first index.
                        this_idx = idxs.pop(0)

                    idxs.append(this_idx)
                    grad_diffs[this_idx] = grad_diff
                    steps[this_idx] = step
                    hessian_diag = sTgd / scipy.inner(grad_diff, grad_diff)

                direction, info = self.find_direction(
                    grad_diffs, steps, -grad, hessian_diag, idxs) 

            if not is_nonzerofinite(direction):
                self.logfunc(
                    {'message': 'direction is invalid--need to bail out.'})
                break
            ##################################################################

            
            # Update parameters.
            # Similar to tonga, except that the direction returned by lbfgs  (ie the gradient times the inverse of the Hessian) is used instead of the gradient
            # mu is the running estimate of the mean of Newton directions

            
            if (i%self.b == 0) and (i>0):
                G = np.empty((mu.size, self.k))
                offset = 0
                #block diagonal approximation of eigenvectors 
                for size in self.blocksizes:
                    g = G_m1[offset:offset+size]
                    lamb, V =  eigsh(scipy.dot(g, g.T), k=self.k, ncv=(2*self.k+2))
                    G[offset:offset+size] = np.sqrt(scipy.minimum(self.BC, lamb/(self.N * mu_norm)))*V
                    ## lamb, V = eigsh(scipy.dot(g.T, g), k=self.k, ncv=(2*self.k+2))
                    ## U[offset:offset+size] = np.sqrt(minimum(self.BC, lamb/(self.N * mu_norm))) * scipy.dot(g, V*np.sqrt(lamb))
                    offset += size
                y = np.zeros(self.k)
                y[-1] = 1                
                size = self.k
            else :
                size = G_m1[0].size+1
                G = np.empty((mu.size, size))
                G[:,:-1] = self.gamma_sqrt * G_m1[:,:]
                G[:,-1] = self.gamma_sqrt * self.gamma_sqrt_m1 * (direction - mu)
                y = np.zeros(size)
                y[-1] = self.gamma_sqrt * self.gamma_sqrt_m1

            step = scipy.dot(scipy.dot(G, scipy.linalg.inv(np.eye(size) + scipy.dot(G.T,G)/(self.N * mu_norm))), (y - scipy.dot(G.T, mu) / (self.N * mu_norm))) + mu
            
            mu = self.gamma * mu + (1-self.gamma)*direction
            mu_norm = (mu**2).sum()                                                                                                     
            self.wrt += step
            



            loss = self.f(self.wrt, *next_args, **next_kwargs)     
            info = {
                'n_iter': i,
                'loss': loss,
                'direction':direction,
                'gradient': grad,
                'mu':mu,
                'G':G,
                'step': step,
                'args': args,
                'kwargs': kwargs}
            yield info

            # Prepare for next loop.
            args, kwargs = next_args, next_kwargs
            grad_m1[:], grad[:] = grad, self.fprime(self.wrt, *args, **kwargs)
            grad_diff = grad - grad_m1
            G_m1 = G
