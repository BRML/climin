# -*- coding: utf-8 -*-

import numpy as np
import scipy
import scipy.linalg
import scipy.optimize
from scipy.sparse.linalg import eigsh

from base import Minimizer, is_nonzerofinite
from hf import HessianFree
from linesearch import WolfeLineSearch, BackTrack

###TODO: not copy code from LBFGS ###
class NaturalNewton(Minimizer):
    """
    Working version, but not performing better than lbfgs
    Diagonal approximation of the covariance matrix
    No search on the hyperparameters space was performed
    """

    def __init__(self, wrt, f, fprime,
                 initial_hessian_diag=1,
                 n_factors=10, line_search=None,
                 args=None, 
                 countC=8, N=1,
                 gamma=0.995, BC=2,
                 
                 logfunc=None):
        super(NaturalNewton, self).__init__(wrt, args=args, logfunc=logfunc)

        self.f = f
        self.fprime = fprime
        self.countC = countC
        self.gamma = gamma
        self.BC = BC
        self.N = N

        #for LBFGS#
        self.initial_hessian_diag = initial_hessian_diag
        self.n_factors = n_factors
        if line_search is not None:
            self.line_search = line_search
        else:
            self.line_search = WolfeLineSearch(wrt, self.f, self.fprime)
        ##################################################################



    ### unused functions for now, potentially helpful for low-rank approximation###
    def min_(self, B, M):
         V , U  = eigsh(M, k=self.k, ncv=(2*self.k+2))
         return scipy.dot(scipy.minimum(V,B)*(U.T), U)
         
         ## ret = [min(V[i], B) * scipy.outer(U[i], U[i]) for i in range(self.k)]
         ## return np.array(ret).sum(axis=0)
     
    def invA (self, u, v, X, fA_m1):
        Au = fA_m1(u)
        return (fA_m1(X)-scipy.dot(fA_m1(v), X)/(1+scipy.dot(v,Au)) * Au)
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
        cov = np.zeros(mu.size)
        D = np.ones(mu.size)

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
            step = D * direction
            self.wrt += step
            


            if (i%self.countC == 0) and (i>0):                    
                cov = self.gamma * cov + self.gamma*(1-self.gamma)*(direction - mu)**2
                mu = self.gamma*mu + (1-self.gamma)*direction
                mu_norm = (mu**2).sum()
                D = 1/(1 + scipy.minimum(self.BC, cov/(self.N * mu_norm)))
                #D = scipy.linalg.inv(np.eye(mu.len())+ self.min_(self.BC, cov)/(self.N*(mu**2).sum()))
                ## fact = self.N*(mu**2).sum()
                ## lamb, V  = eigsh(cov, k=self.k, ncv=(2*self.k+2))
                ## U = scipy.minimum(lamb/fact, B)*(V.T)
                ## D = lambda w: w
                ## for j in range(self.k):
                    ##     D = lambda X: self.invA(u[j], v[j], X, D)
               

            loss = self.f(self.wrt, *next_args, **next_kwargs)     
            info = {
                'n_iter': i,
                'loss': loss,
                'direction':direction,
                'gradient': grad,
                'mu':mu,
                'cov':cov,
                'step': step,
                'args': args,
                'kwargs': kwargs}
            yield info

            # Prepare for next loop.
            args, kwargs = next_args, next_kwargs
            grad_m1[:], grad[:] = grad, self.fprime(self.wrt, *args, **kwargs)
            grad_diff = grad - grad_m1
