# -*- coding: utf-8 -*-

import itertools

import scipy
import numpy as np
import scipy.linalg
import scipy.optimize

from base import Minimizer
from linesearch import WolfeLineSearch


class NonlinearConjugateGradient(Minimizer):
    """
    Nonlinear Conjugate Gradient Method 
    """

    def __init__(self, wrt, f, fprime, epsilon = 1e-6,
                 args=None, logfunc=None):
        super(NonlinearConjugateGradient, self).__init__(
            wrt, args=args, logfunc=logfunc)
        self.f = f
        self.fprime = fprime
        
        self.line_search = WolfeLineSearch(wrt, self.f, self.fprime, c2 = 0.2)
        self.epsilon = epsilon

    def __iter__(self):
        args, kwargs = self.args.next()
        grad = self.fprime(self.wrt, *args, **kwargs)
        grad_m1 = scipy.zeros(grad.shape)
        f_val = self.f(self.wrt, *args, **kwargs)
        f_old = 0
        
        for i, (next_args, next_kwargs) in enumerate(self.args):
            # If the gradient is exactly zero, we stop. Otherwise, the
            # updates will lead to NaN errors because the direction will
            # be zero.
            
            if (grad == 0.0).all():
                self.logfunc({'message': 'converged - residual is null'})
                break

            if i == 0:                  
                direction = -grad
            else:                
                # Computation of beta as a compromise between Fletcher-Reeves 
                # and Polak-Ribiere.
                old_grad_norm = scipy.dot(grad_m1, grad_m1)
                
                betaFR = scipy.dot(grad, grad) / old_grad_norm
                betaPR = scipy.dot(grad, grad - grad_m1) / old_grad_norm
                betaHS = scipy.dot(grad, grad - grad_m1) / scipy.dot(direction, grad - grad_m1)
                beta = max(-betaFR, min(betaPR, betaFR))
                
                # Restart if not a direction of sufficient descent, ie if two
                # consecutive gradients are far from orthogonal.
                if scipy.dot(grad, grad_m1)/old_grad_norm  >  0.1 :
                    beta = 0
                         
                direction = - grad + beta * direction
                
            #line search minimization          
            initialization = min(1, 2 * (f_val - f_old) / scipy.dot(grad, direction))
            alpha = self.line_search.search(
                direction, initialization,  args, kwargs)
            self.wrt += alpha * direction
            if (abs(grad)<self.epsilon).all():
                self.logfunc({'message': 'converged - residual is null'})
                break

            # Prepare everything for the next loop.
            args, kwargs = next_args, next_kwargs
            grad_m1[:], grad[:] = grad, self.fprime(self.wrt, *args, **kwargs)
            f_old, f_val = f_val, self.f(self.wrt, *args, **kwargs)

            loss = self.f(self.wrt, *args, **kwargs)
            info = {
                'loss': loss,
                'steplength': alpha,
                'n_iter': i,
                'args': args,
                'kwargs': kwargs,
            }
            self.logfunc(info)
            yield info
