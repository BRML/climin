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
        loss = self.f(self.wrt, *args, **kwargs)
        loss_m1 = 0

        grad_norm_m1 = float('inf')

        for i, (next_args, next_kwargs) in enumerate(self.args):
            if i == 0:                  
                direction = -grad
            else:                
                # Computation of beta as a compromise between Fletcher-Reeves 
                # and Polak-Ribiere.
                
                grad_diff = grad - grad_m1
                betaFR = scipy.dot(grad, grad) / grad_norm_m1
                betaPR = scipy.dot(grad, grad_diff) / grad_norm_m1 
                betaHS = scipy.dot(grad, grad_diff) / scipy.dot(direction, grad_diff)
                beta = max(-betaFR, min(betaPR, betaFR))
                
                # Restart if not a direction of sufficient descent, ie if two
                # consecutive gradients are far from orthogonal.
                if scipy.dot(grad, grad_m1) / grad_norm_m1 > 0.1 :
                    beta = 0
                         
                direction = - grad + beta * direction
                
            #line search minimization          
            initialization = min(1, 2 * (loss - loss_m1) / scipy.dot(grad, direction))
            step_length = self.line_search.search(
                direction, initialization,  args, kwargs)
            self.wrt += step_length * direction

            # If we don't bail out here, we will enter regions of numerical
            # instability.
            if (abs(grad) < self.epsilon).all():
                self.logfunc(
                    {'message': 'converged - gradient smaller than epsilon'})
                break

            # Prepare everything for the next loop.
            args, kwargs = next_args, next_kwargs
            grad_m1[:], grad[:] = grad, self.fprime(self.wrt, *args, **kwargs)
            loss_m1, loss = loss, self.f(self.wrt, *args, **kwargs)
            grad_norm_m1 = scipy.dot(grad_m1, grad_m1)

            yield {
                'loss': loss,
                'step_length': step_length,
                'n_iter': i,
                'args': args,
                'kwargs': kwargs,
            }
