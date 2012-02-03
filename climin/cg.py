# -*- coding: utf-8 -*-

import itertools

import scipy
import numpy as np
import scipy.linalg
import scipy.optimize

from base import Minimizer


class ConjugateGradient(Minimizer):
    """
    Linear Conjugate Gradient Method for quadratic function :
    f = 1/2 * xTAx -bx
    fprime = Ax - b
    """

    def __init__(self, wrt, f, fprime, epsilon = 1e-14,
                 args=None, stop=1, logfunc=None):
        super(ConjugateGradient, self).__init__(wrt, args=args, logfunc=logfunc)
        self.f = f
        self.fprime = fprime
        self.b = - self.fprime(np.zeros(wrt.size))
        self.epsilon = epsilon

    def __iter__(self):
        args, kwargs = self.args.next()
        grad = self.fprime(self.wrt, *args, **kwargs)
        
        for i, (next_args, next_kwargs) in enumerate(self.args):
            # If the gradient is exactly zero, we stop. Otherwise, the
            # updates will lead to NaN errors because the direction will
            # be zero.
            if (grad == 0).all():
                self.logfunc({'message': 'gradient is 0'})
                break

            if i == 0:
                direction = -grad
            else:
                Ap = self.fprime(direction)+ self.b
                rr = np.dot(grad, grad)
                alpha = rr / np.dot(direction, Ap)
                self.wrt += alpha * direction
                grad = grad + alpha * Ap
                beta = np.dot(grad, grad)/ rr
                direction = - grad + beta * direction
            
            if (abs(grad) < self.epsilon).all():
                self.logfunc(
                    {'message': 'converged - residual smaller than epsilon'})
                break

            # Prepare everything for the next loop.
            args, kwargs = next_args, next_kwargs

            if i > 0 and i % self.stop == 0:
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
