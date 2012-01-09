# -*- coding: utf-8 -*-

import itertools

import scipy
import scipy.linalg
import scipy.optimize

from base import Minimizer
from linesearch import WolfeLineSearch


class Bfgs(Minimizer):

    def __init__(self, wrt, f, fprime, initial_inv_hessian=None,
                 line_search=None,
                 args=None, stop=1, verbose=False):
        super(Bfgs, self).__init__(wrt, args=args, verbose=verbose)

        self.f = f
        self.fprime = fprime
        self.inv_hessian = initial_inv_hessian
        
        if line_search is not None:
            self.line_search = line_search
        else:
            self.line_search = WolfeLineSearch(wrt, self.f, self.fprime)

    def __iter__(self):
        args, kwargs = self.args.next()
        grad = self.fprime(self.wrt, *args, **kwargs)

        if self.inv_hessian is None:
            self.inv_hessian = scipy.eye(grad.shape[0])

        for i, (args, kwargs) in enumerate(self.args):
            if i > 0 and i % self.stop == 0:
                loss = self.f(self.wrt, *args, **kwargs)
                yield dict(loss=loss, step=step, grad=grad, 
                           direction=direction, steplength=steplength,
                           inv_hessian=self.inv_hessian)

            # If the gradient is exactly zero, we stop. Otherwise, the
            # updates will lead to NaN errors because the direction will
            # be zero.
            if (grad == 0.0).all():
                if self.verbose:
                    print 'gradient is 0'
                break

            direction = scipy.dot(self.inv_hessian, -grad)
            direction /= abs(direction).max()
            steplength = self.line_search.search(direction, args, kwargs)

            if steplength == 0:
                if self.verbose:
                    print 'steplength is 0'
                break

            step = steplength * direction
            self.wrt += step
            grad_m1 = grad
            grad = self.fprime(self.wrt, *args, **kwargs)

            # Update for inverse Hessian approximation.
            # We will do some abbreviations here to keep the code short.
            grad_diff = grad - grad_m1
            s = step
            y = grad_diff
            B = self.inv_hessian

            sTy = scipy.inner(s, y)
            yTBy = scipy.inner(y, scipy.dot(B, y))
            ssT = scipy.outer(s, s)

            ysT = scipy.outer(y, s)
            BysT = scipy.dot(B, ysT)
            syB = scipy.dot(ysT, B)

            self.inv_hessian += (sTy + yTBy) * ssT / sTy**2
            self.inv_hessian -= (BysT + syB) / sTy
