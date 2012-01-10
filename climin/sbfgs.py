# -*- coding: utf-8 -*-

import itertools

import scipy as sp
import numpy as np
import scipy.linalg
import scipy.optimize

from base import Minimizer
from linesearch import WolfeLineSearch


class SBfgs(Minimizer):

    def __init__(self, wrt, f, fprime, initial_inv_hessian=None,
                 line_search=None,
                 args=None, stop=1, verbose=False):
        super(SBfgs, self).__init__(wrt, args=args, verbose=verbose)

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
        grad_m1 = scipy.zeros(grad.shape)

        if self.inv_hessian is None:
            self.inv_hessian = scipy.eye(grad.shape[0])

        for i, (next_args, next_kwargs) in enumerate(self.args):
            if i > 0 and i % self.stop == 0:
                loss = self.f(self.wrt, *args, **kwargs)
                yield dict(loss=loss)

            if (grad == 0.0).all():
                if self.verbose:
                    print 'gradient is 0'
                break

            if i == 0:
                direction = -grad
            else:
                grad_diff = grad - grad_m1
                ys = np.inner(grad_diff, step)
                ss = np.inner(step, step)
                yy = np.inner(grad_diff, grad_diff)
                if i == 1:
                    # Make initial Hessian approximation
                    # via scaled identity 
                    H = np.eye(grad.size)*ys/yy
                #
                Hy = np.dot(H, grad_diff)
                yHy = np.inner(grad_diff, Hy)
                gamma = ys/yHy
                v = scipy.sqrt(yHy) * (step/ys - Hy/yHy)
                # update H inplace
                H += np.outer(v, v) - np.outer(Hy, Hy)/yHy 
                H *= gamma
                H += np.outer(step, step)/ys
                direction = - np.dot(H, grad)
            steplength = self.line_search.search(direction, args, kwargs)
            step = steplength * direction
            self.wrt += step

            # Prepare everything for the next loop.
            args, kwargs = next_args, next_kwargs
            # TODO: not all line searches have .grad!
            grad_m1[:], grad[:] = grad, self.line_search.grad

