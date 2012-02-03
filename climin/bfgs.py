# -*- coding: utf-8 -*-

import itertools

import scipy
import numpy as np
import scipy.linalg
import scipy.optimize

from base import Minimizer
from linesearch import WolfeLineSearch


class Bfgs(Minimizer):

    def __init__(self, wrt, f, fprime, initial_inv_hessian=None,
                 line_search=None, args=None, stop=1, logfunc=None):
        super(Bfgs, self).__init__(wrt, args=args, logfunc=logfunc)
        self.f = f
        self.fprime = fprime
        if initial_inv_hessian is not None:
            self.inv_hessian = initial_inv_hessian
        else:
            self.inv_hessian = np.eye(wrt.size)

        if line_search is not None:
            self.line_search = line_search
        else:
            self.line_search = WolfeLineSearch(wrt, self.f, self.fprime)

    def __iter__(self):
        args, kwargs = self.args.next()
        grad = self.fprime(self.wrt, *args, **kwargs)
        grad_m1 = scipy.zeros(grad.shape)

        # TODO: Following lines should be cleaned up.
        if self.inv_hessian is None:
            self.inv_hessian = scipy.eye(grad.shape[0])

        for i, (next_args, next_kwargs) in enumerate(self.args):
            # If the gradient is exactly zero, we stop. Otherwise, the
            # updates will lead to NaN errors because the direction will
            # be zero.
            if (grad == 0.0).all():
                self.logfunc({'message': 'gradient is 0'})
                break

            if i == 0:
                direction = -grad
            else:
                H = self.inv_hessian
                grad_diff = grad - grad_m1
                ys = np.inner(grad_diff, step)
                ss = np.inner(step, step)
                yy = np.inner(grad_diff, grad_diff)
                Hy = np.dot(H, grad_diff)
                yHy = np.inner(grad_diff, Hy)
                H += (ys + yHy)*np.outer(step, step)/ys**2 - (np.outer(Hy, step) + np.outer(step, Hy))/ys
                direction = -np.dot(H, grad)

            steplength = self.line_search.search(direction, None, args, kwargs)
            if steplength == 0:
                self.logfunc({'message': 'converged - steplength is 0'})
                break
            step = steplength * direction
            self.wrt += step

            # Prepare everything for the next loop.
            args, kwargs = next_args, next_kwargs
            # TODO: not all line searches have .grad!
            grad_m1[:], grad[:] = grad, self.line_search.grad

            if i > 0 and i % self.stop == 0:
                loss = self.f(self.wrt, *args, **kwargs)
                info = {
                    'loss': loss,
                    'steplength': steplength,
                    'n_iter': i,
                    'args': args,
                    'kwargs': kwargs,
                }
                self.logfunc(info)
                yield info

