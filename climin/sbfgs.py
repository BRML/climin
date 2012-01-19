# -*- coding: utf-8 -*-

import itertools

import scipy as sp
import numpy as np
import scipy.linalg
import scipy.optimize

from base import Minimizer
from linesearch import WolfeLineSearch
from logging import taggify


class SBfgs(Minimizer):

    def __init__(self, wrt, f, fprime, initial_inv_hessian=None,
                 line_search=None,
                 args=None, stop=1, logger=None):
        super(SBfgs, self).__init__(wrt, args=args, logger=logger)

        self.f = f
        self.fprime = fprime
        self.inv_hessian = initial_inv_hessian
        
        if line_search is not None:
            self.line_search = line_search
        else:
            self.line_search = WolfeLineSearch(wrt, self.f, self.fprime, typ=4)
        self.line_search.logger = taggify(self.logger, 'linesearch')

    def __iter__(self):
        args, kwargs = self.args.next()
        grad = self.fprime(self.wrt, *args, **kwargs)
        grad_m1 = scipy.zeros(grad.shape)

        if self.inv_hessian is None:
            self.inv_hessian = scipy.eye(grad.shape[0])

        for i, (next_args, next_kwargs) in enumerate(self.args):
            # If the gradient is exactly zero, we stop. Otherwise, the
            # updates will lead to NaN errors because the direction will
            # be zero.
            if (grad == 0.0).all():
                self.logger.send({'message': 'converged - gradient is 0'})
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
                    if self.inv_hessian is None:
                        self.inv_hessian = H = np.eye(grad.size)*ys/yy
                    else:
                        H = self.inv_hessian

                Hy = np.dot(H, grad_diff)
                yHy = np.inner(grad_diff, Hy)
                gamma = ys/yHy
                v = scipy.sqrt(yHy) * (step/ys - Hy/yHy)
                v = scipy.real(v)
                H[:] = gamma * (H - np.outer(Hy, Hy) / yHy + np.outer(v, v))
                H += np.outer(step, step) / ys
                direction = - np.dot(H, grad)

            steplength = self.line_search.search(direction, args, kwargs)
            if steplength == 0:
                self.logger.send({'message': 'converged - steplength is 0'})
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
                self.logger.send(info)
                yield info
