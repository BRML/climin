# -*- coding: utf-8 -*-

import itertools

import scipy
import numpy as np
import scipy.linalg

from base import Minimizer, is_nonzerofinite
from linesearch import WolfeLineSearch


class Bfgs(Minimizer):

    def __init__(self, wrt, f, fprime, initial_inv_hessian=None,
                 line_search=None, args=None, logfunc=None):
        super(Bfgs, self).__init__(wrt, args=args, logfunc=logfunc)
        self.f = f
        self.fprime = fprime
        self.inv_hessian = initial_inv_hessian

        if line_search is not None:
            self.line_search = line_search
        else:
            self.line_search = WolfeLineSearch(wrt, self.f, self.fprime)

    def find_direction(self, grad_m1, grad, step, inv_hessian):
        H = self.inv_hessian
        grad_diff = grad - grad_m1
        ys = np.inner(grad_diff, step)
        ss = np.inner(step, step)
        yy = np.inner(grad_diff, grad_diff)
        Hy = np.dot(H, grad_diff)
        yHy = np.inner(grad_diff, Hy)
        H += (ys + yHy) * np.outer(step, step) / ys**2 
        H -= (np.outer(Hy, step) + np.outer(step, Hy)) / ys
        direction = -np.dot(H, grad)
        return direction, {'gradient_diff': grad_diff}

    def __iter__(self):
        args, kwargs = self.args.next()
        grad = self.fprime(self.wrt, *args, **kwargs)
        grad_m1 = scipy.zeros(grad.shape)

        if self.inv_hessian is None:
            self.inv_hessian = scipy.eye(grad.shape[0])

        for i, (next_args, next_kwargs) in enumerate(self.args):
            if i == 0:
                direction, info = -grad, {}
            else:
                direction, info = self.find_direction(
                    grad_m1, grad, step, self.inv_hessian)

            if not is_nonzerofinite(direction):
                self.logfunc(
                    {'message': 'direction is invalid -- need to bail out.'})
                break

            step_length = self.line_search.search(
                direction, None, args, kwargs)

            if step_length != 0:
                step = step_length * direction
                self.wrt += step
            else:
                self.logfunc(
                    {'message': 'step length is 0--need to bail out.'})
                break

            # Prepare everything for the next loop.
            args, kwargs = next_args, next_kwargs
            # TODO: not all line searches have .grad!
            grad_m1[:], grad[:] = grad, self.line_search.grad

            info.update({
                'step_length': step_length,
                'n_iter': i,
                'args': args,
                'kwargs': kwargs,
            })
            yield info
