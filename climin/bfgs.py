# -*- coding: utf-8 -*-

import itertools

import scipy
import scipy.linalg
import scipy.optimize

from base import Minimizer, repeat_or_iter


class Bfgs(Minimizer):

    def __init__(self, wrt, f, fprime, initial_hessian=None,
                 args=None, stop=1, verbose=False):
        super(Bfgs, self).__init__(wrt, args=args, stop=stop, verbose=verbose)

        self.f = f
        self.fprime = fprime
        self.hessian = initial_hessian

    def __iter__(self):

        def f(x, *args, **kwargs):
            old = self.wrt.copy()
            self.wrt[:] = x
            res = self.f(*args, **kwargs)
            self.wrt[:] = old
            return res

        def fprime(x, *args, **kwargs):
            old = self.wrt.copy()
            self.wrt[:] = x
            res = self.fprime(*args, **kwargs)
            self.wrt[:] = old
            return res

        args, kwargs = self.args.next()
        grad = self.fprime(*args, **kwargs)
        if self.hessian is None:
            self.hessian = scipy.diag(grad**2)
        for i, (args, kwargs) in enumerate(self.args):
            if i > 0 and i % self.stop == 0:
                loss = self.f(*args, **kwargs)
                yield dict(loss=loss, step=step, grad=grad, 
                           direction=direction, steplength=steplength,
                           inv_hessian=inv_hessian)
            inv_hessian = scipy.linalg.inv(self.hessian)
            direction = scipy.dot(inv_hessian, -grad)
            # TODO does not support kwargs, should raise exception
            steplength = scipy.optimize.line_search(
                f, fprime, self.wrt, direction, grad, args=args)[0]
            step = steplength * direction
            self.wrt += step
            grad_m1 = grad
            grad = self.fprime(*args, **kwargs)
            grad_diff = grad - grad_m1

            # Update for Hessian approximation.
            outer_grad_diff = scipy.outer(grad_diff, grad_diff)
            inner_grad_diff_step = scipy.inner(grad_diff, step)

            hessian_times_step = scipy.dot(self.hessian, step)
            hessian_times_step_outer = scipy.outer(
                hessian_times_step, hessian_times_step)
            step_dot_hessian = scipy.dot(step.T, hessian_times_step)

            self.hessian += outer_grad_diff / inner_grad_diff_step
            self.hessian -= hessian_times_step_outer / step_dot_hessian
