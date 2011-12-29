# -*- coding: utf-8 -*-

import itertools

import scipy
import scipy.linalg
import scipy.optimize

from base import Minimizer

# TODO:
#
# - use own line search


class Bfgs(Minimizer):

    def __init__(self, wrt, f, fprime, initial_inv_hessian=None,
                 args=None, stop=1, verbose=False):
        super(Bfgs, self).__init__(wrt, args=args, stop=stop, verbose=verbose)

        self.f = f
        self.fprime = fprime
        self.inv_hessian = initial_inv_hessian

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

        if self.inv_hessian is None:
            self.inv_hessian = scipy.eye(grad.shape[0])

        for i, (args, kwargs) in enumerate(self.args):
            if i > 0 and i % self.stop == 0:
                loss = self.f(*args, **kwargs)
                yield dict(loss=loss, step=step, grad=grad, 
                           direction=direction, steplength=steplength,
                           inv_hessian=self.inv_hessian)

            # If the gradient is exactly zero, we stop. Otherwise, the
            # updates will lead to NaN errors because the direction will
            # be zero.
            if (grad == 0.0).all():
                break

            direction = scipy.dot(self.inv_hessian, -grad)

            # TODO does not support kwargs, should raise exception
            steplength = scipy.optimize.line_search(
                f, fprime, self.wrt, direction, grad, args=args)[0]
            step = steplength * direction
            self.wrt += step
            grad_m1 = grad
            grad = self.fprime(*args, **kwargs)

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
