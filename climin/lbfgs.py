# -*- coding: utf-8 -*-

import itertools

import scipy
import scipy.linalg
import scipy.optimize

from base import Minimizer
from linesearch import exponential_distant


class Lbfgs(Minimizer):

    def __init__(self, wrt, f, fprime, initial_hessian_diag=1,
                 n_factors=10, line_search=None,
                 args=None, stop=1, verbose=False):
        super(Lbfgs, self).__init__(wrt, args=args, stop=stop, verbose=verbose)

        self.f = f
        self.fprime = fprime
        self.initial_hessian_diag = initial_hessian_diag
        self.n_factors = 10
        if line_search is not None:
            self.line_search = line_search
        else:
            self.line_search = exponential_distant

    def f_with_x(self, x, *args, **kwargs):
        old = self.wrt.copy()
        self.wrt[:] = x
        res = self.f(*args, **kwargs)
        self.wrt[:] = old
        return res

    def fprime_with_x(self, x, *args, **kwargs):
        old = self.wrt.copy()
        self.wrt[:] = x
        res = self.fprime(*args, **kwargs)
        self.wrt[:] = old
        return res

    def inv_hessian_dot_gradient(self, grad_diffs, steps, grad, idxs):
        grad = grad.copy()  # We will change this.
        n_current_factors = len(idxs)

        # TODO: find a good name for this variable.
        rho = scipy.empty(n_current_factors)

        # TODO: vectorize this function
        for i in idxs:
            rho[i] = scipy.inner(grad_diffs[i], steps[i])

        # TODO: find a good name for this variable as well.
        alpha = scipy.empty(n_current_factors)

        for i in idxs:
            alpha[i] = rho[i] * scipy.inner(steps[i], grad)
            grad -= alpha[i] * grad_diffs[i]
        z = self.initial_hessian_diag * grad

        # TODO: find a good name for this variable (surprise!)
        beta = scipy.empty(n_current_factors)

        for i in idxs:
            beta[i] = rho[i] * scipy.inner(grad_diffs[i], z)
            z += steps[i] * (alpha[i] - beta[i])

        return z

    def __iter__(self):
        args, kwargs = self.args.next()
        grad = self.fprime(*args, **kwargs)
        factor_shape = self.n_factors, self.wrt.shape[0]
        grad_diffs = scipy.zeros(factor_shape)
        steps = scipy.zeros(factor_shape)

        # We need to keep track in which order the different statistics
        # from different runs are saved. 
        #
        # Why?
        #
        # Each iteration, we save statistics such as the difference between
        # gradients and the actual steps taken. This are then later combined
        # into an approximation of the Hessian. We call them factors. Since we
        # don't want to create a new matrix of factors each iteration, we
        # instead keep track externally, which row of the matrix corresponds
        # to which iteration. `idxs` now is a list which maps its i'th element
        # to the corresponding index for the array. Thus, idx[i] contains the
        # rowindex of the for the (n_factors - i)'th iteration prior to the
        # current one.
        idxs = []

        for i, (args, kwargs) in enumerate(self.args):
            if i > 0 and i % self.stop == 0:
                loss = self.f(*args, **kwargs)
                yield dict(loss=loss)

            # If the gradient is exactly zero, we stop. Otherwise, the
            # updates will lead to NaN errors because the direction will
            # be zero.
            if (grad == 0.0).all():
                break

            if i == 1:
                direction = -grad
            else:
                direction = -self.inv_hessian_dot_gradient(
                    grad_diffs, steps, grad, idxs)

            steplength = self.line_search(
                self.f_with_x, self.wrt, direction, args=args, kwargs=kwargs)
            step = steplength * direction
            self.wrt += step
            grad = self.fprime(*args, **kwargs)
            grad_m1 = grad

            # Determine index for the current update. 
            if not idxs:
                # First iteration.
                this_idx = 0
            elif len(idxs) < self.n_factors:
                # We are not "full" yet. Thus, append the next idxs.
                this_idx = idxs[-1] + 1
            else:
                # we are full and discard the first index.
                this_idx = idxs.pop(0)

            idxs.append(this_idx)
            grad_diffs[this_idx] = grad - grad_m1
            steps[this_idx] = step
