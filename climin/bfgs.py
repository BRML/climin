# -*- coding: utf-8 -*-

# TODO: document module
# TODO: check if gnumpy compatible

import warnings

import scipy
import numpy as np
import scipy.linalg
import scipy.optimize

from base import Minimizer, is_nonzerofinite
from linesearch import WolfeLineSearch


class Bfgs(Minimizer):
    # TODO: document class

    def __init__(self, wrt, f, fprime, initial_inv_hessian=None,
                 line_search=None, args=None):
        # TODO: document method
        super(Bfgs, self).__init__(wrt, args=args)
        self.f = f
        self.fprime = fprime
        self.inv_hessian = initial_inv_hessian

        if line_search is not None:
            self.line_search = line_search
        else:
            self.line_search = WolfeLineSearch(wrt, self.f, self.fprime)

    def find_direction(self, grad_m1, grad, step, inv_hessian):
        # TODO: document method
        H = self.inv_hessian
        grad_diff = grad - grad_m1
        ys = np.inner(grad_diff, step)
        Hy = np.dot(H, grad_diff)
        yHy = np.inner(grad_diff, Hy)
        H += (ys + yHy) * np.outer(step, step) / ys ** 2
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
                # TODO: inform the user here.
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


class Sbfgs(Bfgs):
    # TODO document

    def __init__(self, wrt, f, fprime, initial_inv_hessian=None,
                 line_search=None, args=None):
        # TODO document
        super(Sbfgs, self).__init__(
            wrt, f, fprime, line_search, args=args)

    def find_direction(self, grad_m1, grad, step, inv_hessian):
        # TODO document
        H = inv_hessian
        grad_diff = grad - grad_m1
        ys = np.inner(grad_diff, step)
        Hy = np.dot(H, grad_diff)
        yHy = np.inner(grad_diff, Hy)
        gamma = ys / yHy
        v = scipy.sqrt(yHy) * (step / ys - Hy / yHy)
        v = scipy.real(v)
        H[:] = gamma * (H - np.outer(Hy, Hy) / yHy + np.outer(v, v))
        H += np.outer(step, step) / ys
        direction = -np.dot(H, grad)
        return direction, {}


class Lbfgs(Minimizer):
    # TODO: document class

    def __init__(self, wrt, f, fprime, initial_hessian_diag=1,
                 n_factors=10, line_search=None,
                 args=None):
        # TODO: document method
        super(Lbfgs, self).__init__(wrt, args=args)

        self.f = f
        self.fprime = fprime
        self.initial_hessian_diag = initial_hessian_diag
        self.n_factors = n_factors
        if line_search is not None:
            self.line_search = line_search
        else:
            self.line_search = WolfeLineSearch(wrt, self.f, self.fprime)

    def find_direction(self, grad_diffs, steps, grad, hessian_diag, idxs):
        grad = grad.copy()  # We will change this.
        n_current_factors = len(idxs)

        # TODO: find a good name for this variable.
        rho = scipy.empty(n_current_factors)

        # TODO: vectorize this function
        for i in idxs:
            rho[i] = 1 / scipy.inner(grad_diffs[i], steps[i])

        # TODO: find a good name for this variable as well.
        alpha = scipy.empty(n_current_factors)

        for i in idxs[::-1]:
            alpha[i] = rho[i] * scipy.inner(steps[i], grad)
            grad -= alpha[i] * grad_diffs[i]
        z = hessian_diag * grad

        # TODO: find a good name for this variable (surprise!)
        beta = scipy.empty(n_current_factors)

        for i in idxs:
            beta[i] = rho[i] * scipy.inner(grad_diffs[i], z)
            z += steps[i] * (alpha[i] - beta[i])

        return z, {}

    def __iter__(self):
        args, kwargs = self.args.next()
        grad = self.fprime(self.wrt, *args, **kwargs)
        grad_m1 = scipy.zeros(grad.shape)
        factor_shape = self.n_factors, self.wrt.shape[0]
        grad_diffs = scipy.zeros(factor_shape)
        steps = scipy.zeros(factor_shape)
        hessian_diag = self.initial_hessian_diag
        step_length = None
        step = scipy.empty(grad.shape)
        grad_diff = scipy.empty(grad.shape)

        # We need to keep track in which order the different statistics
        # from different runs are saved.
        #
        # Why?
        #
        # Each iteration, we save statistics such as the difference between
        # gradients and the actual steps taken. These are then later combined
        # into an approximation of the Hessian. We call them factors. Since we
        # don't want to create a new matrix of factors each iteration, we
        # instead keep track externally, which row of the matrix corresponds
        # to which iteration. `idxs` now is a list which maps its i'th element
        # to the corresponding index for the array. Thus, idx[i] contains the
        # rowindex of the for the (n_factors - i)'th iteration prior to the
        # current one.
        idxs = []

        for i, (next_args, next_kwargs) in enumerate(self.args):
            if i == 0:
                direction = -grad
                info = {}
            else:
                sTgd = scipy.inner(step, grad_diff)
                if sTgd > 1E-10:
                    # Don't do an update if this value is too small.
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
                    grad_diffs[this_idx] = grad_diff
                    steps[this_idx] = step
                    hessian_diag = sTgd / scipy.inner(grad_diff, grad_diff)

                direction, info = self.find_direction(
                    grad_diffs, steps, -grad, hessian_diag, idxs)

            if not is_nonzerofinite(direction):
                warnings.warn('search direction is either 0, nan or inf')
                break

            step_length = self.line_search.search(
                direction, None, args, kwargs)

            step[:] = step_length * direction
            if step_length != 0:
                self.wrt += step
            else:
                warnings.warn('step length is 0')
                pass

            # Prepare everything for the next loop.
            args, kwargs = next_args, next_kwargs
            # TODO: not all line searches have .grad!
            grad_m1[:], grad[:] = grad, self.line_search.grad
            grad_diff = grad - grad_m1

            info.update({
                'step_length': step_length,
                'n_iter': i,
                'args': args,
                'kwargs': kwargs,
                'loss': self.line_search.val,
                'gradient': grad,
                'gradient_m1': grad_m1,
            })
            yield info
