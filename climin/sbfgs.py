# -*- coding: utf-8 -*-
# TODO document


import numpy as np
import scipy.linalg
import scipy.optimize

from bfgs import Bfgs


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
