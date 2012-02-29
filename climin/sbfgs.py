# -*- coding: utf-8 -*-

import itertools

import scipy as sp
import numpy as np
import scipy.linalg
import scipy.optimize

from bfgs import Bfgs
from linesearch import WolfeLineSearch
from logging import taggify


class Sbfgs(Bfgs):

    def __init__(self, wrt, f, fprime, initial_inv_hessian=None,
                 line_search=None,
                 args=None, logfunc=None):
        super(Sbfgs, self).__init__(
            wrt, f, fprime, line_search, args=args, logfunc=logfunc)

    def find_direction(self, grad_m1, grad, step, inv_hessian):
        H = inv_hessian
        grad_diff = grad - grad_m1
        ys = np.inner(grad_diff, step)
        ss = np.inner(step, step)
        yy = np.inner(grad_diff, grad_diff)
        Hy = np.dot(H, grad_diff)
        yHy = np.inner(grad_diff, Hy)
        gamma = ys/yHy
        v = scipy.sqrt(yHy) * (step/ys - Hy/yHy)
        v = scipy.real(v)
        H[:] = gamma * (H - np.outer(Hy, Hy) / yHy + np.outer(v, v))
        H += np.outer(step, step) / ys
        direction = -np.dot(H, grad)
        return direction, {}
