# -*- coding: utf-8 -*-

# TODO document

import warnings

import numpy as np

from base import Minimizer, is_nonzerofinite


class Smd(Minimizer):
    # TODO document

    # eta steprate
    # gain

    def __init__(self, wrt, f, fprime, f_Hp, lmbd=0.99, mu=2e-2, eta0=5e-5,
                 args=None):
        # TODO fin better variable names
        # TODO document
        super(Smd, self).__init__(wrt, args=args)

        self.f = f
        self.fprime = fprime
        self.f_Hp = f_Hp
        self.lmbd = lmbd
        self.mu = mu
        self.eta0 = eta0

    def __iter__(self):
        p = np.size(self.wrt)
        v = 0.0001 * np.random.randn(p)
        eta = self.eta0 * np.ones(p)

        for i, (args, kwargs) in enumerate(self.args):
            gradient = self.fprime(self.wrt, *args, **kwargs)

            if not is_nonzerofinite(gradient):
                warnings.warn('gradient is either zero, nan or inf')
                break

            Hp = self.f_Hp(self.wrt, v, *args, **kwargs)

            eta = eta * np.maximum(0.5, 1 + self.mu * v * gradient)
            v *= self.lmbd
            v += eta * (gradient - self.lmbd * Hp)

            self.wrt -= eta * gradient

            yield {
                'n_iter': i,
                'args': args,
                'kwargs': kwargs,

                'gradient': gradient,
                'v': v, 'eta': eta
            }
