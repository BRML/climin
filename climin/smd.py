# -*- coding: utf-8 -*-

import numpy as np

from base import Minimizer, is_nonzerofinite


class Smd(Minimizer):

    # eta steprate
    # gain

    def __init__(self, wrt, f, fprime, f_Hp, lmbd=0.99,
            mu=2e-2, eta0=5e-5, args=None, logfunc=None):
        super(Smd, self).__init__(wrt, args=args, logfunc=logfunc)

        self.f = f
        self.fprime = fprime
        self.f_Hp = f_Hp
        self.lmbd = lmbd
        self.mu = mu
        self.eta0 =eta0

    def __iter__(self):
        p = np.size(self.wrt)
        v = 0.0001*np.random.randn(p)
        eta = self.eta0 * np.ones(p)

        for i, (args, kwargs) in enumerate(self.args):
            gradient = self.fprime(self.wrt, *args, **kwargs)

            if not is_nonzerofinite(gradient):
                self.logfunc(
                    {'message': 'gradient is invalid -- need to bail out.'})
                break

            Hp = self.f_Hp(self.wrt, v, *args, **kwargs)

            eta = eta * np.maximum(0.5, 1 + self.mu * v * gradient)
            v *= self.lmbd
            v += eta*(gradient - self.lmbd*Hp)

            self.wrt -= eta*gradient

            yield {'gradient': gradient, 'v': v, 'eta': eta}
