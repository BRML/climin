# -*- coding: utf-8 -*-

"""This module provides an implementation of Adam."""


from base import Minimizer
from mathadapt import sqrt, ones_like, clip


class Adam(Minimizer):

    # TODO add docstring

    state_fields = 'n_iter step_rate decay decay_mom1 decay_mom2 step offset est_mom1_b est_mom2_b'.split()

    def __init__(self, wrt, fprime, step_rate=.0002,
                 decay=1-1e-8,
                 decay_mom1=0.1,
                 decay_mom2=0.001,
                 momentum=0,
                 offset=1e-8, args=None):
        # TODO add docstring
        super(Adam, self).__init__(wrt, args=args)

        self.fprime = fprime
        self.step_rate = step_rate
        self.decay = decay
        self.decay_mom1 = decay_mom1
        self.decay_mom2 = decay_mom2
        self.offset = offset
        self.momentum = momentum
        # TODO: add check

        self.est_mom1 = 0
        self.est_mom2 = 0
        self.est_mom1_b = 0
        self.est_mom2_b = 0
        self.step = 0

    def _iterate(self):
        for args, kwargs in self.args:
            step_m1 = self.step
            d = self.decay
            dm1 = self.decay_mom1
            dm2 = self.decay_mom2
            o = self.offset
            t = self.n_iter + 1

            est_mom1_b_m1 = self.est_mom1_b
            est_mom2_b_m1 = self.est_mom2_b

            coeff1 = 1 - (1 - dm1) * d ** (t - 1)
            gradient = self.fprime(self.wrt, *args, **kwargs)
            self.est_mom1_b = coeff1 * gradient + (1 - coeff1) * est_mom1_b_m1
            self.est_mom2_b = dm2 * gradient ** 2 + (1 - dm2) * est_mom2_b_m1

            self.est_mom1 = self.est_mom1_b / (1 - (1 - dm1) ** t + o)
            self.est_mom2 = self.est_mom2_b / (1 - (1 - dm2) ** t + o)

            self.step = self.step_rate * self.est_mom1 / ((self.est_mom2) ** 0.5 + o)

            self.wrt -= self.step

            self.n_iter += 1

            yield {
                'n_iter': self.n_iter,
                'gradient': gradient,
                'args': args,
                'kwargs': kwargs,
            }
