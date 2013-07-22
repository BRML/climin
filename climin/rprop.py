# -*- coding: utf-8 -*-

import itertools
import mathadapt as ma

from base import Minimizer, repeat_or_iter

class Rprop(Minimizer):

    def __init__(self, wrt, f, fprime, step_shrink=0.5, step_grow=1.2,
                 min_step=1E-6, max_step=1, changes_max=0.1,
                 args=None, logfunc=None):
        super(Rprop, self).__init__(wrt, args=args, logfunc=logfunc)

        self.f = f
        self.fprime = fprime
        self.step_shrink = step_shrink
        self.step_grow = step_grow
        self.min_step = min_step
        self.max_step = max_step
        self.changes_max = changes_max

    def __iter__(self):
        grad_m1 = ma.zero_like(self.wrt)
        changes = ma.random_like(self.wrt) * self.changes_max

        for i, (args, kwargs) in enumerate(self.args):
            grad = self.fprime(self.wrt, *args, **kwargs)
            changes_min = changes * self.step_grow
            changes_max = changes * self.step_shrink
            gradprod = grad_m1 * grad
            changes_min *= gradprod > 0
            changes_max *= gradprod < 0
            changes *= gradprod == 0

            # TODO actually, this should be done to changes
            changes_min = ma.clip(changes_min, self.min_step, self.max_step)
            changes_max = ma.clip(changes_max, self.min_step, self.max_step)

            changes += changes_min + changes_max
            step = -changes * ma.sign(grad)
            self.wrt += step

            grad_m1 = grad

            yield dict(args=args, kwargs=kwargs, gradient=grad,
                       gradient_m1=grad_m1, n_iter=i,
                       step=step)
