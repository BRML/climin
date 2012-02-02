# -*- coding: utf-8 -*-

import itertools

import scipy

from base import Minimizer, repeat_or_iter


class Rprop(Minimizer):

    def __init__(self, wrt, f, fprime, stepshrink=0.5, stepgrow=1.2,
                 minstep=1E-6, maxstep=1, changes_max=0.1,
                 args=None, stop=1, logfunc=None):
        super(Rprop, self).__init__(wrt, args=args, stop=stop, logfunc=logfunc)

        self.f = f
        self.fprime = fprime
        self.stepshrink = stepshrink
        self.stepgrow = stepgrow
        self.minstep = minstep
        self.maxstep = maxstep
        self.changes_max = changes_max

    def __iter__(self):
        grad_m1 = scipy.zeros(self.wrt.shape)
        changes = scipy.random.random(self.wrt.shape) * self.changes_max

        for i, (args, kwargs) in enumerate(self.args):
            grad = self.fprime(self.wrt, *args, **kwargs)
            changes_min = changes * self.stepgrow
            changes_max = changes * self.stepshrink
            gradprod = grad_m1 * grad
            changes_min *= gradprod > 0
            changes_max *= gradprod < 0
            changes *= gradprod == 0

            # TODO actually, this should be done to changes
            changes_min = scipy.clip(changes_min, self.minstep, self.maxstep)
            changes_max = scipy.clip(changes_max, self.minstep, self.maxstep)

            changes += changes_min + changes_max
            step = -changes * scipy.sign(grad)
            self.wrt += step

            grad_m1 = grad

            # TODO: yield correct loss
            # TODO: log something
            if i > 0 and i % self.stop == 0:
                loss = self.f(self.wrt, *args, **kwargs)
                yield dict(loss=loss, args=args, kwargs=kwargs, grad=grad,
                           step=step)
