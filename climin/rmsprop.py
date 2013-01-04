# -*- coding: utf-8 -*-


import numpy as np

from base import Minimizer


class RmsProp(Minimizer):

    def __init__(self, wrt, fprime, steprate, decay, momentum=0,
                 args=None, logfunc=None):
        super(RmsProp, self).__init__(wrt, args=args, logfunc=logfunc)

        self.fprime = fprime
        self.steprate = steprate
        self.decay = decay
        self.momentum = momentum

    def __iter__(self):
        moving_mean_squared = 1
        step_m1 = 0

        for i, (args, kwargs) in enumerate(self.args):
            gradient = self.fprime(self.wrt, *args, **kwargs)
            moving_mean_squared = (
                self.decay * moving_mean_squared
                + (1 - self.decay) * gradient**2)
            step = self.steprate * gradient / np.sqrt(moving_mean_squared + 1e-4)
            step += step_m1 * self.momentum
            self.wrt -= step
            step_m1 = step

            yield dict(args=args, kwargs=kwargs, gradient=gradient,
                       n_iter=i,
                       moving_mean_squared=moving_mean_squared,
                       step=step)
