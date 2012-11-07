# -*- coding: utf-8 -*-

import itertools

import numpy as np

from gd import GradientDescent


class Ilne(GradientDescent):

    def __iter__(self):
        step_m1 = 0
        periterargs = itertools.izip(self.steprates, self.momentums, self.args)
        for i, j in enumerate(periterargs):
            steprate, momentum, (args, kwargs) = j

            big_jump = momentum * step_m1
            self.wrt -= big_jump

            gradient = self.fprime(self.wrt, *args, **kwargs)
            correction = steprate * gradient
            self.wrt -= correction

            step = big_jump + correction
            yield dict(gradient=gradient, steprate=steprate,
                       args=args, kwargs=kwargs, n_iter=i,
                       momentum=momentum, step=step)

            step_m1 = step
