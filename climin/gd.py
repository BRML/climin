# -*- coding: utf-8 -*-

import itertools


from base import Minimizer, repeat_or_iter


class GradientDescent(Minimizer):

    def __init__(self, wrt, fprime, steprate=0.1, momentum=0.0, 
                 args=None, logfunc=None):
        super(GradientDescent, self).__init__(
            wrt, args=args, logfunc=logfunc)

        self.fprime = fprime
        self.steprates = repeat_or_iter(steprate)
        self.momentums = repeat_or_iter(momentum)

    def __iter__(self):
        step_m1 = 0
        periterargs = itertools.izip(self.steprates, self.momentums, self.args)
        for i, j in enumerate(periterargs):
            steprate, momentum, (args, kwargs) = j
            gradient = self.fprime(self.wrt, *args, **kwargs)
            step = gradient * steprate + momentum * step_m1
            self.wrt -= step

            yield dict(gradient=gradient, steprate=steprate, 
                       args=args, kwargs=kwargs, n_iter=i,
                       momentum=momentum, step=step)

            step_m1 = step
