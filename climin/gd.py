# -*- coding: utf-8 -*-

import itertools


from base import Minimizer, repeat_or_iter


class GradientDescent(Minimizer):

    def __init__(self, wrt, f, fprime, steprate, momentum=0.0, 
                 args=None, stop=1, logfunc=None):
        super(GradientDescent, self).__init__(
            wrt, args=args, stop=stop, logfunc=logfunc)

        self.f = f
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

            if i > 0 and i % self.stop == 0:
                loss = self.f(self.wrt, *args, **kwargs)
                info = dict(
                    loss=loss, gradient=gradient, steprate=steprate, 
                    args=args, kwargs=kwargs,
                    momentum=momentum, step=step, wrt=self.wrt)
                self.logfunc(info)
                yield info

            step_m1 = step
