# -*- coding: utf-8 -*-

import itertools


from base import Minimizer, repeat_or_iter


class GradientDescent(Minimizer):

    def __init__(self, wrt, fandprime, steprate, momentum=0.0, args=None,
                 verbose=False):
        super(GradientDescent, self).__init__(wrt, args, verbose)
        self.steprates = repeat_or_iter(steprate)
        self.momentums = repeat_or_iter(momentum)
        self.fandprime = fandprime

    def __iter__(self):
        step_m1 = 0
        for i in itertools.izip(self.steprates, self.momentums, self.args):
            steprate, momentum, (args, kwargs) = i
            loss, gradient = self.fandprime(*args, **kwargs)
            step = gradient * steprate + momentum * step_m1
            self.wrt -= step
            yield dict(
                loss=loss, gradient=gradient, steprate=steprate, 
                momentum=momentum, step=step, wrt=self.wrt)
            step_m1 = step
