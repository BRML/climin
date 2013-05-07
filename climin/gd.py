# -*- coding: utf-8 -*-

import itertools


from base import Minimizer, repeat_or_iter


def decaying(start, decay):
    """Return an iterator that starts out with `start` and decays with
    a factor of `decay`."""
    return (start * decay**i for i in itertools.count(0))


def linear_annealing(start, stop, n_steps):
    """Return an iterator that starts out with `start`, anneals linearly
    towards `stop` over `n_steps` iterations, and subsequently yields `stop`."""
    inc = (stop - start) / n_steps
    for i in range(n_steps):
        yield start + i * inc
    while True:
        yield stop


def repeater(iter, n):
    """Return an iterator that repeats each element of `iter` exactly
    `n` times before moving on to the next element."""
    for i in iter:
        for j in range(n):
            yield i


class GradientDescent(Minimizer):

    def __init__(self, wrt, fprime, steprate=0.1, momentum=0.0,
                 momentum_type='standard',
                 args=None, logfunc=None):
        super(GradientDescent, self).__init__(
            wrt, args=args, logfunc=logfunc)

        self.fprime = fprime
        self.steprates = repeat_or_iter(steprate)
        self.momentums = repeat_or_iter(momentum)
        self.momentum_type = momentum_type

    def __iter__(self):
        step_m1 = 0
        periterargs = itertools.izip(self.steprates, self.momentums, self.args)
        for i, j in enumerate(periterargs):
            steprate, momentum, (args, kwargs) = j

            if self.momentum_type == 'standard':
                gradient = self.fprime(self.wrt, *args, **kwargs)
                step = gradient * steprate + momentum * step_m1
                self.wrt -= step
            elif self.momentum_type == 'nesterov':
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
