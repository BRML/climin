# -*- coding: utf-8 -*-

"""This module provides an implementation of adadelta."""


from base import Minimizer
from mathadapt import sqrt, ones_like, clip


class Adadelta(Minimizer):

    state_fields = 'n_iter gms sms step step_rate decay offset momentum'.split()

    def __init__(self, wrt, fprime, step_rate=1, decay=0.9, momentum=0, offset=1e-2,
                 args=None):
        """Create an Adadelta object.

        Parameters
        ----------

        wrt : array_like
            Array that represents the solution. Will be operated upon in
            place.  ``fprime`` should accept this array as a first argument.

        step_rate : scalar or array_like
            Value to multiply steps with before they are applied to the
            parameter vector.

        fprime : callable
            Callable that given a solution vector as first parameter and *args
            and **kwargs drawn from the iterations ``args`` returns a
            search direction, such as a gradient.

        decay : float
            Decay parameter for the moving average. Must lie in [0, 1) where
            lower numbers means a shorter "memory".

        offset : float
            Before taking the square root of the running averages, this offset
            is added.

        args : iterable
            Iterator over arguments which ``fprime`` will be called with.
        """
        super(Adadelta, self).__init__(wrt, args=args)

        self.fprime = fprime
        self.step_rate = step_rate
        self.decay = decay
        self.offset = offset
        self.momentum = momentum

        self.gms = 0
        self.sms = 0
        self.step = 0

    def _iterate(self):
        for args, kwargs in self.args:
            step_m1 = self.step
            d = self.decay
            o = self.offset
            m = self.momentum
            step1 = step_m1 * m * self.step_rate
            self.wrt -= step1

            gradient = self.fprime(self.wrt, *args, **kwargs)

            self.gms = (d * self.gms) + (1 - d) * gradient ** 2
            step2 = sqrt(self.sms + o) / sqrt(self.gms + o) * gradient * self.step_rate
            self.wrt -= step2

            self.step = step1 + step2
            self.sms = (d * self.sms) + (1 - d) * self.step ** 2

            self.n_iter += 1

            yield {
                'n_iter': self.n_iter,
                'gradient': gradient,
                'args': args,
                'kwargs': kwargs,
            }
