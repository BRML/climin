# -*- coding: utf-8 -*-

"""This module provides an implementation of adadelta."""


from base import Minimizer
from mathadapt import sqrt, ones_like, clip


class Adadelta(Minimizer):
    def __init__(self, wrt, fprime, decay=0.9, offset=1e-2, args=None):
        """Create an Adadelta object.

        Parameters
        ----------

        wrt : array_like
            Array that represents the solution. Will be operated upon in
            place.  ``fprime`` should accept this array as a first argument.

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
        self.decay = decay
        self.offset = offset

    def __iter__(self):
        gms = 0     # running average of the squared gradients
        sms = 0     # running average of the squared updates
        d = self.decay
        o = self.offset

        for i, (args, kwargs) in enumerate(self.args):
            gradient = self.fprime(self.wrt, *args, **kwargs)
            gms = (d * gms) + (1 - d) * gradient ** 2
            step = sqrt(sms + o) / sqrt(gms + o) * gradient
            sms = (d * sms) + (1 - d) * step ** 2
            self.wrt -= step

            yield {
                'n_iter': i,
                'gradient': gradient,
                'gms': gms,
                'step': sms,
                'args': args,
                'kwargs': kwargs,
            }
