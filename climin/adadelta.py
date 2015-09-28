# -*- coding: utf-8 -*-

"""This module provides an implementation of adadelta."""

from __future__ import absolute_import

from .base import Minimizer
from .mathadapt import sqrt, ones_like, clip


class Adadelta(Minimizer):
    """Adadelta optimizer.

    Adadelta [zeiler2013adadelta]_ is a method that uses the magnitude of recent
    gradients and steps to obtain an adaptive step rate. An exponential moving
    average over the gradients and steps is kept; a scale of the learning rate
    is then obtained by their ration.

    Let :math:`f'(\\theta_t)` be the derivative of the loss with respect to the
    parameters at time step :math:`t`. In its
    basic form, given a step rate :math:`\\alpha`, a decay term
    :math:`\\gamma` and an offset :math:`\\epsilon` we perform the following
    updates:

    .. math::
       g_t &=& (1 - \\gamma)~f'(\\theta_t)^2 + \\gamma g_{t-1}

    where :math:`g_0 = 0`. Let :math:`s_0 = 0` for updating the parameters:

    .. math::
       \\Delta \\theta_t &=& \\alpha {\sqrt{s_{t-1} + \\epsilon} \over \sqrt{g_t + \\epsilon}}~f'(\\theta_t), \\\\
       \\theta_{t+1} &=& \\theta_t - \\Delta \\theta_t.

    Subsequently we adapt the moving average of the steps:

    .. math::
       s_t &=& (1 - \\gamma)~\\Delta\\theta_t^2 + \\gamma s_{t-1}.

    To extend this with Nesterov's accelerated gradient, we need a momentum
    coefficient :math:`\\beta` and incorporate it by using slightly different
    formulas:

    .. math::
        \\theta_{t + {1 \over 2}} &=& \\theta_t - \\beta \\Delta \\theta_{t-1}, \\\\
       g_t &=& (1 - \\gamma)~f'(\\theta_{t + {1 \over 2}})^2 + \\gamma g_{t-1}, \\\\
       \\Delta \\theta_t &=& \\alpha {\sqrt{s_{t-1} + \\epsilon} \over \sqrt{g_t + \\epsilon}}~f'(\\theta_{t + {1 \over 2}}).

    In its original formulation, the case :math:`\\alpha = 1, \\beta = 0` was
    considered only.

    .. [zeiler2013adadelta] Zeiler, Matthew D.
       "ADADELTA: An adaptive learning rate method."
       arXiv preprint arXiv:1212.5701 (2012).
    """

    state_fields = 'n_iter gms sms step step_rate decay offset momentum'.split()

    def __init__(self, wrt, fprime, step_rate=1, decay=0.9, momentum=0,
                 offset=1e-4, args=None):
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

        step_rate : scalar or array_like, optional [default: 1]
            Value to multiply steps with before they are applied to the
            parameter vector.

        decay : float, optional [default: 0.9]
            Decay parameter for the moving average. Must lie in [0, 1) where
            lower numbers means a shorter "memory".

        momentum : float or array_like, optional [default: 0]
          Momentum to use during optimization. Can be specified analoguously
          (but independent of) step rate.

        offset : float, optional, [default: 1e-4]
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
            step1 = step_m1 * m
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
