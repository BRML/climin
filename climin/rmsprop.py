# -*- coding: utf-8 -*-

"""This module provides an implementation of rmsprop."""

from __future__ import absolute_import

import numpy as np

from .base import Minimizer
from .mathadapt import sqrt, ones_like, clip


class RmsProp(Minimizer):
    """RmsProp optimizer.

    RmsProp [tieleman2012rmsprop]_ is an optimizer that utilizes the magnitude
    of recent gradients to normalize the gradients. We always keep a moving
    average over the root mean squared (hence Rms) gradients, by which we
    divide the current gradient. Let :math:`f'(\\theta_t)` be the derivative of
    the loss with respect to the parameters at time step :math:`t`. In its
    basic form, given a step rate :math:`\\alpha` and a decay term
    :math:`\\gamma` we perform the following updates:

    .. math::
        r_t &=& (1 - \\gamma)~f'(\\theta_t)^2 + \\gamma r_{t-1} , \\\\
        v_{t+1} &=&  {\\alpha \over \sqrt{r_t}} f'(\\theta_t), \\\\
        \\theta_{t+1} &=& \\theta_t - v_{t+1}.

    In some cases, adding a momentum term :math:`\\beta` is beneficial. Here,
    Nesterov momentum is used:

    .. math::
        \\theta_{t+{1 \over 2}} &=& \\theta_t - \\beta v_t, \\\\
        r_t &=& (1 - \\gamma)~f'(\\theta_{t + {1 \\over 2}})^2 + \\gamma r_{t-1}, \\\\
        v_{t+1} &=& \\beta v_t + {\\alpha \over \sqrt{r_t}} f'(\\theta_{t + {1 \over 2}}), \\\\
        \\theta_{t+1} &=& \\theta_t - v_{t+1}

    Additionally, this implementation has adaptable step rates. As soon as the
    components of the step and the momentum point into the same direction (thus
    have the same sign) the step rate for that parameter is multiplied with
    ``1 + step_adapt``. Otherwise, it is multiplied with ``1 - step_adapt``.
    In any way, the minimum and maximum step rates ``step_rate_min`` and
    ``step_rate_max`` are respected and exceeding values truncated to it.

    RmsProp has several advantages; for one, it is a very robust optimizer which
    has pseudo curvature information. Additionally, it can deal with stochastic
    objectives very nicely, making it applicable to mini batch learning.

    .. note::
       Works with gnumpy.

    .. [tieleman2012rmsprop]  Tieleman, T. and Hinton, G. (2012),
       Lecture 6.5 - rmsprop, COURSERA: Neural Networks for Machine Learning


    Attributes
    ----------
    wrt : array_like
        Current solution to the problem. Can be given as a first argument to \
        ``.fprime``.

    fprime : Callable
        First derivative of the objective function. Returns an array of the \
        same shape as ``.wrt``.

    step_rate : float or array_like
        Step rate of the optimizer. If an array, means that per parameter step
        rates are used.

    momentum : float or array_like
        Momentum of the optimizer. If an array, means that per parameter
        momentums are used.

    step_adapt : float or bool
        Constant to adapt step rates. If False, step rate adaption is not done.

    step_rate_min : float, optional, default 0
        When adapting step rates, do not move below this value.

    step_rate_max : float, optional, default inf
        When adapting step rates, do not move above this value.
    """

    @property
    def step_rate(self):
        return self._step_rate

    @step_rate.setter
    def step_rate(self, value):
        self._step_rate = value

        # If we adapt step rates, we need one for each parameter.
        if self.step_adapt:
            self._step_rate *= ones_like(self.wrt)

    state_fields = ('n_iter decay momentum step_adapt step_rate_min step_rate_max '
                    'step_rate moving_mean_squared step').split()

    def __init__(self, wrt, fprime, step_rate, decay=0.9, momentum=0,
                 step_adapt=False, step_rate_min=0, step_rate_max=np.inf,
                 args=None):
        """Create an RmsProp object.

        Parameters
        ----------

        wrt : array_like
            Array that represents the solution. Will be operated upon in
            place.  ``fprime`` should accept this array as a first argument.

        fprime : callable
            Callable that given a solution vector as first parameter and *args
            and **kwargs drawn from the iterations ``args`` returns a
            search direction, such as a gradient.

        step_rate : float or array_like
            Step rate to use during optimization. Can be given as a single
            scalar value or as an array for a different step rate of each
            parameter of the problem.

        decay : float
            Decay parameter for the moving average. Must lie in [0, 1) where
            lower numbers means a shorter "memory".

        momentum : float or array_like
          Momentum to use during optimization. Can be specified analoguously
          (but independent of) step rate.

        step_adapt : float or bool
            Constant to adapt step rates. If False, step rate adaption is not done.

        step_rate_min : float, optional, default 0
            When adapting step rates, do not move below this value.

        step_rate_max : float, optional, default inf
            When adapting step rates, do not move above this value.

        args : iterable
            Iterator over arguments which ``fprime`` will be called with.
        """
        super(RmsProp, self).__init__(wrt, args=args)

        self.fprime = fprime
        self.decay = decay
        self.momentum = momentum
        self.step_adapt = step_adapt
        self.step_rate_min = step_rate_min
        self.step_rate_max = step_rate_max

        # Call here since setter depends on existence of step_adapt.
        self.step_rate = step_rate

        self.moving_mean_squared = 1
        self.step = 0

    def _iterate(self):
        for args, kwargs in self.args:
            step_m1 = self.step
            # We use Nesterov momentum: first, we make a step according to the
            # momentum and then we calculate the gradient.
            step1 = step_m1 * self.momentum
            self.wrt -= step1
            gradient = self.fprime(self.wrt, *args, **kwargs)

            self.moving_mean_squared = (
                self.decay * self.moving_mean_squared
                + (1 - self.decay) * gradient ** 2)
            step2 = self.step_rate * gradient
            step2 /= sqrt(self.moving_mean_squared + 1e-8)
            self.wrt -= step2

            step = step1 + step2

            # Step rate adaption. If the current step and the momentum agree,
            # we slightly increase the step rate for that dimension.
            if self.step_adapt:
                # This code might look weird, but it makes it work with both
                # numpy and gnumpy.
                step_non_negative = step > 0
                step_m1_non_negative = step_m1 > 0
                agree = (step_non_negative == step_m1_non_negative) * 1.
                adapt = 1 + agree * self.step_adapt * 2 - self.step_adapt
                self.step_rate *= adapt
                self.step_rate = clip(
                    self.step_rate, self.step_rate_min, self.step_rate_max)

            self.step = step
            self.n_iter += 1
            yield dict(gradient=gradient, args=args, kwargs=kwargs)
