# -*- coding: utf-8 -*-

# TODO document


import numpy as np

from base import Minimizer
from mathadapt import sqrt, ones_like, clip


class RmsProp(Minimizer):

    # TODO document
    def __init__(self, wrt, fprime, steprate, decay=0.9, momentum=0,
                 step_adapt=False, step_rate_min=0, step_rate_max=np.inf,
                 args=None):
        """Create an RmsProp object.

        :param wrt: Numpy array of the parameters to optimize.
        :param fprime: Function calculating the gradient of the loss.
        :param steprate: Step rate to use.
        :param: Decay of the moving average over squared gradients.
        :param momentum: Momentum value.
        :param step_adapt: Adapt the step rate for each parameter. If the
            momentum and the current step agree, the step rate will be increase
            by a factor of ``1 + step_adapt``. If they do not agree, the step
            rate will be decreased by a factor of ``1 - step_adapt``.
        :param args: Iterator over arguments which ``fprime`` will be called
            with.
        :param logfunc: Function that will be called to log messages.
        """
        # TODO Adapt documentation to numpydoc.
        super(RmsProp, self).__init__(wrt, args=args)

        self.fprime = fprime
        self.steprate = steprate
        self.decay = decay
        self.momentum = momentum
        self.step_adapt = step_adapt
        self.step_rate_min = step_rate_min
        self.step_rate_max = step_rate_max

    def __iter__(self):
        moving_mean_squared = 1
        step_m1 = 0
        step_rate = self.steprate

        # If we adapt step rates, we need one for each parameter.
        if self.step_adapt:
            step_rate *= ones_like(self.wrt)

        for i, (args, kwargs) in enumerate(self.args):
            # We use Nesterov momentum: first, we make a step according to the
            # momentum and then we calculate the gradient.
            step1 = step_m1 * self.momentum
            self.wrt -= step1

            gradient = self.fprime(self.wrt, *args, **kwargs)

            moving_mean_squared = (
                self.decay * moving_mean_squared
                + (1 - self.decay) * gradient ** 2)
            step2 = step_rate * gradient
            step2 /= sqrt(moving_mean_squared + 1e-8)
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
                step_rate *= adapt
                step_rate = clip(
                    step_rate, self.step_rate_min, self.step_rate_max)

            step_m1 = step
            yield {
                'n_iter': i,
                'gradient': gradient,
                'moving_mean_squared': moving_mean_squared,
                'step': step_m1,
                'args': args,
                'kwargs': kwargs,
                'step_rate': step_rate
            }
