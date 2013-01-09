# -*- coding: utf-8 -*-


import numpy as np

from base import Minimizer


class RmsProp(Minimizer):

    def __init__(self, wrt, fprime, steprate, decay=0.9, momentum=0,
                 step_adapt=False,
                 args=None, logfunc=None):
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
        super(RmsProp, self).__init__(wrt, args=args, logfunc=logfunc)

        self.fprime = fprime
        self.steprate = steprate
        self.decay = decay
        self.momentum = momentum
        self.step_adapt = step_adapt

    def __iter__(self):
        moving_mean_squared = 1
        step_m1 = 0
        step_rate = np.empty(self.wrt.shape)
        step_rate = self.steprate

        # If we adapt step rates, we need one for each parameter.
        if self.step_adapt:
            step_rate *= np.ones(self.wrt.shape)

        for i, (args, kwargs) in enumerate(self.args):
            # We use Nesterov momentum: first, we make a step according to the
            # momentum and then we calculate the gradient.
            step1 = step_m1 * self.momentum
            self.wrt -= step1

            gradient = self.fprime(self.wrt, *args, **kwargs)

            moving_mean_squared = (
                self.decay * moving_mean_squared
                + (1 - self.decay) * gradient**2)
            step2 = self.steprate * gradient
            step2 /= np.sqrt(moving_mean_squared + 1e-8)
            self.wrt -= step2

            step = step1 + step2

            # Step rate adaption. If the current step and the momentum agree,
            # we slightly increase the step rate for that dimension.
            if self.step_adapt:
                agree = (np.sign(step) == np.sign(step_m1)).astype('float32')
                adapt = 1 + agree * self.step_adapt * 2 - self.step_adapt
                step_rate *= adapt

            step_m1 = step
            yield dict(args=args, kwargs=kwargs, gradient=gradient,
                       n_iter=i,
                       moving_mean_squared=moving_mean_squared,
                       step=step_m1,
                       step_rate=step_rate)
