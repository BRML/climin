# -*- coding: utf-8 -*-

# TODO: document module
# TODO: check if gnumpy compatible

from base import Minimizer


class Asgd(Minimizer):
    # TODO: document class

    def __init__(self, wrt, fprime, eta0=1e-5, lmbd=1e-4, alpha=0.75, t0=1e8,
                 args=None):
        # TODO document method
        # TODO rename parameters to sth sensible, not greek letters
        # TODO give reference
        super(Asgd, self).__init__(wrt, args=args)

        self.fprime = fprime
        self.eta0 = eta0
        self.lmbd = lmbd
        self.alpha = alpha
        self.t0 = t0
        self.mu_t = 1
        self.eta_t = eta0

    def __iter__(self):
        # do 'determineEta0' (see Bottou) here?
        # w is current estimated 'optimal' parameter
        self.w = self.wrt.copy()
        # wrt is average over 'optimal' w's
        self.wrt *= 0
        step = 0
        for i, (args, kwargs) in enumerate(self.args):
            gradient = self.fprime(self.w, *args, **kwargs)

            # decay w
            self.w *= (1 - self.lmbd * self.eta_t)
            # and update (descent direction)
            self.w -= self.eta_t * gradient

            # use 'optimal' w for some time at the start
            if self.mu_t < 1:
                step = self.mu_t * (self.w - self.wrt)
                self.wrt += step
            else:
                self.wrt *= 0
                self.wrt += self.w

            self.mu_t = 1. / max(1, (i + 1) - self.t0)
            self.eta_t = self.eta0 / ((1 + self.lmbd * self.eta0 * (i + 1)) ** self.alpha)

            yield {
                'n_iter': i,
                'gradient': gradient,
                'mu_t': self.mu_t,
                'eta_t': self.eta_t,
                'step': step,
                'args': args,
                'kwargs': kwargs
            }
