# -*- coding: utf-8 -*-

import warnings

import numpy as np
import scipy.linalg
import scipy.optimize

from base import Minimizer, is_nonzerofinite
from linesearch import WolfeLineSearch


class NonlinearConjugateGradient(Minimizer):
    """
    Nonlinear Conjugate Gradient Method
    """
    # TODO: document class

    def __init__(self, wrt, f, fprime, epsilon=1e-6, args=None):
        # TODO: document method
        super(NonlinearConjugateGradient, self).__init__(wrt, args=args)
        self.f = f
        self.fprime = fprime

        self.line_search = WolfeLineSearch(wrt, self.f, self.fprime, c2=0.2)
        self.epsilon = epsilon

    def find_direction(self, grad_m1, grad, direction_m1):
        # TODO: document method
        # Computation of beta as a compromise between Fletcher-Reeves
        # and Polak-Ribiere.
        grad_norm_m1 = np.dot(grad_m1, grad_m1)
        grad_diff = grad - grad_m1
        betaFR = np.dot(grad, grad) / grad_norm_m1
        betaPR = np.dot(grad, grad_diff) / grad_norm_m1
        betaHS = np.dot(grad, grad_diff) / np.dot(direction_m1, grad_diff)
        beta = max(-betaFR, min(betaPR, betaFR))

        # Restart if not a direction of sufficient descent, ie if two
        # consecutive gradients are far from orthogonal.
        if np.dot(grad, grad_m1) / grad_norm_m1 > 0.1:
            beta = 0

        direction = -grad + beta * direction_m1
        return direction, {}

    def __iter__(self):
        args, kwargs = self.args.next()
        grad = self.fprime(self.wrt, *args, **kwargs)
        grad_m1 = np.zeros(grad.shape)
        loss = self.f(self.wrt, *args, **kwargs)
        loss_m1 = 0

        for i, (next_args, next_kwargs) in enumerate(self.args):
            if i == 0:
                direction, info = -grad, {}
            else:
                direction, info = self.find_direction(grad_m1, grad, direction)

            if not is_nonzerofinite(direction):
                warnings.warn('gradient is either zero, nan or inf')
                break

            # Line search minimization.
            initialization = 2 * (loss - loss_m1) / np.dot(grad, direction)
            initialization = min(1, initialization)
            step_length = self.line_search.search(
                direction, initialization,  args, kwargs)
            self.wrt += step_length * direction

            # If we don't bail out here, we will enter regions of numerical
            # instability.
            if (abs(grad) < self.epsilon).all():
                warnings.warn('gradient is too small')
                break

            # Prepare everything for the next loop.
            args, kwargs = next_args, next_kwargs
            grad_m1[:], grad[:] = grad, self.line_search.grad
            loss_m1, loss = loss, self.line_search.val

            info.update({
                'loss': loss,
                'step_length': step_length,
                'n_iter': i,
                'args': args,
                'gradient': grad,
                'gradient_m1': grad_m1,
                'kwargs': kwargs,
            })
            yield info
