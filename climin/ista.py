# -*- coding: utf-8 -*-

"""This module provides functionality for iterative soft-thresholding,
as described in

    Machine learning: a probabilistic perspective
    Kevin Murphy (2012).
"""


import itertools

import numpy as np

from base import Minimizer


def soft(a, b):
    """Return `a` soft thresholded by `b`.

    The mathematical formulation is
    :math:`soft(a, b) = \\text{sign}(a)\max(|a| - b, 0)`.
    """
    return np.sign(a) * np.maximum((abs(a) - b), 0)


class Ista(Minimizer):
    """Ista is a gradient descent based minimizer of a generic loss respecting
    an additional l1 regularizer (or Laplace prior) on the parameters of the
    solution."""

    def __init__(self, wrt, f_loss, f_residual, f_prime, c_l1, step_rate,
                 max_fraction=0.01, max_bad_steps=5,
                 min_step_rate=1e-8, max_step_rate=50, args=None):
        """Create an Ista object.

        Ista works by picking a local regularization scalar and then adapting
        it. to a certain rule. As soon as the regularization scalar given by the
        user is reached, optimization stops.

        :param wrt: Numpy parameter array to optimize.
        :param f_loss: Function returning the loss (without the regularization
            term). The signature is `f(wrt, X, Z) = l` where `wrt` are the
            current parameters, `X` the inputs and `Z` the targets and `l` a
            scalar.
        :param f_residual: Function returning the residuals of the current
            parameters; in most cases, this will be the difference of the
            prediction and the targets. The signature is `f(wrt, X, Z) = r`
            where `wrt` are the current parameters, `X` the inputs, `Z` the
            targets and `r` a numpy array of the same shape as Z containing the
            residuals.
        :param f_prime: Function returning the derivative of the current
            parameters wrt to the parameters.
            The signature is `f(wrt, X, Z) = g`
            where `wrt` are the current parameters, `X` the inputs, `Z` the
            targets and `g` a numpy array of the same shape as `wrt`.
        :param c_l1: The coefficient for the l1 regularization, commonly
            referred to as lambda.
        :param step_rate: The initial step rate.
        :param max_fraction: A constant for picking the next local c_l1.
            The higher the constant, earlier it stops.
        :param max_bad_steps: Maximum number of bad steps the algorithm takes
            before moving on to the next c_l1. Bad steps are those which
            increase the loss.
        :param min_step_rate: Minimum step rate.
        :param max_step_rate: Maximum step rate.
        :param args: Iterator over the arguments for the loss. E.g. an infinite
            iterator over `(X, Z), {}` where `X` is the input and `Z` the target
            data.
        """
        super(Ista, self).__init__(wrt, args=args)

        # For now, it only works with linear least squares. But that will
        # possible be dealt with.
        self.f_loss = f_loss
        self.f_residual = f_residual
        self.f_prime = f_prime

        self.c_l1 = float(c_l1)
        self.step_rate = step_rate
        self.max_fraction = float(max_fraction)
        self.max_bad_steps = max_bad_steps

        self.min_step_rate = min_step_rate
        self.max_step_rate = max_step_rate

    def __iter__(self):
        # Initialize a couple of variables that we will need during the
        # iterations.
        step_rate = self.step_rate
        wrt_m1 = np.zeros(self.wrt.shape)
        gradient_m1 = None
        grad_diff = None
        iter_count = itertools.count()

        # We will need this in every iteration, but it is more efficient to
        # calculate it at its end, so we can report the loss with it. Thus
        # we have to do it once before we loop.
        (D, target), _ = self.args.next()

        # Some shortcuts to fields of the class.
        wrt = self.wrt
        n_samples = target.shape[0]

        residual = self.f_residual(wrt, D, target)

        for (D, target), _ in self.args:
            correlation = np.dot(residual, D.T)
            greatest_correlation = abs(correlation).max()
            c_l1 = max(self.max_fraction * greatest_correlation,
                       self.c_l1)

            losses = []
            while True:
                n_iter = iter_count.next()
                gradient = self.f_prime(wrt, D, target)
                update = wrt - step_rate * gradient

                wrt_diff = wrt - wrt_m1
                if gradient_m1 is not None:
                    grad_diff = gradient - gradient_m1
                    step_rate = (np.dot(wrt_diff.T, wrt_diff)
                                 / (np.dot(wrt_diff.T, grad_diff) + 1e-8))
                    step_rate = np.clip(
                        step_rate, self.min_step_rate, self.max_step_rate)

                wrt_m1[:] = wrt
                threshold = c_l1 * step_rate
                wrt[:] = soft(update, threshold)
                gradient_m1 = gradient

                loss = self.f_loss(wrt, D, target)

                yield {'loss': loss, 'n_iter': n_iter, 'step_rate': step_rate,
                       'update': update, 'c_l1': c_l1, 'gradient': gradient,
                       'threshold': threshold,
                       'greatest_correlation': greatest_correlation,
                       'wrt_diff': wrt_diff, 'grad_diff': grad_diff}

                losses.append(loss)

                if len(losses) > self.max_bad_steps:
                    improvement = -(loss - losses[-self.max_bad_steps - 1])
                    if improvement <= -1e-3:
                        break
            if abs(c_l1 - self.c_l1) < 1e-6:
                break
