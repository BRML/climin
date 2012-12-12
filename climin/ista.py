# -*- coding: utf-8 -*-


import itertools

import numpy as np

from base import Minimizer


def soft(a, b):
    return np.sign(a) * np.maximum((abs(a) - b), 0)


class Ista(Minimizer):

    def __init__(self, wrt, dictionary, target, c_l1, step_rate,
                 max_fraction, max_bad_steps=5,
                 min_step_rate=1e-8, max_step_rate=50):
        super(Ista, self).__init__(wrt)

        # For now, it only works with linear least squares. But that will
        # possible be dealt with.
        self.f_predict = lambda wrt, x: np.dot(x, wrt)
        self.f_prime = lambda wrt, x, y: np.dot(x.T, np.dot(x, wrt) - y) / y.shape[0]

        self.dictionary = dictionary
        self.target = target
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

        # Some shortcuts to fields of the class.
        D = self.dictionary
        wrt = self.wrt
        target = self.target
        n_samples = target.shape[0]

        # We will need this in every iteration, but it is more efficient to
        # calculate it at its end, so we can report the loss with it. Thus
        # we have to do it once before we loop.
        residual = self.f_predict(wrt, D) - target

        while True:
            correlation = np.dot(residual.T, D)
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

                residual = self.f_predict(wrt, D) - target
                l1_loss = self.c_l1 * abs(self.wrt).sum() / n_samples
                rss = (residual**2).mean(axis=0).sum()
                loss = rss + l1_loss

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
