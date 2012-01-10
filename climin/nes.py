# -*- coding: utf-8 -*-


import scipy
import scipy.linalg

from base import Minimizer


class Xnes(Minimizer):

    def __init__(self, wrt, f, args=None, stop=1, verbose=False):
        super(Xnes, self).__init__(wrt, args=args, stop=stop, verbose=verbose)
        self._f = f
        # Set some default values.
        dim = self.wrt.shape[0]
        log_dim = scipy.log(dim)
        self.step_rate = 0.6 * (3 + log_dim) / dim / scipy.sqrt(dim)
        self.batch_size = 4 + int(scipy.floor(3 * log_dim))

    def f(self, x, *args, **kwargs):
        return -self._f(x, *args, **kwargs)

    def __iter__(self):
        dim = self.wrt.shape[0]
        I = scipy.eye(dim)
    
        # Square root of covariance matrix.
        A = scipy.eye(dim)
        center = self.wrt.copy()
        n_evals = 0
        best_wrt = None
        best_x = float('-inf')
        for i, (args, kwargs) in enumerate(self.args):
            # Draw samples, evaluate and update best solution if a better one
            # was found.
            samples = scipy.random.standard_normal((self.batch_size, dim))
            samples = scipy.dot(samples, A) + center
            fitnesses = [self.f(samples[j], *args, **kwargs) 
                         for j in range(samples.shape[0])]
            fitnesses = scipy.array(fitnesses)
            if fitnesses.max() > best_x:
                best_loss = fitnesses.max()
                self.wrt[:] = samples[fitnesses.argmax()]

            # Update center and variances.
            utilities = self.compute_utilities(fitnesses)
            center += scipy.dot(A, scipy.dot(utilities, samples))
            # TODO: vectorize this
            cov_gradient = sum([u * (scipy.outer(s, s) - I) 
                                for (s, u) in zip(samples, utilities)])
            update = scipy.linalg.expm2(A * cov_gradient * self.step_rate * 0.5)
            A[:] = scipy.dot(A, update)

            yield dict(loss=-best_x)

    def compute_utilities(self, fitnesses):
        n_fitnesses = fitnesses.shape[0]
        ranks = scipy.zeros_like(fitnesses)
        l = sorted(enumerate(fitnesses), key=lambda x: x[1])
        for i, (j, _) in enumerate(l):
            ranks[j] = i
        # smooth reshaping
        utilities = -scipy.log(n_fitnesses - ranks)
        utilities += scipy.log(n_fitnesses / 2. + 1.0)
        utilities = scipy.clip(utilities, 0, float('inf'))
        utilities /= utilities.sum()       # make the utilities sum to 1
        utilities -= 1. / n_fitnesses  # baseline
        return utilities
