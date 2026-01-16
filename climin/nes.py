# -*- coding: utf-8 -*-

# This code has been adapted from PyBrain. See http://pybrain.org.
#
# Copyright (c) 2009, PyBrain-Developers
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright notice,
#       this list of conditions and the following disclaimer in the documentation
#       and/or other materials provided with the distribution.
#     * Neither the name of PyBrain nor the names of its contributors
#       may be used to endorse or promote products derived from this software
#       without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import scipy.linalg

from .base import Minimizer


class Xnes(Minimizer):
    # TODO: document class

    def __init__(self, wrt, f, args=None):
        # TODO: document method
        super(Xnes, self).__init__(wrt, args=args)
        self._f = f
        # Set some default values.
        dim = self.wrt.shape[0]
        log_dim = np.log(dim)
        self.step_rate = 0.6 * (3 + log_dim) / dim / np.sqrt(dim)
        self.batch_size = 4 + int(np.floor(3 * log_dim))

    def set_from_info(self, info):
        raise NotImplementedError("nobody has found the time to implement this yet")

    def extended_info(self, **kwargs):
        raise NotImplementedError("nobody has found the time to implement this yet")

    def f(self, x, *args, **kwargs):
        return -self._f(x, *args, **kwargs)

    def __iter__(self):
        dim = self.wrt.shape[0]
        I = np.eye(dim)

        # Square root of covariance matrix.
        A = np.eye(dim)
        center = self.wrt.copy()
        n_evals = 0
        best_wrt = None
        best_x = float("-inf")
        for i, (args, kwargs) in enumerate(self.args):
            # Draw samples, evaluate and update best solution if a better one
            # was found.
            samples = np.random.standard_normal((self.batch_size, dim))
            samples = np.dot(samples, A) + center
            fitnesses = [self.f(samples[j], *args, **kwargs) for j in range(samples.shape[0])]
            fitnesses = np.array(fitnesses).flatten()

            if fitnesses.max() > best_x:
                best_loss = fitnesses.max()
                self.wrt[:] = samples[fitnesses.argmax()]

            # Update center and variances.
            utilities = self.compute_utilities(fitnesses)
            center += np.dot(np.dot(utilities, samples), A)
            # TODO: vectorize this
            cov_gradient = sum([u * (np.outer(s, s) - I) for (s, u) in zip(samples, utilities)])
            update = scipy.linalg.expm(A * cov_gradient * self.step_rate * 0.5)
            A[:] = np.dot(A, update)

            yield dict(loss=-best_x, n_iter=i)

    def compute_utilities(self, fitnesses):
        n_fitnesses = fitnesses.shape[0]
        ranks = np.zeros_like(fitnesses)
        l = sorted(enumerate(fitnesses), key=lambda x: x[1])
        for i, (j, _) in enumerate(l):
            ranks[j] = i
        # smooth reshaping

        # If we do not cast to float64 here explicitly, numpy will at random
        # points crash with a weird AttributeError.
        utilities = -np.log((n_fitnesses - ranks).astype("float64"))
        utilities += np.log(n_fitnesses / 2.0 + 1.0)
        utilities = np.clip(utilities, 0, float("inf"))
        utilities /= utilities.sum()  # make the utilities sum to 1
        utilities -= 1.0 / n_fitnesses  # baseline
        return utilities
