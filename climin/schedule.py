# -*- coding: utf-8 -*-

"""This module holds various schedules for parameters such as the step
rate or momentum for gradient descent.

A schedule is implemented as an iterator. This allows it to have iterators
of infinite length. It also makes it possible to manipulate scheduls with
the ``itertools`` python module, e.g. for chaining iterators.
"""

import itertools
import math

import numpy as np


def decaying(start, decay):
    """Return an iterator of exponentially decaying values.

    The first value is ``start``. Every further value is obtained by multiplying
    the last one by a factor of ``decay``.
    """
    return (start * decay**i for i in itertools.count(0))


def linear_annealing(start, stop, n_steps):
    """Return an iterator that anneals linearly to a point linearly.

    The first value is ``start``, the last value is ``stop``. The annealing will
    be linear over ``n_steps`` iterations. After that, ``stop`` is yielded."""
    inc = (stop - start) / n_steps
    for i in range(n_steps):
        yield start + i * inc
    while True:
        yield stop


def repeater(iter, n):
    """Return an iterator that repeats each element of `iter` exactly
    `n` times before moving on to the next element."""
    for i in iter:
        for j in range(n):
            yield i


def sutskever_blend(max_momentum, stretch=250):
    """Return a schedule that step-wise increases from zero to a maximum value,
    as described in [sutskever2013importance]_.

    .. [sutskever2013importance] On the importance of initialization and
       momentum in deep learning, Sutskever et al (ICML 2013)
    """
    for i in itertools.count(1):
        m = 1 - (2 ** (-1 - math.log(np.floor_divide(i, stretch) + 1, 2)))
        yield min(m, max_momentum)
