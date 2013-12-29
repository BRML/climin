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

    Examples
    --------

    >>> from climin.schedule import decaying
    >>> s = decaying(10, .9)
    >>> [s.next() for i in range(5)]
    [10.0, 9.0, 8.100000000000001, 7.290000000000001, 6.561]
    """
    return (start * decay ** i for i in itertools.count(0))


def linear_annealing(start, stop, n_steps):
    """Return an iterator that anneals linearly to a point linearly.

    The first value is ``start``, the last value is ``stop``. The annealing will
    be linear over ``n_steps`` iterations. After that, ``stop`` is yielded.

    Examples
    --------

    >>> from climin.schedule import linear_annealing
    >>> s = linear_annealing(1, 0, 4)
    >>> [s.next() for i in range(10)]
    [1.0, 0.75, 0.5, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    """
    start, stop = float(start), float(stop)
    inc = (stop - start) / n_steps
    for i in range(n_steps):
        yield start + i * inc
    while True:
        yield stop


def repeater(iter, n):
    """Return an iterator that repeats each element of `iter` exactly
    `n` times before moving on to the next element.

    Examples
    --------

    >>> from climin.schedule import repeater
    >>> s = repeater([1, 2, 3], 2)
    >>> [s.next() for i in range(6)]
    [1, 1, 2, 2, 3, 3]
    """
    for i in iter:
        for j in range(n):
            yield i


class SutskeverBlend(object):
    """Class representing a schedule that step-wise increases from zero to a
    maximum value, as described in [sutskever2013importance]_.

    Examples
    --------

    >>> from climin.schedule import SutskeverBlend
    >>> s = iter(SutskeverBlend(0.9, 2))
    >>> [s.next() for i in range(10)]
    [0.5, 0.75, 0.75, 0.8333333333333333, 0.8333333333333333, 0.875, 0.875, 0.9, 0.9, 0.9]

    .. [sutskever2013importance] On the importance of initialization and
       momentum in deep learning, Sutskever et al (ICML 2013)
    """

    def __init__(self, max_momentum, stretch=250):
        self.max_momentum = max_momentum
        self.stretch = stretch

    def __iter__(self):
        for i in itertools.count(1):
            m = 1 - (2 ** (-1 - math.log(np.floor_divide(i, self.stretch) + 1, 2)))
            yield min(m, self.max_momentum)
