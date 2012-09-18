# -*- coding: utf-8 -*-


import time

# Stop criterions can be simple functions like the above one, but
# most of the following functions are actually stop criterion
# factories, that return a stop criterion upon initialization.


def after_n_iterations(n):
    """Return a stop criterion that stops after `n` iterations. """
    def inner(info):
        return info['n_iter'] >= n - 1
    return inner


def modulo_n_iterations(n):
    """Return a stop criterion that stops at each `n`-th iteration.

    E.g.  for n=5, stops at n_iter = 0, 5, 10, 15, ...
    """
    def inner(info):
        return info['n_iter'] % n == 0
    return inner


def time_elapsed(sec):
    """Return a stop criterion that stops after `sec` seconds after
    initializing."""
    start = time.time()
    def inner(info):
        return time.time() - start > sec
    return inner


def converged(func_or_key, n=10, epsilon=1e-5):
    """Return a stop criterion that remembers the last `n` values of
    `func_or_key`() and stops if the difference of their maximum and their
    minimum is smaller than `epsilon`.

    `func_or_key` needs to be a callable that returns a scalar value or a
    string which is a key referring to an entry in the info dict it is given.
    """
    ringbuffer = [None for i in xrange(n)]
    def inner(info):
        if isinstance(func_or_key, (str, unicode)):
            val = info[func_or_key]
        else:
            val = func_or_key()
        ringbuffer.append(val)
        ringbuffer.pop(0)
        if not None in ringbuffer:
            ret = max(ringbuffer) - min(ringbuffer) < epsilon
        else:
            ret = False

        return ret
    return inner


def rising(func, n=1, epsilon=0):
    """Return a stop criterion that remembers the last `n` results of `func` and
    returns True if the its return value rose at least by `epsilon` in the
    meantime."""
    results = []
    def inner(info):
        results.append(func())
        if len(results) < n + 1:
            return False
        if results[-n - 1] + epsilon < results[-1]:
            return True
        else:
            return False
    return inner


def all_(criterions):
    """Return a stop criterion that given a list `criterions` of stop criterions
    only returns True, if all of criterions return True.

    This basically implements a logical AND for stop criterions.
    """
    def inner(info):
        return all(c(info) for c in criterions)
    return inner


def any_(criterions):
    """Return a stop criterion that given a list `criterions` of stop criterions
    only returns True, if any of the criterions returns True.

    This basically implements a logical OR for stop criterions.
    """
    def inner(info):
        return any(c(info) for c in criterions)
    return inner


