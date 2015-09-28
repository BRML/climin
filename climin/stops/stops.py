# -*- coding: utf-8 -*-

"""Module that contains functionality to monitor stopping.


Rationale
---------

In machine learning, optimization is usually not performed until the objective
is minimized; instead, if this is the case, the true objective (to which the
loss function being minimized is often just a proxy) is what is more important.

To achieve good results, several heuristics have been proposed to monitor for
convergence. This module collects some of these.


Usage
-----

A stopping criterion is a function  which takes a climin ``info`` dictionary as
its only argument. It the returns ``True`` if the stopping criterion is
fulfilled, that is if we should stop. The functions in this module are mostly
functions which create these functions. The idea behind this is that we have
a common API with functions which are supposed to have a state, which can be
realized by generator functions or objects with a ``__call__`` magic method.
"""

from __future__ import absolute_import

import itertools
import signal
import sys
import time

from climin.mathadapt import isnan

from ..compat import basestring

class AfterNIterations(object):
    """AfterNIterations class.

    Useful for stopping after an amount of iterations performed.

    Internally, the ``n_iter`` field of the climin info dictionary is
    inspected; if the value in there exceeds ``n`` by one, the criterion
    returns ``True``.


    Attributes
    ----------

    max_iter : int
        Maximum amount of iterations after which we stop.


    Examples
    --------

    >>> S.AfterNIterations(10)({'n_iter': 10})
    True
    >>> S.AfterNIterations(10)({'n_iter': 5})
    False
    >>> S.AfterNIterations(10)({'n_iter': 9})
    True
    """

    def __init__(self, max_iter):
        """Create AfterNIterations object.


         Parameters
         ----------

         max_iter : int
            Maximum amount of iterations after which we stop.
        """
        self.max_iter = max_iter

    def __call__(self, info):
        return info['n_iter'] >= self.max_iter - 1


class ModuloNIterations(object):
    """Class representing a stop criterion that stops at each `n`-th iteration.

    This is useful if one wants a regular pause in optimization, e.g. to save
    data to disk or give feedback to the user.

    Attributes
    ----------

    n : int
      Number of iterations to perform between pauses.


    Examples
    --------

    >>> S.ModuleNIterations(10)({'n_iter': 9})
    False
    >>> S.ModuleNIterations(10)({'n_iter': 10})
    True
    >>> S.ModuleNIterations(10)({'n_iter': 11})
    False
    """

    def __init__(self, n):
        """Create a ModuloNIterations object.

        Parameters
        ----------

        n : int
            Number of iterations to perform between pauses.
        """
        self.n = n

    def __call__(self, info):
        return info['n_iter'] % self.n == 0


class TimeElapsed(object):
    """Stop criterion that stops after `sec` seconds after
    initializing.

    Attributes
    ----------

    sec : float
      Number of seconds until the criterion returns True.

    Examples
    --------

    >>> stop = S.TimeElapsed(.5); stop({})
    False
    >>> time.sleep(0.5)
    >>> stop({})
    True
    >>> stop2 = S.TimeElapsed(10); stop2({'runtime': 9})
    False
    >>> stop3 = S.TimeElapsed(10); stop2({'runtime': 11})
    True
    """

    def __init__(self, sec):
        """Create a TimeElapsed object.

        Parameters
        ----------

        sec : float
            Number of seconds until the criterion returns True.
        """
        self.sec = sec
        self.start = time.time()

    def __call__(self, info):
        if 'runtime' in info:
            return info['runtime'] > self.sec
        else:
            return time.time() - self.start > self.sec


def All(criterions):
    """Class representing a stop criterion that given a list `criterions` of
    stop criterions only returns True, if all of criterions return True.

    This basically implements a logical AND for stop criterions.
    """
    # TODO document

    def __init__(self, criterions):
        self.criterions = criterions

    def __call__(self, info):
        return all(c(info) for c in self.criterions)


class Any(object):
    """Class representing a stop criterion that given a list `criterions` of
    stop criterions only returns True, if any of the criterions returns True.

    This basically implements a logical OR for stop criterions.
    """
    # TODO document

    def __init__(self, criterions):
        self.criterions = criterions

    def __call__(self, info):
        return any(c(info) for c in self.criterions)


class NotBetterThanAfter(object):
    """Stop criterion that returns True if the error is not less than
    `minimal` after `n_iter` iterations."""

    def __init__(self, minimal, after, key='loss'):
        self.minimal = minimal
        self.after = after
        self.key = key

    def __call__(self, info):
        return info['n_iter'] > self.after and info[self.key] >= self.minimal


class IsNaN(object):
    """Stop criterion that returns True if any value corresponding to
    user-specified keys is NaN.

    Attributes
    ----------

    keys : list
      List of keys to check whether nan or not

    Examples
    --------

    >>> stop = S.IsNaN(['test']); stop({'test': 0})
    False
    >>> stop({'test': numpy.nan})
    True
    >>> stop({'test': gnumpy.as_garray(numpy.nan)})
    True
    """

    def __init__(self, keys=[]):
        self.keys = keys

    def __call__(self, info):
        return any([isnan(info.get(key, 0)) for key in self.keys])


class Patience(object):
    """Stop criterion inspired by Bengio's patience method.

    The idea is to increase the number of iterations until stopping by
    a multiplicative and/or additive constant once a new best candidate is
    found.

    Attributes
    ----------

    func_or_key : function, hashable
        Either a function or a hashable object. In the first case, the function
        will be called to get the latest loss. In the second case, the loss
        will be obtained from the in the corresponding field of the ``info``
        dictionary.

    initial : int
        Initial patience. Lower bound on the number of iterations.

    grow_factor : float
        Everytime we find a sufficiently better candidate (determined by
        ``threshold``) we increase the patience multiplicatively by
        ``grow_factor``.

    grow_offset : float
        Everytime we find a sufficiently better candidate (determined by
        ``threshold``) we increase the patience additively by ``grow_offset``.

    threshold : float, optional, default: 1e-4
        A loss of a is assumed to be a better candidate than b, if a is larger
        than b by a margin of ``threshold``.

    """

    def __init__(self, func_or_key, initial, grow_factor=1., grow_offset=0.,
                 threshold=1e-4):
        if grow_factor == 1 and grow_offset == 0:
            raise ValueError('need to specify either grow_factor != 1'
                             'or grow_offset != 0)')

        self.func_or_key = func_or_key
        self.patience = initial
        self.grow_factor = grow_factor
        self.grow_offset = grow_offset
        self.threshold = threshold

        self.best_iter = 0
        self.best_loss = float('inf')
        self.count = itertools.count()

    def __call__(self, info):
        i = info['n_iter']
        if isinstance(self.func_or_key, basestring):
            loss = info[self.func_or_key]
        else:
            loss = self.func_or_key()

        if loss < self.best_loss:
            if (self.best_loss - loss) > self.threshold and i > 0:
                self.patience = max(i * self.grow_factor + self.grow_offset,
                                    self.patience)
            self.best_iter = i
            self.best_loss = loss

        return i >= self.patience


class OnUnixSignal(object):
    """Stopping criterion that is sensitive to some signal."""

    def __init__(self, sig=signal.SIGINT):
        """Return a stopping criterion that stops upon a signal.

        Previous handler will be overwritten.


        Parameters
        ----------

        sig : signal, optional [default: signal.SIGINT]
            Signal upon which to stop.
        """
        self.sig = sig
        self.stopped = False
        self._register()

    def _register(self):
        self.prev_handler = signal.signal(self.sig, self.handler)

    def handler(self, signal, frame):
        self.stopped = True

    def __call__(self, info):
        res, self.stopped = self.stopped, False
        return res

    def __del__(self):
        signal.signal(self.sig, self.prev_handler)

    def __setstate__(self, dct):
        self.__dict__.update(dct)
        self._register()


class OnWindowsSignal(object):
    """Stopping criterion that is sensitive to signals Ctrl-C or Ctrl-Break
    on Windows."""

    def __init__(self, sig=None):
        """Return a stopping criterion that stops upon a signal.

        Previous handlers will be overwritten.


        Parameters
        ----------

        sig : signal, optional [default: [0,1]]
            Signal upon which to stop.
            Default encodes signal.SIGINT and signal.SIGBREAK.
        """
        self.sig = [0, 1] if sig is None else sig
        self.stopped = False
        self._register()

    def _register(self):
        import win32api
        win32api.SetConsoleCtrlHandler(self.handler, 1)

    def handler(self, ctrl_type):
        if ctrl_type in self.sig:  # Ctrl-C and Ctrl-Break
            self.stopped = True
            return 1  # don't chain to the next handler
        return 0  # chain to the next handler

    def __call__(self, info):
        res, self.stopped = self.stopped, False
        return res

    def __setstate__(self, dct):
        self.__dict__.update(dct)
        self._register()


OnSignal = OnWindowsSignal if sys.platform == 'win32' else OnUnixSignal


def never(info):
    return False


def always(info):
    return True
