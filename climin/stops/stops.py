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


import itertools
import signal
import time


class AfterNIterations(object):
    """AfterNIterations class.

    Useful for monitoring thenumber of iterations.

    Internally, the ``n_iter`` field of the climin info dictionary is
    inspected; if the value in there exceeds ``n`` by one, the criterion
    returns ``True``.


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
           Number of iterations after which True is returned.
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
        self.n = n

    def __call__(self, info):
        return info['n_iter'] % self.n == 0


def time_elapsed(sec):
    """Return a stop criterion that stops after `sec` seconds after
    initializing.

    Parameters
    ----------

    sec : float
      Number of seconds until the criterion returns True.


    Returns
    -------

    f : function
      Stopping criterion function.


    Examples
    --------

    >>> stop = S.time_elapsed(.5); stop({})
    False
    >>> time.sleep(0.5)
    >>> stop({})
    True
    """
    start = time.time()

    def inner(info):
        return time.time() - start > sec

    return inner


def converged(func_or_key, n=10, epsilon=1e-5, patience=0):
    """Return a stop criterion that remembers the last `n` values of
    `func_or_key`() and stops if the difference of their maximum and their
    minimum is smaller than `epsilon`.

    `func_or_key` needs to be a callable that returns a scalar value or a
    string which is a key referring to an entry in the info dict it is given.

    If `patience` is non zero, the first `patience` iterations are not checked
    against the criterion.
    """
    ringbuffer = [None for i in xrange(n)]
    counter = itertools.count()

    def inner(info):
        if counter.next() <= patience:
            return False
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


def rising(func_or_key, n=1, epsilon=0, patience=0):
    """Return a stop criterion that remembers the last `n` values obtained via
    `func_or_key` and returns True if the its return value rose at least by
    `epsilon` in the meantime.

    `func_or_key` needs to be either a callable that returns a scalar value or
    a string which is a key referring to an entry in the info dict it is given.

    If `patience` is non zero, the first `patience` iterations are not checked
    against the criterion.
    """
    # TODO explain patience
    results = []
    counter = itertools.count()

    def inner(info):
        if counter.next() <= patience:
            return False
        if isinstance(func_or_key, (str, unicode)):
            val = info[func_or_key]
        else:
            val = func_or_key()
        results.append(val)
        if len(results) < n + 1:
            return False
        if results[-n - 1] + epsilon <= results[-1]:
            return True
        else:
            return False

    return inner


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


def not_better_than_after(minimal, n_iter):
    """Return a stop criterion that returns True if the error is not less than
    `minimal` after `n_iter` iterations."""

    def inner(info):
        return info['n_iter'] > n_iter and info['loss'] >= minimal

    return inner


def patience(func_or_key, initial, grow_factor=1., grow_offset=0.,
             threshold=1e-4):
    """Return a stop criterion inspired by Bengio's patience method.

    The idea is to increase the number of iterations until stopping by
    a multiplicative and/or additive constant once a new best candidate is
    found.

    Parameters
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

    Returns
    -------

    func : callable
        Function that expects a single info dictionary as its only argument.
    """
    if grow_factor == 1 and grow_offset == 0:
        raise ValueError('need to specify either grow_factor != 1'
                         'or grow_offset != 0)')
    # This is in a dict to compensate for Python's lookup of local variables.
    state  = {
        'patience': initial,
        'best_iter': 0,
        'best_loss': float('inf')
    }
    count = itertools.count()

    def inner(info):
        i = info['n_iter']
        if isinstance(func_or_key, (str, unicode)):
            loss = info[func_or_key]
        else:
            loss = func_or_key()

        if loss < state['best_loss']:
            if (state['best_loss'] - loss) > threshold and i > 0:
                state['patience'] = max(i * grow_factor + grow_offset,
                                        state['patience'])
            state['best_iter'] = i
            state['best_loss'] = loss

        return i >= state['patience']

    return inner


class OnSignal(object):
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


def never(info):
    return False


def always(info):
    return True


# For backwards compatibility.
after_n_iterations = AfterNIterations
modulo_n_iterations = ModuloNIterations
any_ = Any
all_ = All

