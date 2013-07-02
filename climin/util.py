# -*- coding: utf-8 -*-


import inspect
import random
import warnings

from asgd import Asgd
from gd import GradientDescent
from lbfgs import Lbfgs
from ncg import NonlinearConjugateGradient
from rprop import Rprop
from rmsprop import RmsProp
from smd import Smd


def coroutine(f):
    """Turn a generator function into a coroutine by calling .next() once."""
    def started(*args, **kwargs):
        cr = f(*args, **kwargs)
        cr.next()
        return cr
    return started


def aslist(item):
    if not isinstance(item, (list, tuple)):
        item = [item]
    return item


def mini_slices(n_samples, batch_size):
    """Yield slices of size `batch_size` that work with a container of length
    `n_samples`."""
    n_batches, rest = divmod(n_samples, batch_size)
    if rest != 0:
        n_batches += 1

    return [slice(i * batch_size, (i + 1) * batch_size) for i in range(n_batches)]


def draw_mini_slices(n_samples, batch_size, with_replacement=False):
    slices = mini_slices(n_samples, batch_size)
    idxs = range(len(slices))

    if with_replacement:
        yield random.choice(slices)
    else:
        while True:
            random.shuffle(idxs)
            for i in idxs:
                yield slices[i]


def draw_mini_indices(n_samples, batch_size):
    assert n_samples > batch_size
    idxs = range(n_samples)
    random.shuffle(idxs)
    pos = 0

    while True:
        while pos + batch_size <= n_samples:
             yield idxs[pos:pos + batch_size]
             pos += batch_size
        
        batch = idxs[pos:]
        needed = batch_size - len(batch)
        random.shuffle(idxs)
        batch += idxs[0:needed]
        yield batch
        pos = needed

        

def optimizer(identifier, wrt, *args, **kwargs):
    """Return an optimizer with the desired configuration.

    This is a convenience function if one wants to try out different optimizers
    but wants to change as little code as possible.

    Additional arguments and keyword arguments will be passed to the constructor
    of the class. If the found class does not take the arguments supplied, this
    will `not` throw an error, but pass silently.

    :param identifier: String identifying the optimizer to use. Can be either
        ``asgd``, ``gd``, ``lbfgs``, ``ncg``, ``rprop`` or  ``smd``.
    :param wrt: Numpy array pointing to the data to optimize.
    """
    klass_map = {
        'asgd': Asgd,
        'gd': GradientDescent,
        'lbfgs': Lbfgs,
        'ncg': NonlinearConjugateGradient,
        'rprop': Rprop,
        'rmsprop': RmsProp,
        'smd': Smd,
    }
    # Find out which arguments to pass on.
    klass = klass_map[identifier]
    argspec = inspect.getargspec(klass.__init__)
    if argspec.keywords is None:
        # Issue a warning for each of the arguments that have been passed
        # to this optimizer but were not used.
        expected_keys = set(argspec.args)
        given_keys = set(kwargs.keys())
        unused_keys = given_keys - expected_keys
        for i in unused_keys:
            warnings.warn('Argument named %s is not expected by %s'
                          % (i, klass))

        # We need to filter stuff out.
        used_keys = expected_keys & given_keys
        kwargs = dict((k, kwargs[k]) for k in used_keys)
    try:
        opt = klass(wrt, *args, **kwargs)
    except TypeError:
        raise TypeError('required arguments for %s: %s' % (klass, argspec.args))

    return opt

