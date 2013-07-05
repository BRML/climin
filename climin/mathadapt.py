"""This module provides functionality which is usable for
coding towards gnumpy and numpy: the idea is to avoid if clauses
in the optimizer code.

"""

import numpy as np
import gnumpy as gp
import theano

GPU = theano.config.device == 'gpu'

def sqrt(x):
    """Return an array of the same shape containing the element square
    root of `x`."""
    return x**(0.5)


def zeros(shape):
    """Return a new array of given shape and type, filled with zeros."""
    if GPU:
        return gp.zeros(shape)
    else:
        return np.zeros(shape)


def zero_like(x):
    """Return an array of the same shape as `x` containing only zeros."""
    return x * 0.


def ones_like(x):
    """Return an array of the same shape as `x` containing only ones."""
    return x * 0. + 1.


def clip(a, a_min, a_max):
    """Clip (limit) the values in an array.

    Given an interval, values outside the interval are clipped to the interval 
    edges. For example, if an interval of [0, 1] is specified, values smaller 
    than 0 become 0, and values larger than 1 become 1."""
    if isinstance(a, gp.garray):
        max_mask = (a > a_max)
        max_tar = gp.ones(a.shape) * a_max
        min_mask = (a < a_min)
        min_tar = gp.ones(a.shape) * a_min
        a_clipped = a*(1-max_mask-min_mask) + max_tar*max_mask + min_tar*min_mask
        return a_clipped
    else:
        return np.clip(a, a_min, a_max)


def sign(x):
    """Returns an element-wise indication of the sign of a number."""
    if isinstance(x, gp.garray):
        return gp.sign(x)
    else:
        return np.sign(x)



class random(object):
    """Random sampling"""

    @staticmethod
    def random(shape):
        """Return random floats in the half-open interval [0.0, 1.0)."""
        if GPU:
            return gp.rand(shape)
        else:
            return np.random.random(shape)



