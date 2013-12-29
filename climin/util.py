# -*- coding: utf-8 -*-


import inspect
import itertools
import random
import warnings

import numpy as np

from gd import GradientDescent
from bfgs import Lbfgs
from cg import NonlinearConjugateGradient
from rprop import Rprop
from rmsprop import RmsProp

try:
    from sklearn.grid_search import ParameterSampler
except ImportError:
    pass


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
        'gd': GradientDescent,
        'lbfgs': Lbfgs,
        'ncg': NonlinearConjugateGradient,
        'rprop': Rprop,
        'rmsprop': RmsProp,
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


def shaped_from_flat(flat, shapes):
    """Given a one dimensional array ``flat``, return a list of views of shapes
    ``shapes`` on that array.

    Each view will point to a distinct memory region, consecutively allocated
    in flat.

    Parameters
    ----------

    flat : array_like
        Array of one dimension.

    shapes : list of tuples of ints
        Each entry of this list specifies the shape of the corresponding view
        into ``flat``.

    Returns
    -------

    views : list of arrays
        Each entry has the shape given in ``shapes`` and points as a view into
        ``flat``.
    """
    shapes = [(i,) if isinstance(i, int) else i for i in shapes]
    sizes = [np.prod(i) for i in shapes]

    n_used = 0
    views = []
    for size, shape in zip(sizes, shapes):
        this = flat[n_used:n_used + size]
        n_used += size
        this.shape = shape
        views.append(this)

    return views


def empty_with_views(shapes, empty_func=np.empty):
    """Create an array and views shaped according to ``shapes``.

    The ``shapes`` parameter is a list of tuples of ints.  Each tuple
    represents a desired shape for an array which will be allocated in a bigger
    memory region. This memory region will be represented by an array as well.

    For example, the shape speciciation ``[2, (3, 2)]`` will create an array
    ``flat`` of size 8. The first view will have a size of ``(2,)`` and point
    to the first two entries, i.e. ``flat`[:2]`, while the second array will
    have a shape of ``(3, 2)`` and point to the elements ``flat[2:8]``.


    Parameters
    ----------

    spec : list of tuples of ints
        Specification of the desired shapes.

    empty_func : callable
        function that returns a memory region given an integer of the desired
        size. (Examples include ``numpy.empty``, which is the default,
        ``gnumpy.empty`` and ``theano.tensor.empty``.


    Returns
    -------

    flat : array_like (depending on ``empty_func``)
        Memory region containing all the views.

    views : list of array_like
        Variable number of results. Each contains a view into the array
        ``flat``.


    Examples
    --------

    >>> from climin.util import empty_with_views
    >>> flat, (w, b) = empty_with_views([(3, 2), 2])
    >>> w[...] = 1
    >>> b[...] = 2
    >>> flat
    array([ 1.,  1.,  1.,  1.,  1.,  1.,  2.,  2.])
    >>> flat[0] = 3
    >>> w
    array([[ 3.,  1.],
           [ 1.,  1.],
           [ 1.,  1.]])

    """
    shapes = [(i,) if isinstance(i, int) else i for i in shapes]
    sizes = [np.prod(i) for i in shapes]
    n_pars = sum(sizes)
    flat = empty_func(n_pars)

    views = shaped_from_flat(flat, shapes)

    return flat, views


def minibatches(arr, batch_size, d=0):
    """Return a list of views of the given arr.

    Each view represents a mini bach of the data.

    Parameters
    ----------

    arr : array_like
        Array to obtain batches from. Practically, anything that accepts slicing
        can be used.

    batch_size : int
        Size of a batch. Last batch might be smaller if ``batch_size`` is not a
        divisor of ``arr``.

    d : int, optional, default: 0
        Dimension along which the data samples are separated and thus slicing
        should be done.

    Returns
    -------

    mini_batches : list
        Each item of the list is a view of ``arr``. Views are ordered.
    """
    n_batches, rest = divmod(arr.shape[d], batch_size)
    if rest != 0:
        n_batches += 1

    slices = (slice(i * batch_size, (i + 1) * batch_size)
              for i in range(n_batches))
    if d == 0:
        res = [arr[i] for i in slices]
    elif d == 1:
        res = [arr[:, i] for i in slices]
    elif d == 2:
        res = [arr[:, :, i] for i in slices]

    return res


def iter_minibatches(lst, batch_size, dims, n_cycles=False):
    """Return an iterator that successively yields tuples containing aligned
    minibatches of size `batch_size` from slicable objects given in `lst`, in
    random order without replacement.

    Because different containers might require slicing over different
    dimensions, the dimension of each container has to be givens as a list
    `dims`.


    Parameters
    ----------

    lst : list of array_like
        Each item of the list will be sliced into mini batches in alignemnt with
        the others.

    batch_size : int
        Size of each batch. Last batch might be smaller.

    dims : list
        Aligned with ``lst``, gives the dimension along which the data samples
        are separated.

    n_cycles : int or False, optional [default: False]
        Number of cycles after which to stop the iterator. If ``False``, will
        yield forever.


    Returns
    -------

    batches : iterator
        Infinite iterator of mini batches in random order (without
        replacement).
    """
    batches = [minibatches(i, batch_size, d) for i, d in zip(lst, dims)]
    if len(batches) > 1:
        if any(len(i) != len(batches[0]) for i in batches[1:]):
            raise ValueError("containers to be batched have different lengths")
    counter = itertools.count()
    while True:
        indices = [i for i, _ in enumerate(batches[0])]
        while True:
            random.shuffle(indices)
            for i in indices:
                yield tuple(b[i] for b in batches)
            count = counter.next()
            if n_cycles and count >= n_cycles:
                raise StopIteration()


class OptimizerDistribution(object):
    """OptimizerDistribution class.

    Can be used for specifying optimizers in scikit-learn's randomized parameter
    search.

    Attributes
    ----------

    options : dict
        Maps an optimizer key to a grid to sample from.
    """

    def __init__(self, **options):
        """Create an OptimizerDistribution object.

        Parameters
        ----------

        options : dict
            Maps an optimizer key to a grid to sample from.
        """
        self.options = options

    def rvs(self):
        opt = random.choice(self.options.keys())
        grid = self.options[opt]
        sample = list(ParameterSampler(grid, n_iter=1))[0]
        return opt, sample
