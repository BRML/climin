# -*- coding: utf-8 -*-

import itertools
import collections

import numpy as np


def repeat_or_iter(obj):
    try:
        return iter(obj)
    except TypeError:
        return itertools.repeat(obj)


class Minimizer(object):

    def __init__(self, wrt, args=None):
        self.wrt = wrt
        if args is None:
            self.args = itertools.repeat(([], {}))
        else:
            self.args = args

        self.n_iter = 0

    def set_from_info(self, info):
        """Populate the fields of this object with the corresponding fields of
        a dictionary.

        Parameters
        ----------

        info : dict
            Has to contain a key for each of the objects in the
            ``.state_fields`` list. The field will be set according to the entry
            in the dictionary.
        """
        for f in self.state_fields:
            self.__dict__[f] = info[f]

    def extended_info(self, **kw):
        """Return a dictionary populated with the values of the state fields.
        Further values can be given as keyword arguments.

        Parameters
        ----------

        **kw : dict
            Arbitrary data to place into dictionary.

        Returns
        -------

        dct : dict
            Contains all attributes of the class given by the ``state_fields``
            attribute. Additionally updated with elements from ``kw``.
        """
        dct = dict((f, getattr(self, f)) for f in self.state_fields)
        dct.update(kw)
        return dct

    def minimize_until(self, criterions):
        """Minimize until one of the supplied `criterions` is met.

        Each criterion is a callable that, given the info object yielded by
        an optimizer, returns a boolean indicating whether to stop. False means
        to continue, True means to stop."""
        if not criterions:
            raise ValueError('need to supply at least one criterion')

        # if criterions is a single criterion, wrap it in iterable list
        if not isinstance(criterions, collections.Iterable):
            criterions = [criterions]

        info = {}
        for info in self:
            for criterion in criterions:
                if criterion(info):
                    return info
        return info

    def __iter__(self):
        for info in self._iterate():
            yield self.extended_info(**info)


def is_nonzerofinite(arr):
    """Return True if the array is neither zero, NaN or infinite."""
    return (arr != 0).any() and np.isfinite(arr).all()
