# -*- coding: utf-8 -*-

import numpy as np

import climin
from climin.util import optimizer, OptimizerDistribution

from nose.plugins.skip import SkipTest


def test_optimizer():
    pairs = [('gd', climin.GradientDescent),
             ('lbfgs', climin.Lbfgs),
             ('ncg', climin.NonlinearConjugateGradient),
             ('rprop', climin.Rprop),
             ('rmsprop', climin.RmsProp),
             ]

    for ident, klass in pairs:
        wrt = np.zeros(10)
        opt = optimizer(ident, wrt, f=None, fprime=None, f_Hp=None, steprate=0.1)
        assert isinstance(opt, klass), 'wrong class for %s: %s' % (ident, type(opt))



def test_optimizer_distribution():
    try:
        import sklearn
        from sklearn.grid_search import ParameterSampler # not available on sklearn < 0.14.
    except ImportError:
        raise SkipTest()
    rv = OptimizerDistribution(gd={'steprate': [.1, .2],
                                   'momentum': [.9, .99]})
    opt = rv.rvs()
    assert opt[0] == 'gd'
    assert opt[1]['steprate'] in [.1, .2]
    assert opt[1]['momentum'] in [.9, .99]

    rv = OptimizerDistribution(gd={'steprate': [.1, .2],
                                   'momentum': [.9, .99]},
                               lbfgs={'n_factors': [10, 25]})
    opt = rv.rvs()

    assert opt[0] in ('lbfgs', 'gd')
    if opt[0] == 'lbfgs':
        assert opt[1]['n_factors'] in [10, 25]
    else:
        assert opt[1]['steprate'] in [.1, .2]
        assert opt[1]['momentum'] in [.9, .99]
