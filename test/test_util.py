# -*- coding: utf-8 -*-

import unittest
import numpy as np

import climin
from climin.util import optimizer, OptimizerDistribution, MinibatchIterator

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
        opt = optimizer(ident, wrt, f=None, fprime=None, f_Hp=None, step_rate=0.1)
        assert isinstance(opt, klass), 'wrong class for %s: %s' % (ident, type(opt))


def test_optimizer_distribution():
    try:
        from sklearn.grid_search import ParameterSampler
    except ImportError:
        raise SkipTest()

    rv = OptimizerDistribution(gd={'step_rate': [.1, .2],
                                   'momentum': [.9, .99]})
    opt = rv.rvs()
    assert opt[0] == 'gd'
    assert opt[1]['step_rate'] in [.1, .2]
    assert opt[1]['momentum'] in [.9, .99]

    rv = OptimizerDistribution(gd={'step_rate': [.1, .2],
                                   'momentum': [.9, .99]},
                               lbfgs={'n_factors': [10, 25]})
    opt = rv.rvs()

    assert opt[0] in ('lbfgs', 'gd')
    if opt[0] == 'lbfgs':
        assert opt[1]['n_factors'] in [10, 25]
    else:
        assert opt[1]['step_rate'] in [.1, .2]
        assert opt[1]['momentum'] in [.9, .99]


class MinibatchTest(unittest.TestCase):

    def setUp(self):
        self.D = np.random.random((13, 5))

    def test_minibatch_size(self):
        """Test if minibatches are correctly generated if given a size."""
        batches = MinibatchIterator(self.D, batch_size=5)
        self.assertEqual(batches[0].shape[0], 5)
        self.assertEqual(batches[1].shape[0], 5)
        self.assertEqual(batches[2].shape[0], 3)
        self.assertEqual(len(batches), 3)
