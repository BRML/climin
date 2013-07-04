# -*- coding: utf-8 -*-

import numpy as np

import climin
from climin.util import optimizer


def test_optimizer():
    pairs = [('asgd', climin.Asgd),
             ('gd', climin.GradientDescent),
             ('lbfgs', climin.Lbfgs),
             ('ncg', climin.NonlinearConjugateGradient),
             ('rprop', climin.Rprop),
             ('smd', climin.Smd)]

    for ident, klass in pairs:
        wrt = np.zeros(10)
        opt = optimizer(ident, wrt, f=None, fprime=None, f_Hp=None)
        assert isinstance(opt, klass), 'wrong class for %s: %s' % (ident, type(opt))
