import itertools

import nose
import numpy as np

from climin import Ilne

from losses import Quadratic, LogisticRegression, Rosenbrock


def test_ilne_quadratic():
    obj = Quadratic()
    opt = Ilne(
        obj.pars, obj.fprime, steprate=0.01, momentum=.9)
    for i, info in enumerate(opt):
        if i > 500:
            break
    assert obj.solved(), 'did not find solution'


def test_ilne_rosen():
    obj = Rosenbrock()
    opt = Ilne(
        obj.pars, obj.fprime, steprate=0.001, momentum=.99)
    for i, info in enumerate(opt):
        if i > 5000:
            break
    assert ((1 - obj.pars) < 0.01).all(), 'did not find solution'


def test_ilne_lr():
    obj = LogisticRegression()
    args = itertools.repeat(((obj.X, obj.Z), {}))
    opt = Ilne(
        obj.pars, obj.fprime, steprate=0.01, momentum=.9, args=args)
    for i, info in enumerate(opt):
        if i > 500:
            break
    assert obj.solved(), 'did not find solution'
