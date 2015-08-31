from __future__ import absolute_import

import itertools

from climin import Lbfgs

from .losses import Quadratic, LogisticRegression, Rosenbrock


def test_lbfgs_quadratic():
    obj = Quadratic()
    opt = Lbfgs(obj.pars, obj.f, obj.fprime)
    for i, info in enumerate(opt):
        if i > 50:
            break
    assert obj.solved(), 'did not find solution'


def test_lbfgs_rosen():
    obj = Rosenbrock()
    opt = Lbfgs(obj.pars, obj.f, obj.fprime)
    for i, info in enumerate(opt):
        if i > 50:
            break
    assert obj.solved(), 'did not find solution'


def test_lbfgs_lr():
    obj = LogisticRegression(seed=10101)
    args = itertools.repeat(((obj.X, obj.Z), {}))
    opt = Lbfgs(obj.pars, obj.f, obj.fprime, args=args)
    for i, info in enumerate(opt):
        if i > 100:
            break
    assert obj.solved(), 'did not find solution'
