from __future__ import absolute_import

import itertools

from climin import Asgd

from .losses import Quadratic, LogisticRegression, Rosenbrock


def test_asgd_quadratic():
    obj = Quadratic()
    opt = Asgd(obj.pars, obj.fprime, eta0=0.01, lmbd=0.2, t0=0.01)
    for i, info in enumerate(opt):
        if i > 10000:
            break
    assert obj.solved(0.1), 'did not find solution'


def test_asgd_rosen():
    obj = Rosenbrock()
    opt = Asgd(obj.pars, obj.fprime, eta0=2e-3, t0=1)
    for i, info in enumerate(opt):
        if i > 100000:
            break
    assert ((1 - obj.pars) < 0.01).all(), 'did not find solution'


def test_asgd_lr():
    obj = LogisticRegression()
    args = itertools.repeat(((obj.X, obj.Z), {}))
    opt = Asgd(obj.pars, obj.fprime, eta0=0.2, lmbd=1e-2, t0=0.1, args=args)
    for i, info in enumerate(opt):
        if i > 3000:
            break
    assert obj.solved(0.15), 'did not find solution'
