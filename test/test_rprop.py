from __future__ import absolute_import

import itertools

from climin import Rprop

from .losses import Quadratic, LogisticRegression, Rosenbrock
from .common import continuation


def test_rprop_quadratic():
    obj = Quadratic()
    opt = Rprop(obj.pars, obj.fprime, step_shrink=0.1, step_grow=1.2,
                min_step=1e-6, max_step=0.1)
    for i, info in enumerate(opt):
        if i > 500:
            break
    assert obj.solved(), 'did not find solution'


def test_rprop_rosen():
    obj = Rosenbrock()
    opt = Rprop(obj.pars, obj.fprime, step_shrink=0.6, step_grow=1.2,
                min_step=1e-8, max_step=1.)
    for i, info in enumerate(opt):
        if i > 2000:
            break
    assert obj.solved(), 'did not find solution'


def test_rprop_lr():
    obj = LogisticRegression()
    args = itertools.repeat(((obj.X, obj.Z), {}))
    opt = Rprop(obj.pars, obj.fprime, step_shrink=0.5, step_grow=1.2,
                min_step=1e-6, max_step=0.1, args=args)
    for i, info in enumerate(opt):
        if i > 500:
            break
    assert obj.solved(), 'did not find solution'

def test_rprop_continue():
    obj = LogisticRegression(n_inpt=2, n_classes=2)
    args = itertools.repeat(((obj.X, obj.Z), {}))
    opt = Rprop(obj.pars, obj.fprime, step_shrink=0.1, step_grow=1.2,
                min_step=1e-6, max_step=0.1, args=args)

    continuation(opt)
