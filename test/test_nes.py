from __future__ import absolute_import

import nose
import itertools

from climin import Xnes

from .losses import Quadratic, LogisticRegression, Rosenbrock


@nose.tools.nottest
def test_xnes_quadratic():
    obj = Quadratic()
    opt = Xnes(obj.pars, obj.f)
    for i, info in enumerate(opt):
        if i > 5000:
            break
    assert obj.solved(), 'did not find solution'


def test_xnes_rosen():
    obj = Rosenbrock()
    opt = Xnes(obj.pars, obj.f)
    for i, info in enumerate(opt):
        if i > 10000:
            break
    assert obj.solved(0.3), 'did not find solution'


@nose.tools.nottest
def test_xnes_lr():
    obj = LogisticRegression(seed=10101)
    args = itertools.repeat(((obj.X, obj.Z), {}))
    opt = Xnes(obj.pars, obj.f, args=args)
    for i, info in enumerate(opt):
        if i > 100:
            break
    assert obj.solved(), 'did not find solution'
