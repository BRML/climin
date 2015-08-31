from __future__ import absolute_import

import nose
import itertools

from climin import Bfgs

from .losses import Quadratic, LogisticRegression, Rosenbrock


@nose.tools.nottest
def test_bfgs_quadratic():
    obj = Quadratic()
    opt = Bfgs(obj.pars, obj.f, obj.fprime)
    for i, info in enumerate(opt):
        if i > 50:
            break
    assert obj.solved(), 'did not find solution'


def test_bfgs_rosen():
    obj = Rosenbrock()
    opt = Bfgs(obj.pars, obj.f, obj.fprime)
    for i, info in enumerate(opt):
        if i > 50:
            break
    assert obj.solved(), 'did not find solution'


@nose.tools.nottest
def test_bfgs_lr():
    obj = LogisticRegression()
    args = itertools.repeat(((obj.X, obj.Z), {}))
    opt = Bfgs(obj.pars, obj.f, obj.fprime, args=args)
    for i, info in enumerate(opt):
        if i > 50:
            break
    assert obj.solved(), 'did not find solution'
