from __future__ import absolute_import

import itertools

from climin import NonlinearConjugateGradient

from .losses import Quadratic, LogisticRegression, Rosenbrock


def test_ncg_quadratic():
    obj = Quadratic()
    opt = NonlinearConjugateGradient(obj.pars, obj.f, obj.fprime)
    for i, info in enumerate(opt):
        if i > 50:
            break
    assert obj.solved(), 'did not find solution'


def test_ncg_rosen():
    obj = Rosenbrock()
    opt = NonlinearConjugateGradient(obj.pars, obj.f, obj.fprime)
    for i, info in enumerate(opt):
        if i > 14:
            break
    assert obj.solved(), 'did not find solution'


def test_ncg_lr():
    obj = LogisticRegression()
    args = itertools.repeat(((obj.X, obj.Z), {}))
    opt = NonlinearConjugateGradient(obj.pars, obj.f, obj.fprime, args=args)
    for i, info in enumerate(opt):
        if i > 50:
            break
    assert obj.solved(), 'did not find solution'
