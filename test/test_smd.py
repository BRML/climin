from __future__ import absolute_import, print_function

import itertools

from climin import Smd

from .losses import Quadratic, LogisticRegression, Rosenbrock


def test_smd_quadratic():
    obj = Quadratic()
    # TODO: I don't know why these parameters work, but they do.
    opt = Smd(obj.pars, obj.f, obj.fprime, obj.f_Hp, eta0=1e-1,
              mu=2e-4, lmbd=.5)
    for i, info in enumerate(opt):
        print(obj.pars)
        if i > 100:
            break
    assert obj.solved(), 'did not find solution'


def test_smd_rosen():
    obj = Rosenbrock()
    opt = Smd(obj.pars, obj.f, obj.fprime, obj.f_Hp, eta0=2e-3, lmbd=0.9)
    for i, info in enumerate(opt):
        if i > 5000:
            break
    assert ((1 - obj.pars) < 0.01).all(), 'did not find solution'


def test_smd_lr():
    obj = LogisticRegression(seed=10101)
    args = itertools.repeat(((obj.X, obj.Z), {}))
    opt = Smd(obj.pars, obj.f, obj.fprime, obj.f_Hp, args=args, eta0=0.1)
    for i, info in enumerate(opt):
        if i > 150:
            break
    assert obj.solved(), 'did not find solution'
