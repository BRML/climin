import itertools

import nose
import numpy as np

from climin import ASGD

from losses import Quadratic, LogisticRegression, Rosenbrock


def test_asgd_quadratic():
    obj = Quadratic()
    opt = ASGD(obj.pars, obj.fprime, eta0=0.04, lmbd=0.5, t0=1)
    for i, info in enumerate(opt):
        if i > 225:
            break
    assert obj.solved(), 'did not find solution'


def test_asgd_rosen():
    obj = Rosenbrock()
    opt = ASGD(obj.pars, obj.fprime, eta0=2e-3, t0=1)
    for i, info in enumerate(opt):      
        if i > 100000:
            break
    assert ((1 - obj.pars) < 0.01).all(), 'did not find solution'


def test_asgd_lr():
    obj = LogisticRegression()
    args = itertools.repeat(((obj.X, obj.Z), {}))
    opt = ASGD(obj.pars, obj.fprime, eta0=1, lmbd=1e-5, t0=5, args=args)
    for i, info in enumerate(opt):      
        if i > 600:
            break
    assert obj.solved(), 'did not find solution'
