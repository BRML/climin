import itertools

import nose
import numpy as np

from climin import Nesterov

from losses import Quadratic, LogisticRegression, Rosenbrock


def test_nesterov_quadratic():
    obj = Quadratic()
    opt = Nesterov(obj.pars, obj.fprime, steprate=0.01)
    for i, info in enumerate(opt):      
        if i > 750:
            break
    assert obj.solved(), 'did not find solution'


def test_nesterov_rosen():
    obj = Rosenbrock()
    opt = Nesterov(obj.pars, obj.fprime, steprate=2e-3)
    for i, info in enumerate(opt):
        if i > 7500:
            break
    assert ((1 - obj.pars) < 0.01).all(), 'did not find solution'


def test_nesterov_lr():
    obj = LogisticRegression()
    args = itertools.repeat(((obj.X, obj.Z), {}))
    opt = Nesterov(obj.pars, obj.fprime, steprate=0.1, args=args)
    for i, info in enumerate(opt):      
        if i > 750:
            break
    assert obj.solved(), 'did not find solution'
