import itertools

import numpy as np

from climin import Rprop

from losses import Quadratic, LogisticRegression, Rosenbrock


def test_rprop_quadratic():
    obj = Quadratic()
    opt = Rprop(obj.pars, obj.f, obj.fprime, step_shrink=0.1, step_grow=1.2,
                min_step=1e-6, max_step=0.1)
    for i, info in enumerate(opt):      
        if i > 500:
            break
    assert obj.solved(), 'did not find solution'


def test_rprop_rosen():
    obj = Rosenbrock()
    opt = Rprop(obj.pars, obj.f, obj.fprime, step_shrink=0.1, step_grow=1.2,
                min_step=1e-6, max_step=0.1)
    for i, info in enumerate(opt):      
        if i > 5000:
            break
    assert obj.solved(), 'did not find solution'


def test_rprop_lr():
    obj = LogisticRegression()
    args = itertools.repeat(((obj.X, obj.Z), {}))
    opt = Rprop(obj.pars, obj.f, obj.fprime, step_shrink=0.1, step_grow=1.2,
                min_step=1e-6, max_step=0.1, args=args)
    for i, info in enumerate(opt):      
        if i > 500:
            break
    assert obj.solved(), 'did not find solution'
