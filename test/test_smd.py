import nose
import itertools

import numpy as np

from climin import SMD

from losses import Quadratic, LogisticRegression, Rosenbrock


def test_smd_quadratic():
    obj = Quadratic()
    opt = SMD(obj.pars, obj.f, obj.fprime, obj.f_Hp, eta0=1e-3)
    for i, info in enumerate(opt):      
        if i > 750:
            break
    assert obj.solved(), 'did not find solution'


def test_smd_rosen():
    obj = Rosenbrock()
    opt = SMD(obj.pars, obj.f, obj.fprime, obj.f_Hp, eta0=2e-3, lmbd=0.9)
    for i, info in enumerate(opt):      
        if i > 5000:
            break
    assert ((1 - obj.pars) < 0.01).all(), 'did not find solution'


def test_smd_lr():
    obj = LogisticRegression(seed=10101)
    args = itertools.repeat(((obj.X, obj.Z), {}))
    opt = SMD(obj.pars, obj.f, obj.fprime, obj.f_Hp, args=args, eta0=0.1)
    for i, info in enumerate(opt):      
        if i > 150:
            break
    assert obj.solved(), 'did not find solution'
