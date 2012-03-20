import nose
import itertools

import numpy as np

from climin import HessianFree

from losses import Quadratic, LogisticRegression, Rosenbrock


def test_hf_quadratic():
    obj = Quadratic()
    opt = HessianFree(obj.pars, obj.f, obj.fprime, obj.f_Hp)
    for i, info in enumerate(opt):      
        if i > 50:
            break
    assert obj.solved(), 'did not find solution'


def test_hf_rosen():
    obj = Rosenbrock()
    opt = HessianFree(obj.pars, obj.f, obj.fprime, obj.f_Hp)
    for i, info in enumerate(opt):      
        if i > 50:
            break
    assert obj.solved(), 'did not find solution'


def test_hf_lr():
    obj = LogisticRegression(seed=1010)
    args = itertools.repeat(((obj.X, obj.Z), {}))
    opt = HessianFree(obj.pars, obj.f, obj.fprime, obj.f_Hp, args=args)
    for i, info in enumerate(opt):
        print(info["loss"])
        if i > 100:
            break
    assert obj.solved(), 'did not find solution'
