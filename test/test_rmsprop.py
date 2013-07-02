import itertools

import nose
import numpy as np

from climin import RmsProp

from losses import Quadratic, LogisticRegression, Rosenbrock


def test_rmsprop_lr():
    obj = LogisticRegression()
    args = itertools.repeat(((obj.X, obj.Z), {}))
    opt = RmsProp(obj.pars, obj.fprime, 0.01, 0.9, args=args)
    for i, info in enumerate(opt):
        print obj.f(opt.wrt, obj.X, obj.Z)
        if i > 3000:
            break
    assert obj.solved(0.15), 'did not find solution'
