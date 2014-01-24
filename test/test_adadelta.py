import itertools

import nose
import numpy as np

from climin import Adadelta

from losses import Quadratic, LogisticRegression, Rosenbrock


def test_adadelta_lr():
    obj = LogisticRegression()
    args = itertools.repeat(((obj.X, obj.Z), {}))
    opt = Adadelta(obj.pars, obj.fprime, 0.9, args=args)
    for i, info in enumerate(opt):
        print obj.f(opt.wrt, obj.X, obj.Z)
        if i > 3000:
            break
    assert obj.solved(0.15), 'did not find solution'
