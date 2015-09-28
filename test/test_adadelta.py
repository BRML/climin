from __future__ import absolute_import, print_function

import itertools

from climin import Adadelta

from .losses import LogisticRegression
from .common import continuation


def test_adadelta_lr():
    obj = LogisticRegression()
    args = itertools.repeat(((obj.X, obj.Z), {}))
    opt = Adadelta(obj.pars, obj.fprime, 0.9, args=args)
    for i, info in enumerate(opt):
        print(obj.f(opt.wrt, obj.X, obj.Z))
        if i > 3000:
            break
    assert obj.solved(0.15), 'did not find solution'


def test_adadelta_continue():
    obj = LogisticRegression(n_inpt=2, n_classes=2)
    args = itertools.repeat(((obj.X, obj.Z), {}))
    opt = Adadelta(obj.pars, obj.fprime, 0.9, args=args)

    continuation(opt)
