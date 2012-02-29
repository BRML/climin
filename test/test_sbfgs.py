import nose
import itertools

import numpy as np

from climin import Sbfgs 

from losses import Quadratic, LogisticRegression, Rosenbrock


@nose.tools.nottest
def test_sbfgs_quadratic():
    obj = Quadratic()
    opt = Sbfgs(obj.pars, obj.f, obj.fprime)
    for i, info in enumerate(opt):      
        if i > 50:
            break
    assert obj.solved(), 'did not find solution'


def test_sbfgs_rosen():
    obj = Rosenbrock()
    opt = Sbfgs(obj.pars, obj.f, obj.fprime)
    for i, info in enumerate(opt):      
        if i > 20:
            break
    assert obj.solved(), 'did not find solution'


@nose.tools.nottest
def test_sbfgs_lr():
    obj = LogisticRegression()
    args = itertools.repeat(((obj.X, obj.Z), {}))
    opt = Sbfgs(obj.pars, obj.f, obj.fprime, args=args)
    for i, info in enumerate(opt):      
        if i > 50:
            break
    assert obj.solved(), 'did not find solution'
