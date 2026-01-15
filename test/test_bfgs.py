import pytest
import itertools

from climin import Bfgs

from .losses import Quadratic, LogisticRegression, Rosenbrock


@pytest.mark.skip
def test_bfgs_quadratic():
    obj = Quadratic()
    opt = Bfgs(obj.pars, obj.f, obj.fprime)
    for i, info in enumerate(opt):
        if i > 50:
            break
    assert obj.solved(), 'did not find solution'


def test_bfgs_rosen():
    obj = Rosenbrock()
    opt = Bfgs(obj.pars, obj.f, obj.fprime)
    for i, info in enumerate(opt):
        if i > 50:
            break
    assert obj.solved(), 'did not find solution'


@pytest.mark.skip
def test_bfgs_lr():
    obj = LogisticRegression()
    args = itertools.repeat(((obj.X, obj.Z), {}))
    opt = Bfgs(obj.pars, obj.f, obj.fprime, args=args)
    for i, info in enumerate(opt):
        if i > 50:
            break
    assert obj.solved(), 'did not find solution'
