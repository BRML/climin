
import scipy
import numpy as np

from climin import ConjugateGradient

from losses import Quadratic



def test_cg_explicit_hessian():
    obj = Quadratic()
    opt = ConjugateGradient(obj.pars, obj.H, obj.b)
    for i, info in enumerate(opt):
        if i > 10:
            break
    assert obj.solved()


def test_cg_implicit_hessian():
    obj = Quadratic()
    def f_Hp(p):
        return obj.f_Hp(obj.pars, p)
    opt = ConjugateGradient(obj.pars, f_Hp=f_Hp, b=obj.b)
    for i, info in enumerate(opt):
        if i > 10:
            break
    assert obj.solved()


def test_cg_preconditioning():
    obj = Quadratic()
    precond = np.array([[1,0], [0, 1e-2]])
    opt = ConjugateGradient(obj.pars, obj.H, obj.b, precond = precond)
    for i, info in enumerate(opt):
        if i > 10:
            break
    assert obj.solved()
