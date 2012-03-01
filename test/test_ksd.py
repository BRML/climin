import nose
import itertools

import numpy as np

from climin import KrylovSubspaceDescent 

from losses import Quadratic, LogisticRegression, Rosenbrock


# There are no more tests here because KSD explicitly constructs the
# Krylov basis and for 2D problems this does not make so much sense.

 
def test_ksd_lr():
    obj = LogisticRegression(seed=20101)
    args = itertools.repeat(((obj.X, obj.Z), {}))
    krylov_args = args
    hessian_args = args
    opt = KrylovSubspaceDescent(
        obj.pars, obj.f, obj.fprime, obj.f_Hp, n_bases=15,
        args=args, krylov_args=krylov_args, hessian_args=hessian_args)
    for i, info in enumerate(opt):      
        if i > 100:
            break
    assert obj.solved(), 'did not find solution'
