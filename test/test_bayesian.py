import itertools

import nose
import numpy as np

from climin import Bayesian 

from losses import Quadratic


def test_bayesian_quadratic():
    obj = Quadratic()
    obj.H = np.eye(2)
    obj.b = np.zeros(2)

    x0s = np.asarray([
        [-1., 1.],
        [-1.2, 1.4],
        [-0.5, -0.3],
        [0.3, 0.3], 
        [1.2, 1.3],
    ])

    opt = Bayesian(obj.pars, obj.f, x0s, n_inner_iters=100)
    for i, info in enumerate(opt):      
        if i > 50:
            break

    assert obj.solved(0.1), 'did not find solution'
