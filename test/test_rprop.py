import scipy
from scipy.optimize import rosen, rosen_der

from climin import Rprop


quadratic = lambda x: (x**2).sum()
quadraticprime = lambda x: 2 * x


def test_rprop_quadratic():
    dim = 10
    wrt = scipy.random.standard_normal((dim,)) * 10 + 5

    opt = Rprop(wrt, quadratic, quadraticprime, step_shrink=0.1, step_grow=1.2, 
                min_step=1E-6, max_step=0.1)
    for i, info in enumerate(opt):
        if i > 1000:
            break
    assert (abs(wrt) < 0.01).all(), 'did not find solution'


def test_rprop_rosen():
    dim = 2
    wrt = scipy.zeros((dim,))

    opt = Rprop(wrt, rosen, rosen_der, step_shrink=0.5, step_grow=1.1,
            min_step=1E-10, max_step=0.1)
    for i, info in enumerate(opt):
        if i > 10000:
            break
    assert (abs(wrt - [1, 1]) < 0.01).all(), 'did not find solution'
