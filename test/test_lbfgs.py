import scipy
from scipy.optimize import rosen, rosen_der

from climin.lbfgs import Lbfgs 


quadratic = lambda x: (x**2).sum()
quadraticprime = lambda x: 2 * x
quadraticandprime = lambda x: (quadratic(x), quadraticprime(x))


def test_lbfgs_rosen():
    dim = 2
    wrt = scipy.zeros((dim,))

    opt = Lbfgs(wrt, rosen, rosen_der)
    for i, info in enumerate(opt):
        if (abs(wrt - [1, 1]) < 0.01).all():
            success = True
            break
        if i >= 500:
            success = False
            break
    assert success, 'did not find solution'


def test_lbfgs_quadratic():
    dim = 2
    wrt = scipy.random.standard_normal((dim,)) * 10 + 5

    opt = Lbfgs(wrt, quadratic, quadraticprime)
    for i, info in enumerate(opt):
        if i > 100:
            break
    assert (abs(wrt) < 0.01).all(), 'did not find solution'
