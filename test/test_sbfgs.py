import scipy
from scipy.optimize import rosen, rosen_der

from climin import SBfgs
from climin.linesearch import WolfeLineSearch


quadratic = lambda x: (x**2).sum()
quadraticprime = lambda x: 2 * x
quadraticandprime = lambda x: (quadratic(x), quadraticprime(x))


def test_sbfgs_rosen():
    dim = 2
    wrt = scipy.zeros((dim,))

    opt = SBfgs(wrt, rosen, rosen_der)
    for i, info in enumerate(opt):
        if i > 20:
            break
    assert (abs(wrt - [1, 1]) < 0.01).all(), 'did not find solution'


def test_sbfgs_quadratic():
    dim = 2
    wrt = scipy.array([1., 1.])

    opt = SBfgs(wrt, quadratic, quadraticprime)
    for i, info in enumerate(opt):
        if i > 100:
            break
    assert (abs(wrt) < 0.01).all(), 'did not find solution'
