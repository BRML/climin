import scipy
from scipy.optimize import rosen, rosen_der

from climin import Bfgs
from climin.linesearch import WolfeLineSearch


quadratic = lambda x: (x**2).sum()
quadraticprime = lambda x: 2 * x
quadraticandprime = lambda x: (quadratic(x), quadraticprime(x))


def test_bfgs_rosen():
    dim = 2
    wrt = scipy.zeros((dim,))
    f = lambda: rosen(wrt)
    fprime = lambda: rosen_der(wrt)

    opt = Bfgs(wrt, f, fprime)
    for i, info in enumerate(opt):
        if i > 5000:
            break
    print wrt
    assert (abs(wrt - [1, 1]) < 0.01).all(), 'did not find solution'


def test_bfgs_quadratic():
    dim = 2
    wrt = scipy.array([1., 1.])
    f = lambda: quadratic(wrt)
    fprime = lambda: quadraticprime(wrt)

    opt = Bfgs(wrt, f, fprime)
    for i, info in enumerate(opt):
        print wrt
        if i > 100:
            break
    print wrt
    assert (abs(wrt) < 0.01).all(), 'did not find solution'
