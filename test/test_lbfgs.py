import scipy
from scipy.optimize import rosen, rosen_der

from climin.lbfgs import Lbfgs 


quadratic = lambda x: (x**2).sum()
quadraticprime = lambda x: 2 * x
quadraticandprime = lambda x: (quadratic(x), quadraticprime(x))


def test_lbfgs_rosen():
    dim = 2
    wrt = scipy.zeros((dim,))
    f = lambda: rosen(wrt)
    fprime = lambda: rosen_der(wrt)

    opt = Lbfgs(wrt, f, fprime)
    for i, info in enumerate(opt):
        if i > 2000:
            break
    print wrt
    assert (abs(wrt - [1, 1]) < 0.01).all(), 'did not find solution'


def test_lbfgs_quadratic():
    dim = 2
    wrt = scipy.random.standard_normal((dim,)) * 10 + 5
    f = lambda: quadratic(wrt)
    fprime = lambda: quadraticprime(wrt)

    opt = Lbfgs(wrt, f, fprime)
    for i, info in enumerate(opt):
        if i > 100:
            break
    print wrt
    assert (abs(wrt) < 0.01).all(), 'did not find solution'
