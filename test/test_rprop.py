import scipy
from scipy.optimize import rosen, rosen_der

from climin import Rprop


quadratic = lambda x: (x**2).sum()
quadraticprime = lambda x: 2 * x
quadraticandprime = lambda x: (quadratic(x), quadraticprime(x))


def test_rprop_quadratic():
    dim = 10
    wrt = scipy.random.standard_normal((dim,)) * 10 + 5

    opt = Rprop(wrt, quadraticandprime, stepshrink=0.1, stepgrow=1.2, 
                minstep=1E-6, maxstep=0.1)
    for i, info in enumerate(opt):
        if i > 1000:
            break
    assert (abs(wrt) < 0.01).all(), 'did not find solution'


def test_rprop_rosen():
    dim = 2
    wrt = scipy.zeros((dim,))
    rosenandprime = lambda x: (rosen(x), rosen_der(x))

    opt = Rprop(wrt, rosenandprime, stepshrink=0.5, stepgrow=1.1,
            minstep=1E-10, maxstep=0.1)
    for i, info in enumerate(opt):
        if i > 10000:
            break
    print wrt
    assert (abs(wrt - [1, 1]) < 0.01).all(), 'did not find solution'
