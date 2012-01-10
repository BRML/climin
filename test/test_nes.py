import scipy
from scipy.optimize import rosen, rosen_der

from climin.nes import Xnes


quadratic = lambda x: (x**2).sum()
quadraticprime = lambda x: 2 * x
quadraticandprime = lambda x: (quadratic(x), quadraticprime(x))


def test_xnes_rosen():
    dim = 2
    wrt = scipy.ones((dim,)) * 0.6
    rosenandprime = lambda x: (rosen(x), rosen_der(x))

    opt = Xnes(wrt, rosen)
    for i, info in enumerate(opt):
        if i > 5000:
            break
    assert (abs(wrt - [1, 1]) < 0.3).all(), 'did not find solution: %s' % wrt
