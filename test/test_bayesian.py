import scipy
from scipy.optimize import rosen

from climin.bayesian import Bayesian


quadratic = lambda x: (x**2).sum()


def test_bayesian_rosen():
    dim = 2
    wrt = scipy.zeros((dim,))

    x0s = [
        [0, 0],
        [.2, .4],
        [-.2, -.3],
        [-0.5, 0.3]
    ]

    opt = Bayesian(wrt, rosen, x0s)
    for i, info in enumerate(opt):
        if i > 10:
            break
    print wrt
    assert (abs(wrt - [1, 1]) < 0.01).all(), 'did not find solution'


def test_bayesian_quadratic():
    dim = 2
    wrt = scipy.ones(dim)

    x0s = scipy.asarray([
        [-1., 1.],
        [-1.2, 1.4],
        [-0.5, -0.3],
        [0.1, 0.1], 
        [1.2, 1.3],
    ])
    #x0s = [[-5.0, -1.0], [-12.0, 10.0]]

    #x0s = [
    #    [-1],
    #    [-1.2],
    #    [1.2],
    #    [-0.5]
    #]

    opt = Bayesian(wrt, quadratic, x0s)
    for i, info in enumerate(opt):
        print wrt
        if i > 100:
            break
    print wrt
    assert (abs(wrt) < 0.01).all(), 'did not find solution'
