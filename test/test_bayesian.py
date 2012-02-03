import nose
import scipy
from scipy.optimize import rosen

from climin.bayesian import Bayesian


quadratic = lambda x: (x**2).sum()

# The following test is deactivated because it takes so long. Also, it tends to
# produce values which make scikits GP somehow crash. That should be
# investigated actually, so this is to be read as a TODO.

@nose.tools.nottest
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
        if i > 100:
            break
    assert (abs(wrt - [1, 1]) < 0.01).all(), 'did not find solution'


def test_bayesian_quadratic():
    dim = 2
    wrt = scipy.ones(dim)

    x0s = scipy.asarray([
        [-1., 1.],
        [-1.2, 1.4],
        [-0.5, -0.3],
        [0.3, 0.3], 
        [1.2, 1.3],
    ])

    scipy.random.seed(1234)
    opt = Bayesian(wrt, quadratic, x0s, n_inner_iters=100)
    for i, info in enumerate(opt):
        if i > 100:
            break
    assert (abs(wrt) < 0.01).all(), 'did not find solution'
