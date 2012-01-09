import itertools

import scipy
from scipy.optimize import rosen, rosen_der
import theano
import theano.tensor as T

from climin.ksd import KrylovSubspaceDescent
from zeitgeist.model import krylov_subspace


quadratic = lambda x: (x**2).sum()
quadraticprime = lambda x: 2 * x
quadraticandprime = lambda x: (quadratic(x), quadraticprime(x))


def test_ksd_rosen():
    pars = T.dvector('pars')
    wrt = scipy.zeros(2)
    x, y = pars[0], pars[1]
    rosen_expr = (1 - x)**2 + 100 * (y - x**2)**2
    rosen_grad = T.grad(rosen_expr, pars)

    p = T.dvector('point')
    Hp = T.grad(T.sum(rosen_grad *  p), pars)

    k_coeff_var = T.dvector('coeffs var')
    k_basis, k_coeff, k_loss, k_grad = krylov_subspace(rosen_expr, pars, wrt, 2)
    k_coeff.name = 'coeffs'
    k_basis.name = 'basis'
    f_kl = theano.function([pars, k_coeff_var], k_loss, givens={k_coeff: k_coeff_var})
    f_kp = theano.function([pars, k_coeff_var], k_grad, givens={k_coeff: k_coeff_var})
    f_Hp = theano.function([pars, p], Hp)
    fandprime = theano.function([pars], [rosen_expr, rosen_grad])

    args = hessian_args = krylov_args = itertools.repeat(((), {}))

    opt = KrylovSubspaceDescent(**{
        'wrt': wrt,
        'fandprime': fandprime,
        'f_Hp': f_Hp,
        'f_krylov': f_kl,
        'f_krylovprime': f_kp,
        'krylov_basis': k_basis.get_value(borrow=True, return_internal_type=True),
        'krylov_coefficients':
            k_coeff.get_value(borrow=True, return_internal_type=True),
        'floor_fisher': True,
        'floor_hessian': True,
        'precond_hessian': True,
        'args': args,
        'hessian_args': hessian_args,
        'krylov_args': krylov_args,
        })

    for i, info in enumerate(opt):
        if (abs(wrt - [1, 1]) < 0.05).all():
            success = True
            break
        if i >= 100:
            success = False
            break
    print wrt
    assert success, 'did not find solution'
