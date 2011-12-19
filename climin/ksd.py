# -*- coding: utf-8 -*-

import itertools

import scipy

from base import Minimizer, repeat_or_iter
from gd import GradientDescent
from rprop import Rprop


class KrylovSubspaceDescent(Minimizer):
    """Minimize a function using Krylov subspace descent as described in

      Vinyals, Povey (2011)
      http://opt.kyb.tuebingen.mpg.de/papers/opt2011_vinyals.pdf
    """

    def __init__(
        self, wrt, fandprime, f_Hp, f_krylovandprime,
        krylov_basis, krylov_coefficients,
        args, hessian_args, krylov_args,
        stop=1, verbose=False):

        super(KrylovSubspaceDescent, self).__init__(
            wrt, args=args, stop=stop, verbose=verbose)
        self.fandprime = fandprime
        self.f_Hp = f_Hp
        self.f_krylovandprime = f_krylovandprime
        self.krylov_basis = krylov_basis
        self.krylov_coefficients = krylov_coefficients
        self.hessian_args = hessian_args
        self.krylov_args = krylov_args

    def _calc_krylov_basis(self, grad, step):
        args, kwargs = self.hessian_args.next()

        inv_diag_fisher = 1 / grad**2
        self.krylov_basis[0] = step

        v = inv_diag_fisher * grad
        self.krylov_basis[1] =  v / scipy.sqrt(scipy.dot(v.T, v))
        for i in range(2, self.krylov_basis.shape[0]):
            w = self.f_Hp(self.krylov_basis[i - 1], *args, **kwargs)
            u = w * inv_diag_fisher
            for j in range(1, i):
                v_j = self.krylov_basis[j]
                u -= scipy.dot(scipy.dot(u.T, v_j), v_j)
            self.krylov_basis[i] = u / scipy.sqrt(scipy.dot(u.T, u))

    def _f_krylov(self, x, *args, **kwargs):
        old = self.krylov_coefficients.copy()
        self.krylov_coefficients[:] = x
        loss, _ = self.f_krylovandprime(*args, **kwargs)
        self.krylov_coefficients[:] = old
        return loss

    def _fprime_krylov(self, x, *args, **kwargs):
        old = self.krylov_coefficients.copy()
        self.krylov_coefficients[:] = x
        _, grad = self.f_krylovandprime(*args, **kwargs)
        self.krylov_coefficients[:] = old
        return grad

    def __iter__(self):
        step = scipy.ones(self.wrt.shape)
        while True:
            _args, _kwargs = self.args.next()
            self.krylov_coefficients *= 0
            loss, grad = self.fandprime(*_args, **_kwargs)
            self._calc_krylov_basis(grad, step)

            # Minimize subobjective.
            subargs, subkwargs = self.krylov_args.next()
            f = lambda x: self._f_krylov(x, *subargs, **subkwargs)
            fprime = lambda x: self._fprime_krylov(x, *subargs, **subkwargs)

            step_coeffs, f, d = scipy.optimize.fmin_l_bfgs_b(
                f, self.krylov_coefficients, fprime, maxfun=100,
                pgtol=1E-12, factr=10.)
            self.krylov_coefficients[:] = step_coeffs

            # Take search step.
            step[:] = scipy.dot(self.krylov_coefficients, self.krylov_basis)
            self.wrt += step
            yield dict(loss=loss, step=step)
