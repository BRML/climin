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
        self, wrt, fandprime, f_Hp, f_krylov, f_krylovprime,
        krylov_basis, krylov_coefficients,
        args, hessian_args, krylov_args,
        floor_fisher=False,
        precond_hessian=False,
        floor_hessian=False,
        stop=1, verbose=False):

        super(KrylovSubspaceDescent, self).__init__(
            wrt, args=args, stop=stop, verbose=verbose)
        self.fandprime = fandprime
        self.f_Hp = f_Hp
        self.f_krylov = f_krylov
        self.f_krylovprime = f_krylovprime
        self.krylov_basis = krylov_basis
        self.krylov_coefficients = krylov_coefficients
        self.hessian_args = hessian_args
        self.krylov_args = krylov_args
        self.floor_fisher = floor_fisher
        self.precond_hessian = precond_hessian
        self.floor_hessian = floor_hessian
        self.floor_eps = 1E-4

    def _calc_krylov_basis(self, grad, step):
        args, kwargs = self.hessian_args.next()

        diag_fisher = grad**2

        if self.floor_fisher:
            diag_fisher_max = diag_fisher.max()
            df_floor = diag_fisher_max * self.floor_eps
            diag_fisher = scipy.clip(diag_fisher, df_floor, diag_fisher_max)

        inv_diag_fisher = 1 / diag_fisher

        n_bases = self.krylov_coefficients.shape[0]
        H = scipy.empty((n_bases, n_bases))
        V = self.krylov_basis

        V[0] = inv_diag_fisher * grad
        V[0] /= scipy.sqrt(scipy.dot(V[0].T, V[0]))
        for i in range(0, n_bases):
            w = self.f_Hp(V[i], *args, **kwargs)

            if i < n_bases - 1:
                u = w * inv_diag_fisher
            elif i == n_bases - 1:
                u = step.copy()

            for j in range(0, i + 1):
                H[j, i] = H[i, j] = scipy.dot(w.T, V[j])
                u -= scipy.dot(scipy.dot(u.T, V[j]), V[j])

            if i < n_bases - 1:
                V[i + 1] = u / scipy.sqrt(scipy.dot(u.T, u))

        if self.precond_hessian:
            if self.floor_hessian:
                w, v = scipy.linalg.eigh(H)
                w_max = w.max()
                w_floor = w_max * self.floor_eps
                w = scipy.clip(w, w_floor, w_max)
                H = scipy.dot(v, scipy.dot(scipy.diag(w), v.T))

            C = scipy.linalg.cholesky(H)
            Cinv = scipy.linalg.inv(C)
            V[:] = scipy.dot(Cinv, V)

        self.krylov_hessian = H

    def _f_krylov(self, x, *args, **kwargs):
        old = self.krylov_coefficients.copy()
        self.krylov_coefficients[:] = x
        loss = self.f_krylov(*args, **kwargs)
        self.krylov_coefficients[:] = old
        return loss

    def _fprime_krylov(self, x, *args, **kwargs):
        old = self.krylov_coefficients.copy()
        self.krylov_coefficients[:] = x
        grad = self.f_krylovprime(*args, **kwargs)
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
                f, self.krylov_coefficients, fprime, maxfun=20, disp=0,
                m=10,
                pgtol=1E-12, factr=10)
            self.krylov_coefficients[:] = step_coeffs

            # Take search step.
            step[:] = scipy.dot(self.krylov_coefficients, self.krylov_basis)
            self.wrt += step
            yield dict(
                loss=f, step=step, grad=grad,
                krylov_basis=self.krylov_basis,
                krylov_coefficients=self.krylov_coefficients)
