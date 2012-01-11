# -*- coding: utf-8 -*-

import itertools

import scipy

from base import Minimizer, repeat_or_iter
from lbfgs import Lbfgs
from bfgs import Bfgs
from sbfgs import SBfgs
from util import optimize_some, optimize_while


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

    def _inner_f(self, step, *args, **kwargs):
        return self.f_krylov(self.wrt, step, *args, **kwargs)

    def _inner_fprime(self, step, *args, **kwargs):
        return self.f_krylovprime(self.wrt, step, *args, **kwargs)

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
        W = scipy.empty((n_bases, grad.shape[0]))
        V = self.krylov_basis

        V[0] = grad / diag_fisher
        V[0] = V[0] / scipy.sqrt(scipy.inner(V[0], V[0]))

        for i in range(0, n_bases):
            w = self.f_Hp(self.wrt, V[i], *args, **kwargs)
            if i < n_bases - 1:
                u = w / diag_fisher
            elif i == n_bases - 1:
                u = step

            for j in range(i):
                H[i, j] = H[j, i] = scipy.inner(w, V[j])
                u -= scipy.inner(u, V[j]) * V[j]
            if i < n_bases - 1:
                V[i + 1] = u / scipy.sqrt(scipy.inner(u, u))

        if self.precond_hessian:
            if self.floor_hessian:
                w, v = scipy.linalg.eigh(H)
                w_max = w.max()
                w_floor = w_max * self.floor_eps
                w = scipy.clip(w, w_floor, w_max)
                H = scipy.dot(v, scipy.dot(scipy.diag(w), v.T))

            C = scipy.linalg.cholesky(H, lower=True)
            Cinv = scipy.linalg.inv(C)
            V[:] = scipy.dot(Cinv, V)

        self.krylov_hessian = H

    def __iter__(self):
        step = scipy.ones(self.wrt.shape)
        while True:
            _args, _kwargs = self.args.next()
            self.krylov_coefficients *= 0
            loss, grad = self.fandprime(self.wrt, *_args, **_kwargs)
            self._calc_krylov_basis(grad, step)

            # Minimize subobjective.
            subargs, subkwargs = self.krylov_args.next()

            subopt = SBfgs(
                self.krylov_coefficients, self._inner_f, self._inner_fprime,
                args=itertools.repeat((subargs, subkwargs)))

            def log(info):
                print 'inner loop loss', info['loss']
                print '=' * 20

            info = optimize_while(subopt, 1E-4, log=log)

            # Take search step.
            step[:] = scipy.dot(self.krylov_coefficients, self.krylov_basis)
            self.wrt += step
            yield dict(
                loss=loss, step=step, grad=grad,
                krylov_basis=self.krylov_basis,
                krylov_coefficients=self.krylov_coefficients)


            print 'parameter hash', self.wrt.sum(), (self.wrt**2).sum()
            print '-' * 20
            print '=' * 20
            print '-' * 20
