# -*- coding: utf-8 -*-

import itertools

import scipy

from base import Minimizer
from lbfgs import Lbfgs
from bfgs import Bfgs


class KrylovSubspaceDescent(Minimizer):
    """Minimize a function using Krylov subspace descent as described in

      Vinyals, Povey (2011)
      http://opt.kyb.tuebingen.mpg.de/papers/opt2011_vinyals.pdf
    """

    def __init__(
        self, wrt, f, fprime, f_Hp, n_bases,
        args, hessian_args, krylov_args,
        floor_fisher=False, precond_hessian=False, floor_hessian=False,
        logfunc=None):

        super(KrylovSubspaceDescent, self).__init__(
            wrt, args=args, logfunc=logfunc)
        self.f = f
        self.fprime = fprime
        self.f_Hp = f_Hp

        self.n_bases = n_bases
        self.basis = scipy.zeros((n_bases, self.wrt.shape[0]))
        self.coefficients = scipy.zeros(n_bases)

        self.hessian_args = hessian_args
        self.krylov_args = krylov_args
        self.floor_fisher = floor_fisher
        self.precond_hessian = precond_hessian
        self.floor_hessian = floor_hessian
        self.floor_eps = 1E-4

    def _f_krylov(self, x, *args, **kwargs):
        wrt = self.wrt + scipy.dot(x, self.basis)
        return self.f(wrt, *args, **kwargs)

    def _f_krylov_prime(self, x, *args, **kwargs):
        wrt = self.wrt + scipy.dot(x, self.basis)
        df_dwrt = self.fprime(wrt, *args, **kwargs)
        return scipy.dot(self.basis, df_dwrt)

    def _calc_krylov_basis(self, grad, step):
        args, kwargs = self.hessian_args.next()
        n_bases = self.n_bases

        diag_fisher = grad**2

        if self.floor_fisher:
            diag_fisher_max = diag_fisher.max()
            df_floor = diag_fisher_max * self.floor_eps
            diag_fisher = scipy.clip(diag_fisher, df_floor, diag_fisher_max)

        H = scipy.empty((n_bases, n_bases))
        W = scipy.empty((n_bases, grad.shape[0]))
        V = self.basis

        V[0] = grad / diag_fisher
        V[0] = V[0] / scipy.sqrt(scipy.inner(V[0], V[0]))

        for i in range(0, n_bases):
            w = self.f_Hp(self.wrt, V[i], *args, **kwargs)
            if i < n_bases - 1:
                u = w / diag_fisher
            elif i == n_bases - 1:
                u = step

            for j in range(i + 1):
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

        self.hessian = H

    def __iter__(self):
        step = scipy.ones(self.wrt.shape)
        for i in itertools.count():
            _args, _kwargs = self.args.next()
            self.coefficients *= 0
            grad = self.fprime(self.wrt, *_args, **_kwargs)
            self._calc_krylov_basis(grad, step)

            # Minimize subobjective.
            subargs, subkwargs = self.krylov_args.next()

            initial_inv_hessian = scipy.linalg.inv(self.hessian)
            subopt = Bfgs(self.coefficients,
                          self._f_krylov, self._f_krylov_prime,
                          initial_inv_hessian=initial_inv_hessian,
                          args=itertools.repeat((subargs, subkwargs)),
                          logfunc=self.logfunc)

            info = subopt.minimize_until([
                lambda info: info['n_iter'] > 2 * self.n_bases])

            if info is None:
                self.logfunc({'message': 'inner loop took no steps'})
                continue

            # Take search step.
            step[:] = scipy.dot(self.coefficients, self.basis)
            self.wrt += step
            info.update(dict(step=step, grad=grad, basis=self.basis, n_iter=i,
                             coefficients=self.coefficients))
            yield info
