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
        self.krylov_basis[0] = step
        self.krylov_basis[1] = grad
        args, kwargs = self.hessian_args.next()
        for i in range(1, self.krylov_coefficients.shape[0] - 1):
            self.krylov_basis[i + 1] = self.f_Hp(
                self.krylov_basis[i], *args, **kwargs)

    def __iter__(self):
        step = scipy.ones(self.wrt.shape)
        while True:
            _args, _kwargs = self.args.next()
            self.krylov_coefficients *= 0
            loss, grad = self.fandprime(*_args, **_kwargs)
            self._calc_krylov_basis(grad, step)

            # Minimize subobjective.
            subargs, subkwargs = self.krylov_args.next()
            def f(x):
              old = self.krylov_coefficients.copy()
              self.krylov_coefficients[:] = x
              loss, _ = self.f_krylovandprime(*subargs, **subkwargs)
              self.krylov_coefficients[:] = old
              return loss
            def fprime(x):
              old = self.krylov_coefficients.copy()
              self.krylov_coefficients[:] = x
              _, grad = self.f_krylovandprime(*subargs, **subkwargs)
              #print grad
              self.krylov_coefficients[:] = old
              return grad

            step_coeffs, f, d = scipy.optimize.fmin_l_bfgs_b(
                f, self.krylov_coefficients, fprime, maxfun=50,
                pgtol=1E-12, factr=10.)
            ##step_coeffs = scipy.optimize.fmin_cg(
            ##    f, self.krylov_coefficients, fprime, maxiter=30)
            self.krylov_coefficients[:] = step_coeffs

            #opt = Rprop(
            #    self.krylov_coefficients, self.f_krylovandprime,
            #    stepshrink=0.2, stepgrow=1.1,
            #    maxstep=0.1, minstep=1e-10,
            #    changes_max=0.000001,
            #    args=((subargs, subkwargs) for _ in itertools.repeat(())))
            #opt = GradientDescent(
            #    self.krylov_coefficients, self.f_krylovandprime,
            #    steprate=0.001, momentum=0.9,
            #    args=((subargs, subkwargs) for _ in itertools.repeat(())))
            #for i, info in enumerate(opt):
            #    if self.verbose:
            #      #print info['loss']
            #      # print self.krylov_coefficients
            #      pass
            #    if i == 10:
            #      break

            # Take search step.
            step[:] = scipy.dot(self.krylov_coefficients, self.krylov_basis)
            #step = scipy.clip(step, -5, 5)
            #print self.krylov_coefficients
            #print step.shape, self.wrt.shape
            #print self.wrt.sum(), (self.wrt**2).sum()
            self.wrt += step
            yield dict(loss=loss, step=step)
