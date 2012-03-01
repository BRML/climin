# -*- coding: utf-8 -*-

import numpy as np

from base import Minimizer, is_nonzerofinite
from cg import ConjugateGradient
from linesearch import WolfeLineSearch


class HessianFree(Minimizer):

    def __init__(self, wrt, f, fprime, f_Hp, args=None, cg_args=None, 
                 initial_damping=0.1,
                 cg_max_iter=250,
                 line_search=None,
                 logfunc=None):
        super(HessianFree, self).__init__(wrt, args=args, logfunc=logfunc)

        self.f = f
        self.fprime = fprime
        self.f_Hp = f_Hp

        self.cg_args = cg_args if cg_args is not None else self.args
        self.initial_damping = initial_damping
        self.cg_max_iter = cg_max_iter

        if line_search is not None:
            self.line_search = line_search
        else:
            self.line_search = WolfeLineSearch(wrt, self.f, self.fprime)

    def find_direction(self, loss, grad, direction_m1, damping, args, kwargs):
        # TODO really need to copy here?
        direction = direction_m1.copy()             
        q_losses = []

        # Define short hand for the loss of the quadratic approximation.
        def f_q_loss(direction):
            qloss = .5 * np.inner(direction, f_Hp(direction))
            qloss -= np.inner(-grad, direction)
            qloss += loss
            return qloss

        # Define another function for the Hessian vector product that 
        # includes damping.
        def f_Hp(x):
            return self.f_Hp(self.wrt, x, *args, **kwargs) + damping * x 

        opt = ConjugateGradient(direction, f_Hp=f_Hp, b=-grad,
                                logfunc=self.logfunc)
        
        # Calculate once first, because we might exit the loop before
        # calculating it.
        q_loss = f_q_loss(direction)
        q_losses = [q_loss]
                 
        for i, info in enumerate(opt):
            self.logfunc(info)

            if i + 1 >= self.cg_max_iter:
                self.logfunc({
                    'message': 'stopping cg - max iterations reached'})
                break

            lookback = max(i / 10,  10)
            if lookback < len(q_losses):
                if (q_loss - q_losses[-lookback]) / q_loss < 0.0005:
                    self.logfunc({
                        'message': 'stopping cg - stopping criterion'})

            q_loss = f_q_loss(direction)
            q_losses.append(q_loss)

        return direction, {'q-loss': q_loss}

    def __iter__(self):
        damping = self.initial_damping
        direction_m1 = np.ones(self.wrt.size)
        q_loss_m1 = 0

        # Calculcate one loss before going into the loop.
        args, kwargs = self.args.next()
        loss_m1 = self.f(self.wrt, *args, **kwargs)

        for i, (args, kwargs) in enumerate(self.args):
            grad = self.fprime(self.wrt, *args, **kwargs)

            # Obtain search direction via cg.
            cg_args, cg_kwargs = self.cg_args.next()
            direction, info = self.find_direction(
                loss_m1, grad, direction_m1 * 0.95, damping, cg_args, cg_kwargs)

            q_loss = info['q-loss']

            if not is_nonzerofinite(direction):
                self.logfunc(
                    {'message': 'direction is invalid -- need to bail out.'})
                break

            # TODO: do line search on CG args or on args?
            # TODO: actually, we should backtrack here, which means that we go
            # back along the solutions that CG finds. But maybe a line search is
            # fine.
            step_length = self.line_search.search(direction, None, args, kwargs)

            if step_length != 0:
                step = step_length * direction
                self.wrt += step
            else:
                self.logfunc({'message': 'step length is 0.'})

            loss = self.f(self.wrt, *args, **kwargs)

            # Levenberg-Marquardt update of damping.
            ratio = (loss - loss_m1) / (q_loss - q_loss_m1)
            damping *= 1.5 if ratio < 0.25 else (0.666 if ratio > 0.75 else 1)

            loss_m1 = loss
            q_loss_m1 = q_loss
            direction_m1 = direction

            info.update({
                'loss': loss, 'gradient': grad, 'q_loss': q_loss,
                'ratio': ratio, 'damping': damping})
            yield info
