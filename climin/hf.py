# -*- coding: utf-8 -*-

import numpy as np

from base import Minimizer, is_nonzerofinite
from cg import ConjugateGradient
from linesearch import WolfeLineSearch, BackTrack


def relative_improvement_less_than(
    losses, atleast=5e-4, lookback_factor=0.1, min_lookback=10):
    """Return True if the relative improvement in the last few iterations
    was worse than `atleast`.

    The last few iterations are determined by looking back k time steps, where
    k is `lookback_factor` * len(losses), but at least `min_lookback`.

    It is assumed that the loss is bounded by 0 from above.
    """
    lookback = int(max(min_lookback, lookback_factor * len(losses)))
    if lookback >= len(losses):
        return False
    if losses[-1] > 0:
        return False
    if (losses[-1] - losses[-lookback - 1]) / losses[-1] < lookback * atleast:
        return True


class HessianFree(Minimizer):

    def __init__(self, wrt, f, fprime, f_Hp, 
                 args=None, cg_args=None, ls_args=None,
                 initial_damping=0.1,
                 cg_min_iter=1,
                 cg_max_iter=250,
                 line_search=None,
                 precond=False,
                 logfunc=None):
        super(HessianFree, self).__init__(wrt, args=args, logfunc=logfunc)

        self.f = f
        self.fprime = fprime
        self.f_Hp = f_Hp

        self.cg_args = cg_args if cg_args is not None else self.args
        self.ls_args = ls_args if ls_args is not None else self.cg_args

        self.initial_damping = initial_damping

        self.cg_min_iter = cg_min_iter
        self.cg_max_iter = cg_max_iter

        self.precond = precond

        if line_search is not None:
            self.line_search = line_search
        else:
            self.line_search = BackTrack(wrt, self.f, decay=0.8, max_iter=50,
                                         logfunc=logfunc)

    def backtrack_cg(self, directions, base_loss, args, kwargs):
        """Return the first of `directions` (from the back) which has a lower
        loss than `base_loss`."""
        for i, direction in enumerate(reversed(directions)):
            loss = self.f(self.wrt + direction, *args, **kwargs)
            if loss < base_loss:
                break
        return len(directions) - i - 1, loss

    def get_preconditioner(self, grad, damping):
        # Preconditioning matrix.
        if self.precond == 'martens':
            precond = (grad**2 + damping)**0.75
        elif not self.precond:
            precond = np.ones(grad.size)
        else:
            raise ValueError('unknown preconditioning: %s' % self.precond)
        return precond

    def find_direction(self, loss, grad, direction_m1, damping, args, kwargs):
        # Copy, because we will optimize this beast inplace.
        direction = direction_m1.copy()             

        # Define function for the Hessian vector product that 
        # includes damping.
        def f_Hp(x):
            return self.f_Hp(self.wrt, x, *args, **kwargs) + damping * x

        # Define short hand for the loss of the quadratic approximation.
        def f_q_loss(direction):
            qloss = .5 * np.inner(direction, f_Hp(direction))
            qloss -= np.inner(-grad, direction)
            return qloss

        # Variables for the backtracking of cg. We will save the direction every
        # few iterations.
        save_increase_factor= 1.3
        next_saving = 2
        directions = [direction.copy()]

        # Calculate once first, because we might exit the loop right away. 
        q_loss = f_q_loss(direction)
        q_losses = [q_loss]

        # Instantiate cg.
        opt = ConjugateGradient(
            direction, f_Hp=f_Hp, b=-grad, 
            precond=self.get_preconditioner(grad, damping),
            logfunc=self.logfunc)

        for i, info in enumerate(opt):
            self.logfunc(info)

            # Saving current direction for backtracking.
            if i == np.ceil(next_saving):
                next_saving *= save_increase_factor
                directions.append(direction.copy())

            q_losses.append(f_q_loss(direction))

            # Stoping criterions.
            if relative_improvement_less_than(q_losses, 5e-4, 0.1, 10):
                self.logfunc({
                    'message': 'stopping cg - stopping criterion'})
                break

            if i + 1 >= self.cg_max_iter:
                self.logfunc({
                    'message': 'stopping cg - max iterations reached'})
                break

        # Check intermediate CG results for the best.
        idx, loss = self.backtrack_cg(directions, loss, args, kwargs)

        return directions[idx], loss, q_losses[idx]

    def __iter__(self):
        damping = self.initial_damping
        direction_m1 = np.zeros(self.wrt.size)
        args, kwargs = self.args.next()

        # Calculcate one loss before going into the loop.
        loss_m1 = self.f(self.wrt, *args, **kwargs)

        for i, (next_args, next_kwargs) in enumerate(self.args):
            grad = self.fprime(self.wrt, *args, **kwargs)

            # Obtain search direction via cg.
            cg_args, cg_kwargs = self.cg_args.next()
            direction, _, q_loss = self.find_direction(
                loss_m1, grad, direction_m1 * 0.95, damping, cg_args, cg_kwargs)

            if not is_nonzerofinite(direction):
                self.logfunc({'message': 'invalid direction'})
                break

            # Obtain a steplength with a line search.
            ls_args, ls_kwargs = self.ls_args.next()
            step_length = self.line_search.search(
                direction, 1, ls_args, ls_kwargs, loss0=loss_m1)

            # Update parameters.
            step = step_length * direction
            self.wrt += step

            loss = self.f(self.wrt, *next_args, **next_kwargs)

            # Levenberg-Marquardt update of damping. First calculate the current
            # value of the quadratic approximation with no damping, then see 
            # how well it does compared to real loss.
            Hp = self.f_Hp(self.wrt, direction, *cg_args, **cg_kwargs)
            q_loss = 0.5 * np.inner(direction, Hp) - np.inner(-grad, direction)
            ratio = (loss - loss_m1) / q_loss
            if ratio < 0.25:
                damping *= 1.5
            elif ratio > 0.75:
                damping *= 2. / 3

            info = {
                'n_iter': i,
                'loss': loss,
                'loss_m1': loss_m1,
                'gradient': grad,
                'direction': direction,
                'direction_m1': direction_m1,
                'step_length': step_length,
                'q_loss': q_loss,
                'ratio': ratio,
                'damping': damping,
                'args': args,
                'step': step,
                'kwargs': kwargs}
            yield info

            # Prepare for next loop.
            args, kwargs = next_args, next_kwargs
            loss_m1 = loss
            direction_m1 = direction
