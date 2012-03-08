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
            print 'backtracking through losses', loss
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

    def find_direction(self, loss, grad, direction_m1, damping, 
                       cg_args, cg_kwargs, bt_args, bt_kwargs):
        # Copy, because we will optimize this beast inplace.
        direction = direction_m1 * 0.95

        # Define function for the Hessian vector product that 
        # includes damping.
        def f_Hp(x):
            return self.f_Hp(self.wrt, x, *cg_args, **cg_kwargs) + damping * x

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
            # Saving current direction for backtracking.
            if i == np.ceil(next_saving):
                next_saving *= save_increase_factor
                directions.append(direction.copy())

            q_losses.append(f_q_loss(direction))

            info.update({'loss': q_losses[-1]})
            self.logfunc(info)

            # Stopping criterions.
            if relative_improvement_less_than(q_losses, 5e-4, 0.1, 10):
                self.logfunc({
                    'message': 'stopping cg - stopping criterion'})
                break

            if i + 1 >= self.cg_max_iter:
                self.logfunc({
                    'message': 'stopping cg - max iterations reached'})
                break

        # TODO: we do this once too much if the last was a saving iteration.
        directions.append(direction.copy())

        # Check intermediate CG results for the best.
        idx, loss = self.backtrack_cg(directions, loss, bt_args, bt_kwargs)
        self.logfunc({'backtrack-idx': idx})

        return directions[idx], directions[-1], q_losses[idx]

    def __iter__(self):
        damping = self.initial_damping
        cg_minimum = np.zeros(self.wrt.size)

        args, kwargs = self.args.next()
        loss = self.f(self.wrt, *args, **kwargs)
        grad = self.fprime(self.wrt, *args, **kwargs)

        for i, (args, kwargs) in enumerate(self.args):

            # Get minibatches for cg and for backtracking.
            cg_args, cg_kwargs = self.cg_args.next()
            bt_args, bt_kwargs = self.cg_args.next()

            # Obtain search direction via cg.
            direction, cg_minimum, q_loss = self.find_direction(
                loss, grad, cg_minimum, damping, 
                cg_args, cg_kwargs, bt_args, bt_kwargs)

            if not is_nonzerofinite(direction):
                self.logfunc({'message': 'invalid direction'})
                break

            # Get minibatches for line search.
            ls_args, ls_kwargs = self.ls_args.next()
            loss0 = self.f(self.wrt, *ls_args, **ls_kwargs)
            # Obtain a step length with a line search.
            step_length = self.line_search.search(
                direction, 1, ls_args, ls_kwargs, loss0=loss0)

            # Levenberg-Marquardt update of damping. First calculate the current
            # value of the quadratic approximation with no damping, then see
            # how well it does compared to real loss.
            ratio_args, ratio_kwargs = self.cg_args.next()

            # Calculate update and loss reduction.
            step = step_length * direction
            r_loss_m1 = self.f(self.wrt, *ratio_args, **ratio_kwargs)
            r_loss = self.f(self.wrt + cg_minimum, *ratio_args, **ratio_kwargs)
            Hp = self.f_Hp(self.wrt, cg_minimum, *ratio_args, **ratio_kwargs)
            Hp += (damping * cg_minimum)
            q_loss = 0.5 * np.inner(cg_minimum, Hp) - np.inner(-grad, cg_minimum)

            self.wrt += step

            ratio = (r_loss - r_loss_m1) / q_loss
            if ratio < 0.25:
                damping *= 1.5
            elif ratio > 0.75:
                damping *= 2. / 3

            loss = self.f(self.wrt, *args, **kwargs)
            grad = self.fprime(self.wrt, *args, **kwargs)

            info = {
                'n_iter': i,
                'loss': loss,
                'gradient': grad,
                'direction': direction,
                'cg_minimum': cg_minimum,
                'step_length': step_length,
                'q_loss': q_loss,
                'ratio': ratio,
                'r_loss_m1': r_loss_m1,
                'r_loss': r_loss,
                'damping': damping,
                'args': args,
                'step': step,
                'kwargs': kwargs}
            yield info
