# -*- coding: utf-8 -*-


import scipy
import scipy.optimize
from scipy import stats
from sklearn import gaussian_process

from base import Minimizer
from nes import Xnes


# TODO: add logging functionality


# Taken from: http://atpassos.posterous.com/bayesian-optimization

def upper_confidence_bound(sigma):
    """The upper-confidence-bound acquisition function with parameter sigma.
    
    To avoid over-exploration all the way off to infinity (where the
    variance is huge) I truncate the confidence term.
    """
    def acq(gp, best_y):
        def ev(x):
            y, ms = gp.predict(x, eval_MSE=True)
            return -(y[0] + sigma * max(abs(y[0]), scipy.sqrt(ms[0])))
        return ev
    return acq


def expected_improvement(gp, best_loss, atleast=0.00):
    def inner(x):
        mean, var = gp.predict(x, eval_MSE=True)
        if var[0] == 0: 
            return 0.
        std = scipy.sqrt(var)
        improv = mean - best_loss + atleast
        z = improv / std
        return (improv * stats.norm.cdf(z) + std * stats.norm.pdf(z))
    return inner


def calc_proposal(trials, losses, model_factory, acq_func, n_inner_iters=100):
    # Make sure the data given is a proper array.
    trials = scipy.asarray(trials)
    losses = scipy.asarray(losses)

    # Best solution found so far.
    best_loss = losses.min()
    best_trial = trials[losses.argmin()]

    # Fit a model of the cost function...
    model = model_factory()
    model.fit(trials, losses)
    # ... and wrap it into the acquaintance function.
    f = acq_func(model, best_loss)

    # Optimize the model of the cost to obtain new query point.
    new_trial = best_trial.copy()
    inner_opt = Xnes(new_trial, f)
    inner_opt.some(max_iter=n_inner_iters)

    return new_trial


class Bayesian(Minimizer):

    def __init__(self, wrt, f, initial_trials, model_factory=None, 
                 acq_func=None, tolerance=1E-20, n_inner_iters=50,
                 args=None, stop=1, logfunc=None):
        super(Bayesian, self).__init__(wrt, args, stop, logfunc=logfunc)
        self.f = f

        self.initial_trials = initial_trials
        self.n_inner_iters = n_inner_iters
        self.trials = []
        self.losses = []

        if model_factory is None:
            model_factory = gaussian_process.GaussianProcess
        self.model_factory = model_factory

        if acq_func is None:
            acq_func = expected_improvement
        self.acq_func = acq_func

        self.tolerance = tolerance

    def eval_initial_points(self, args, kwargs):
        best_loss = float('inf')

        self.initial_trials = scipy.asarray(self.initial_trials)
        for trial in self.initial_trials:
            loss = self.f(trial, *args, **kwargs)
            self.trials.append(trial)
            self.losses.append(loss)
            if loss < best_loss:
                self.wrt[:] = trial 
                best_loss = loss
        return best_loss

    def __iter__(self):
        # Evaluate the given points first.
        args, kwargs = self.args.next()

        best_loss = self.eval_initial_points(args, kwargs)

        # Now go into Bayesian loop.
        for i, (args, kwargs) in enumerate(self.args):
            new_trial = calc_proposal(
                self.trials, self.losses, self.model_factory, self.acq_func,
                n_inner_iters=self.n_inner_iters)
            new_loss = self.f(new_trial, *args, **kwargs)
            self.trials.append(new_trial)
            self.losses.append(new_loss)

            if new_loss < best_loss:
                self.wrt[:] = new_trial
                best_loss = new_loss

            yield dict(loss=best_loss)
