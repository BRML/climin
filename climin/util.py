# -*- coding: utf-8 -*-


def optimize_some(opt, n_iters, log=None):
    """Run an optimizer for `n_iters` iterations.

    Does not run behind the optimizer's maxiter."""
    if log is None:
        log = lambda x: None
    for i, info in enumerate(opt):
        log(info)
        if i >= n_iters - 1:
            break
    return info 


def optimize_while(opt, loss_improv=1E-5, log=None):
    """Run an optimizer while the loss is improving more than `loss_improv`.

    Returns the last info from iteration over the optimizer."""
    if log is None:
        log = lambda x: None
    loss_m1 = float('inf')
    for info in opt:
        log(info)
        if loss_m1 - info['loss'] < loss_improv:
            break
        loss_m1 = info['loss']
    return info
