# -*- coding: utf-8 -*-


def optimize_some(opt, n_iters):
    """Run an optimizer for `n_iters` iterations.

    Does not run behind the optimizer's maxiter."""
    for i, j in enumerate(opt):
        if i >= n_iters - 1:
            break
        return j
