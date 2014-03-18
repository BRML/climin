# -*- coding: utf-8 -*-


import numpy as np


def continuation(opt):
    # First run for a few steps, then save state.
    for i, info in enumerate(opt):
        if i > 3:
            break

    inter_info = info
    inter_wrt = opt.wrt.copy()

    # Run for further steps, save final position as means of verifying
    # whether equivalent solutions has been found.
    for i, info in enumerate(opt):
        if i > 3:
            break

    final1 = opt.wrt.copy()

    # Reset optimizer to state before that.
    opt.wrt[...] = inter_wrt
    opt.set_from_info(inter_info)

    # Run for 10 more iterations.
    for i, info in enumerate(opt):
        if i > 3:
            break

    assert np.allclose(opt.wrt, final1), 'continuation did not lead to same result'
