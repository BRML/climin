# -*- coding: utf-8 -*-


def exponential_distant(f, wrt, direction, factor=0.8, granularity=50, 
                        args=None, kwargs=None):
    """Return the step length found by performing a line search via evaluating a
    set of points.

    The function will be evaluated at the points wrt + a * direction where
    a is in [factor**i for i in range(granularity)]. 
    
    The steplength that is farthest away and still reduces the function is
    returned. If such a point is not found, 0 is returned."""
    # TODO: add parameter to not recalculate loss0
    f_by_steplength = lambda steplength: f(
        wrt + steplength * direction, *args, **kwargs)
    distances = [factor**i for i in range(granularity)]

    loss0 = f_by_steplength(0)      # Loss at the current position.

    for d in distances:
        loss = f_by_steplength(d)
        cur_reduction = loss0 - loss
        if cur_reduction > 0:
            # We are better! So stop.
            return d
    return 0
