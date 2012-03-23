# -*- coding: utf-8 -*-


import random


def coroutine(f):
    """Turn a generator function into a coroutine by calling .next() once."""
    def started(*args, **kwargs):
        cr = f(*args, **kwargs)
        cr.next()
        return cr
    return started


def aslist(item):
    if not isinstance(item, (list, tuple)):
        item = [item]
    return item


def mini_slices(n_samples, batch_size):
    """Yield slices of size `batch_size` that work with a container of length
    `n_samples`."""
    n_batches, rest = divmod(n_samples, batch_size)
    if rest != 0:
        n_batches += 1
        
    return [slice(i * batch_size, (i + 1) * batch_size) for i in range(n_batches)]


def draw_mini_slices(n_samples, batch_size, with_replacement=False):
    slices = mini_slices(n_samples, batch_size)
    idxs = range(len(slices))

    if with_replacement:
        yield random.choice(slices)
    else:
        while True:
            random.shuffle(idxs)
            for i in idxs:
                yield slices[i]
