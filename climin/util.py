# -*- coding: utf-8 -*-


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
