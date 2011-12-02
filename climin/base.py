import itertools

def repeat_or_iter(obj):
    try:
        return iter(obj)
    except TypeError:
        return itertools.repeat(obj)


class Minimizer(object):

    def __init__(self, wrt, args=None, verbose=False):
        self.wrt = wrt 
        self.verbose = verbose
        if args is None:
            self.args = itertools.repeat(([], {}))
        else:
            self.args = args
