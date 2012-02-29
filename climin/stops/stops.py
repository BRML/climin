import time

def after_1000_iterations(info):
    """ this is a very simple example stop criterion that stops 
        after 1000 iterations. stop criterions need to be callables 
        that take the info dictionary and return either True or False.
    """
    return info['n_iter'] >= 1000-1

# Stop criterions can be simple functions like the above one, but
# most of the following functions are actually stop criterion
# factories, that return a stop criterion upon initialization.

def after_n_iterations(n):
    """ returns a stop criterion that stops after n iterations. """
    def inner(info):
        return info['n_iter'] >= (n-1)
    return inner
    
    
def modulo_n_iterations(n):
    """ returns a stop criterion that stops at each n-th iteration. 
        Example for n=5: stops at n_iter = 0, 5, 10, 15, ...
    """
    def inner(info):
        return info['n_iter'] % n == 0
    return inner
    

def time_elapsed(sec):
    """ returns a stop criterion that stops after `sec` seconds after initializing. """
    start = time.time()
    def inner(info):
        return time.time() - start > sec
    return inner


def converged(func, n=10, epsilon=1e-5):
    """ returns a stop criterion that remembers the last n return values of func() in 
        a buffer and stops if the difference of max(buffer) - min(buffer) is smaller 
        than epsilon. `func` needs to be a callable that returns a scalar value. 
    """
    ringbuffer = [None for i in xrange(n)]
    def inner(info):
        ringbuffer.append(func())
        ringbuffer.pop(0)
        if not None in ringbuffer:
            ret = (max(ringbuffer) - min(ringbuffer)) < epsilon
        else:
            ret = False 

        return ret
    return inner


def and_(criterions):
    """ takes a list of stop criterions and returns a stop criterion that
        only returns True, if all of criterions return True. This basically
        implements a logical AND for stop criterions. 
    """
    def inner(info):
        return all(c(info) for c in criterions)
    return inner



