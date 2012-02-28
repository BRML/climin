import time

def stop_after_1000_iterations(info):
    """ this is a very simple example stop criterion that stops 
        after 1000 iterations. stop criterions need to be callables 
        that take the info dictionary and return either True or False.
    """
    return info['n_iter'] > 1000


# Stop criterions can be simple functions like the above one, but
# most of the following functions are actually stop criterion
# factories, that return a stop criterion upon initialization.


def stop_after_n_iterations(n):
    """ returns a stop criterion that stops after n iterations. """
    def inner(info):
        return info['n_iter'] >= (n-1)
    return inner
    
    
def stop_modulo_n_iterations(n):
    """ returns a stop criterion that stops at each n-th iteration. 
        Example for n=5: stops at n_iter = 0, 5, 10, 15, ...
    """
    def inner(info):
        return info['n_iter'] % n == 0
    return inner
    

def stop_time_elapsed(sec):
    """ returns a stop criterion that stops after `sec` seconds after initializing. """
    start = time.time()
    def inner(info):
        return time.time() - start > sec
    return inner


def stop_and(criterions):
    """ returns a stop criterion that takes a list of other stop criterions and
        only returns True, if all of them return True. This basically
        implements a logical AND for stop criterions. 
    """
    def inner(info):
        return all(c(info) for c in criterions)
    return inner

