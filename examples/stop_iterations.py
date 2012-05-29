import scipy

from climin import GradientDescent
from climin.stops import *

quadratic = lambda x: (x**2).sum()
quadraticprime = lambda x: 2 * x


if __name__ == '__main__':
    dim = 10
    wrt = scipy.random.standard_normal((dim,)) * 10 + 5
    
    modulostop = modulo_n_iterations(50)
    fullstop = after_n_iterations(1000)
    
    opt = GradientDescent(wrt, quadraticprime, steprate=0.01)
    
    for info in opt:
        # show every 50th result
        if modulostop(info):
            print "iteration %3i loss=%g" % (info['n_iter'], quadratic(wrt))
     
        # stop after total of 1000 iterations
        if fullstop(info):
            print "iteration %3i loss=%g" % (info['n_iter'], quadratic(wrt))
            break
    
