import scipy

from climin import GradientDescent
from climin.stops import converged

quadratic = lambda x: (x**2).sum()
quadraticprime = lambda x: 2 * x

if __name__ == '__main__':
    dim = 10
    wrt = scipy.random.standard_normal((dim,)) * 10 + 5
    
    loss_converged = converged(lambda: quadratic(wrt))    
    opt = GradientDescent(wrt, quadraticprime, steprate=0.01)
       
    for info in opt:
        print "iteration %3i loss=%g" % (info['n_iter'], quadratic(wrt))
        if loss_converged(info):
            print "loss converged."
            break
    
    # the same can be achieved with minimize_until, if the user doesn't
    # need control in between steps:
    
    # opt.minimize_until( loss_converged )