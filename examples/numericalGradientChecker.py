import scipy
import numpy as np
from scipy import misc
import itertools
import random
import math


class numericalGradientChecker:
    """
    Numerical Gradient Checker for debugging

    """


    def __init__(self, f, fprime, inputDim, outputDim, args=None, bounds=None):
    
        self.f = f
        self.fprime = fprime
        self.inputDim = inputDim
        self.outputDim = outputDim
        
        if args is None:
            self.args = itertools.repeat(([], {}))
        else:
            self.args = args

        # bounds is of dimension inputDim*2, with a pair of min/max values for each dimension
        if bounds is None:
            self.bounds = np.ones((inputDim,2))
            self.bounds[:,0]*=-10
            self.bounds[:,1]*=10
        else:
            self.bounds = bounds

        self.epsilon = 1e-6 #amplitude of the variation on the input
        self.precision = 1e-4 #expected precision in number of decimals

    def acceptable_deviation(self, a, b):
        # return (np.around(a, self.precision) == np.around(b,  self.precision))
        if (abs(a)>1e-2): #or else, numerical instability
            ret = abs((a-b)/a)
        else:
            ret = abs(a-b)
        return (ret > self.precision)


    def __iter__(self):
        random.seed()
        x0 = np.random.rand(self.inputDim)*(self.bounds[:,1]-self.bounds[:,0]) + self.bounds[:,0]
        info = {}

        for it, (args, kwargs) in enumerate(self.args):
            noError = True
            nbErrors = 0
            if self.outputDim == 1:
                #then we are checking the gradient
                gradient = self.fprime(x0, *args, **kwargs)
                fX = self.f(x0, *args, **kwargs)
                for i in range(self.inputDim):
                    deltaX = np.zeros(self.inputDim)
                    deltaX[i] = self.epsilon
                    derivative = (self.f(x0+deltaX, *args, **kwargs)-fX)/deltaX[i]
                if (self.acceptable_deviation(gradient[i], derivative)):
                    print "Numerical gradient different from given gradient"
                    print derivative, gradient[i]
                    noError = False
                    nbErrors +=1

            else:
                #jacobian
                jacobian = self.fprime(x0, *args, **kwargs)
                fX = self.f(x0, *args, **kwargs)
                for i in range(self.inputDim):
                    deltaX = np.zeros(self.inputDim)
                    deltaX[i] = self.epsilon
                    fXdeltaX = self.f(x0+deltaX, *args, **kwargs)
                    for j in range(self.outputDim):
                        derivative = (fXdeltaX[j]-fX[j])/deltaX[i]
                        if (self.acceptable_deviation(jacobian[j,i], derivative)):
                            print "Numerical gradient different from given gradient"
                            print derivative, jacobian[j,i]
                            noError = False
                            nbErrors +=1
            info.update({ 'errors': nbErrors})   
            yield info       
            

def test_gradient():
    def f(x):
        return (x**2).sum()
    def fprime(x):
        return 2*x
    checker = numericalGradientChecker(f, fprime, 10, 1)
    for i, info in enumerate(checker):
        print "errors in gradient", info['errors']
        if i>1:
            break

            
def test_jacobian():
    def f(x):
        return x**2
    def fprime(x):
        return np.diag(2*x)
    checker = numericalGradientChecker(f, fprime, 10, 10)
    for i, info in enumerate(checker):
        print "errors in jacobian", info['errors']
        if i>1:
            break


