import scipy
import numpy as np
from scipy import misc

def numericalGradientChecker(f, fprime, inputDim, outputDim, args=None, bounds=None, numberOfTry=1):
    """
    bounds is of dimension inputDim*2, with a pair of min/max values for each dimension
    """

    args, kwargs = args.next()
    noError = True
    nbErrors = 0
    nTranspo = 0
    if bounds is None:
        bounds = np.ones((inputDim,2))
        bounds[:,0]*=-10
        bounds[:,1]*=10

    for nIter in range(numberOfTry):
        x0 = np.random.rand(inputDim)*(bounds[:,1]-bounds[:,0])+bounds[:,0]
        
        if outputDim == 1:
            #then we are checking the gradient
            gradient = fprime(x0, *args, **kwargs)
            for i in range(inputDim):
                deltaX = np.zeros(inputDim)
                deltaX[i] = 0.1
                derivative = (f(x0+deltaX, *args, **kwargs)-f(x0, *args, **kwargs))/deltaX[i]
                if (np.around(derivative, 6)!= np.around(gradient[i], 6)):
                    print "Numerical gradient different from given gradient"
                    print derivative, gradient[i]
                    noError = False

            if noError:
                print "succes"
        else:
            #jacobian
            jacobian = fprime(x0, *args, **kwargs)
            fX = f(x0, *args, **kwargs)
            for i in range(inputDim):
                deltaX = np.zeros(inputDim)
                deltaX[i] = 0.1
                fXdeltaX = f(x0+deltaX, *args, **kwargs)
                for j in range(outputDim):
                    derivative = (fXdeltaX[j]-fX[j])/deltaX[i]
                    if (np.around(derivative, 6)!= np.around(jacobian[j,i], 6)):
                        #print "Numerical gradient different from given gradient"
                        print derivative, jacobian[j,i]
                        noError = False
                        nbErrors +=1
                    ## if np.around(derivative, 6) == np.around(jacobian[j,i], 6):
                    ##     print "Maybe the jacobian is transposed"
                    ##     nTranspo +=1

            if noError:
                print "succes"
        print "errors", nbErrors
        print "transpositions", nTranspo
            
    
