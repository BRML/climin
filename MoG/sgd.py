"""
Stochastic Gradient Descent on Mixture of Gaussians

"""

import numpy as np
import scipy
import theano
import theano.tensor as T
import theano.sandbox.linalg.ops
import theano.tensor.nnet
import theano.printing
import math
import pylab
from pylab import *
from matplotlib.figure import Figure
import time
import random
import itertools
import climin

inv = theano.sandbox.linalg.ops.MatrixInverse()
det = theano.sandbox.linalg.ops.Det()
diag = theano.sandbox.linalg.ops.ExtractDiag()

theano.config.compute_test_value = 'off'

class MoG(object):
    """ Mixture of Gaussians
    Defined by :
    - the number of Gaussians k
    - mean vectors for each of these Gaussians
    - covariance matrices for each of these Gaussians

    Depends on :
    -dimension of the inputs dim
    """

    def __init__(self, pars, k, dim):
        """ Initialize the Gaussian Mixture Model
        :type inputs: theano.tensor.TensorType
        :param inputs: symbolic variable that describes the inputs
        
        :type dim: int
        :param dim: dimensionality of the input space

        :type k: int
        :param k: number of Gaussians in the model

        """


        # number of gaussians used in the model
        self.NoG = k
        self.dimOfInput = dim

        self.flat = theano.shared(value=pars, name = 'pars')
        self.mu = self.flat[:self.NoG*self.dimOfInput].reshape((self.NoG,self.dimOfInput ))
        self.U = self.flat[self.NoG*self.dimOfInput : self.NoG*self.dimOfInput *(self.dimOfInput+1)].reshape((self.NoG,self.dimOfInput,self.dimOfInput))
        self.v = self.flat[self.NoG*self.dimOfInput *(self.dimOfInput+1):]

    
    def normalPdf(self, inputs):
        """Return the values of the probability density function given for inputs' points

        """
        
        def oneGaussian(index, inputs):

            mu = self.mu[index]
            UTemp = self.U[index]
            SigmaTemp = T.dot(UTemp.T, UTemp)
            d = inputs - mu.T
            SigmaInv =  inv(SigmaTemp)
            expArg1 = T.dot(d, SigmaInv)
            expArg2 = T.dot(expArg1, d.T)
            expArgDiag = diag(expArg2)
            e = T.exp(-0.5 * expArgDiag)
            detInt = 2 * math.pi * SigmaTemp
            inSqrt = det(detInt)
            factor = inSqrt**(-0.5)
            N = e * factor
            return N
        
 
        MoG, _ = theano.map(fn = oneGaussian,
                            sequences = [T.arange(self.NoG)],
                            non_sequences = inputs)
        
        
        return (MoG)
    
    def logNormalPdf(self, inputs):
        """Return the values of the probability density function given for inputs' points

        """
        
        def logOneGaussian(index, inputs):

            mu = self.mu[index]
            UTemp = self.U[index]
            SigmaTemp = T.dot(UTemp.T, UTemp)
            d = inputs - mu.T
            SigmaInv =  inv(SigmaTemp)
            term1Arg = T.dot(T.dot(d, SigmaInv), d.T)
            term1 = diag(term1Arg)
            detArg = 2 * math.pi * SigmaTemp
            logArg= det(detArg)
            term2 = T.log(logArg)
            N = -0.5 * (term1 + term2)
            return N
        
 
        logMoG, _ = theano.map(fn = logOneGaussian,
                            sequences = [T.arange(self.NoG)],
                            non_sequences = inputs)
        
        
        return (logMoG)
    
    
    def neg_log_likelihood(self, inputs):
        """ Return the log-likelihood of this model given the input data

        """
        return   -(T.sum(T.log(T.sum(T.nnet.softmax(self.v) * self.normalPdf(inputs).T, axis = 1))))
       



def negativeLogLikelihood(pars, inputs):
    N, dim = inputs.shape
    model = MoG(pars, 3, dim )    
    X = T.dmatrix('X')
    result = model.neg_log_likelihood(X)
    f = theano.function([X], result)
    return f(inputs)
    
def negativeLogLikelihood_grad(pars, inputs):
    N, dim = inputs.shape
    model = MoG(pars, 3, dim)
    X = T.dmatrix('X')
    cost = model.neg_log_likelihood(X)
    g_flat = T.grad(cost = cost, wrt = model.flat)
    grad = theano.function(inputs = [X], outputs = g_flat)
    return grad(inputs)


def dataset(N, k):
    res = np.empty((N, 2))
    for i in range(N):
        k = random.randint(1,3)
        if k == 1:
            res[i] =  np.random.randn(2) * 0.2
        elif k == 2:
            res[i] = np.random.randn(2) * 0.4  + np.array([1, 3])
        else:
            res[i] = np.random.randn(2) * 0.4   - 1.5
            
    return res


def show_graph():
    scatter(inputs[:,0], inputs[:,1], c = computedResponsabilities )
    draw()


    
def run():
    #creation of the input set
    k = 3  #number of gaussians
    inputs = dataset(300, k)
    N, dim = inputs.shape

    negLogLik = lambda x, inputs:  negativeLogLikelihood(x, inputs)
    negLogLik_grad = lambda x, inputs:  negativeLogLikelihood_grad(x, inputs)

    args = (((inputs,), {}) for _ in  itertools.count())
    

    wrt = np.random.rand(k*dim*(dim+1) + k)
    
    opt = climin.Lbfgs(wrt, negLogLik, negLogLik_grad, args=args)

    for i, info in enumerate(opt):
        print info['loss']
        if i  + 1 > 100:
            break
    

    
    

    
if __name__ == '__main__':
    run()
