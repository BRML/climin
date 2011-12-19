import itertools
import pylab
import scipy
import theano
import theano.tensor as T
from zeitgeist.model import rnn, lstmrnn
from climin.ksd import KrylovSubspaceDescent
from climin import Rprop


# Hyper parameters.

n_inpt = 1
n_hidden = 20
n_output = 1
n_krylov_bases = 10


# Expressions.

exprs, P = rnn(n_inpt, n_hidden, n_output)
#exprs, P = lstmrnn(n_inpt, n_hidden, n_output)
target = T.tensor3('target')
loss = ((exprs['output'] - target)**2).mean()
lossgrad = T.grad(loss, P.flat)

p = T.vector('p')
#Hp = T.Rop(lossgrad, P.flat, p)
Hp = T.grad(T.sum(lossgrad * p), P.flat)

krylov_basis = theano.shared(scipy.empty((n_krylov_bases, P.data.shape[0])))
krylov_coefficients = theano.shared(scipy.empty(n_krylov_bases))

krylov_loss = theano.clone(
    loss, {P.flat: P.flat + T.dot(krylov_coefficients, krylov_basis)})
krylov_grad = T.grad(krylov_loss, krylov_coefficients)


# Functions.

inpt = exprs['inpt']
fandprime = theano.function([inpt, target], [loss, lossgrad])

f_predict = theano.function([inpt], exprs['output'])

# Build a dataset.
#X = scipy.zeros((100, 20, 1)) * 0.01
#X[:5] = scipy.random.random(X[:5].shape) > 0.5
#X[-5] = scipy.ones(X[0].shape)
#Z = scipy.ones(X.shape) * 0.5
#Z[-5:] = X[:5]
t = scipy.arange(0, 10, 0.05)[:, scipy.newaxis, scipy.newaxis]
X = scipy.sin(t / 2) * 2 + scipy.sin(t)
Z = X[:-20].copy()
X = X[20:]

P.randomize(1E-4)

args = (([X, Z], {}) for _ in itertools.repeat(()))
kargs = (([X, Z], {}) for _ in itertools.repeat(()))
Hargs = (([X, Z], {}) for _ in itertools.repeat(()))

print '#pars:', P.data.size


if True:
  f_Hp = theano.function([p, inpt, target], Hp)
  f_krylovandprime = theano.function([inpt, target], [krylov_loss, krylov_grad])
  opt = KrylovSubspaceDescent(
      P.data, fandprime, f_Hp, f_krylovandprime,
      krylov_basis.get_value(borrow=True, return_internal_type=True),
      krylov_coefficients.get_value(borrow=True, return_internal_type=True),
      args, kargs, Hargs, verbose=True)
else:
  opt = Rprop(P.data, fandprime,
              minstep=1E-9, maxstep=0.1, stepgrow=1.1, stepshrink=0.5,
              args=args)
opt = iter(opt)
prevloss = float('inf')
losses = []
steps = []
for i, info in enumerate(opt):
  loss = info['loss']
  if i > 50 and prevloss - loss < 1E-10 or loss < 0.003:
    break
  print 'loss', loss
  print 'max step', abs(info['step']).max()
  prevloss = loss

  losses.append(info['loss'])
  step = info['step']
  steps.append(scipy.sqrt(scipy.dot(step.T, step)))


pylab.plot(steps)
pylab.plot(losses)
pylab.show()


pylab.plot(f_predict(X)[:, 0, 0])
pylab.plot(Z[:, 0, 0])
pylab.show()

#print f_predict(X)[-5:, 0, 0]
#print Z[-5:, 0, 0]
