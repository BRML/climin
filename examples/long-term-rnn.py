import itertools
import random

import chopmunk

from climin import KrylovSubspaceDescent, HessianFree, Rprop
from climin.linesearch import WolfeLineSearch
from climin.initialize import sparsify_columns

import pylab
import scipy
import numpy as np
import theano
import theano.tensor as T
from zeitgeist.model import rnn, lstmrnn
import zeitgeist.data
from zeitgeist.display import array2d_pil


# Hyper parameters.

n_inpt = 4
n_hidden = 100
n_output = 3
n_memory = 5
initial_damping = .1
damping_to_structural_damping = 0.3

# Expressions for the recurrent network.
# TODO: this functionality needs to be include in here to remove zeitgeist 
# dependency.
exprs, P = rnn(n_inpt, n_hidden, n_output, transferfunc='sig')

# To make the passing of the parameters explicit, we need to substitute it
# later with the givens parameter.
par_sub = T.vector()

# Some tensor for constructing the loss. the loss will only be defined on
# the end of the sequences.
inpt = exprs['inpt']
target = T.tensor3('target')
output = exprs['output']
exp_output = T.exp(output)
output = exp_output / (exp_output.sum(axis=2).dimshuffle(0, 1, 'x'))
output_in = exprs['output-in']
hidden_in_rec = exprs['hidden-in-rec']
hidden = exprs['hidden']
subtarget = target[-n_memory:]
suboutput = output[-n_memory:]

# Shorthand to create a cross entropy expression, which we will need for
# structural damping as well as the overall loss.
def cross_entropy(a, b):
    return -(a * T.log(b)).mean()

# Vector for the expression of the Hessian vector product, where this will
# be the vector.
p = T.vector('p')

# Expression for the change of hidden variables if we move the parameters
# into direction p.
changed_hidden = theano.clone(hidden, {P.flat: P.flat + p})

# Expression for the difference between the hiddens of the moved parameters
# and the previous hiddens.
#diff_hidden = ((hidden - changed_hidden)**2).mean()
def bernoulli_cross_entropy(target, output):
    eps = 1E-8
    return -(T.log(output + eps) * target +
             T.log(1 - output + eps) * (1 - target)).mean() 
diff_hidden = bernoulli_cross_entropy(hidden, changed_hidden)

# The loss and its gradient.
#loss = cross_entropy(target, output)
loss = cross_entropy(target, output)
lossgrad = T.grad(loss, P.flat)
empirical_loss = T.eq(T.gt(suboutput, 0.5), subtarget).mean()

# Expression for the Gauss-Newton matrix for the loss.
Jp = T.Rop(output_in, P.flat, p)
HJp = T.grad(T.sum(T.grad(loss, output_in) * Jp),
             output_in, consider_constant=[Jp])
Hp = T.grad(T.sum(HJp * output_in), P.flat, consider_constant=[HJp, Jp])

# The loss for the damping which will only be included in our Gauss-Newton
# matrix.
damping_factor = T.dscalar('damping-factor')
structural_damping = diff_hidden * 0
structural_damping *= damping_to_structural_damping * damping_factor

d_Jp = T.Rop(hidden_in_rec, P.flat, p)
d_HJp = T.grad(T.sum(T.grad(structural_damping, hidden_in_rec) * d_Jp),
             hidden_in_rec, consider_constant=[d_Jp])
d_Hp = T.grad(T.sum(d_HJp * hidden_in_rec), P.flat, consider_constant=[d_HJp, d_Jp])

Hp += d_Hp


# Functions.
givens = {P.flat: par_sub}
f = theano.function([par_sub, inpt, target], loss, givens=givens)
fprime = theano.function([par_sub, inpt, target], lossgrad, givens=givens)
f_predict = theano.function([par_sub, inpt], output, givens=givens)
f_empirical  = theano.function([par_sub, inpt, target], empirical_loss, givens=givens)
f_hidden = theano.function([par_sub, inpt], exprs['hidden'], givens=givens)

# Build a dataset.

#  First, built the possible bit strings.
bitarrays = [np.array([int(j)
                       for j in ('%5i' % int(bin(i)[2:])).replace(' ', '0')])
             for i in range(2**n_memory)]
bitarrays = np.array(bitarrays).T

n_samples = 2**5
n_timesteps = 60

X = scipy.zeros((n_timesteps, n_samples, 4))
X[:, :, 0] = 1

X[:n_memory, :, 0] = 0
X[:n_memory, :, 1] = bitarrays
X[:n_memory, :, 2] = 1 - bitarrays

X[-n_memory, :, :] = 0, 0, 0, 1

Z = scipy.zeros((n_timesteps, n_samples, 3))
Z[:-n_memory, :, :] = 0, 0, 1
Z[-n_memory:, :, :2] = X[:n_memory, :, 1:3]


print 'X'
print "".join(str(int(i)) for i in X[:, n_samples / 2, 0].tolist())
print "".join(str(int(i)) for i in X[:, n_samples / 2, 1].tolist())
print "".join(str(int(i)) for i in X[:, n_samples / 2, 2].tolist())
print "".join(str(int(i)) for i in X[:, n_samples / 2, 3].tolist())
print 'Z'
print "".join(str(int(i)) for i in Z[:, n_samples / 2, 0].tolist())
print "".join(str(int(i)) for i in Z[:, n_samples / 2, 1].tolist())
print "".join(str(int(i)) for i in Z[:, n_samples / 2, 2].tolist())

#1/0

TX = X
TZ = Z

#P.randomize(1./15)
P.randomize(1.)

# Tune down the values of inputs to hiddens.
#P['inweights'] *= 15

sparsify_columns(P['hiddenweights'], 15)

#hiddens = f_hidden(P.data, X[:, 0:1, :])
#import matplotlib.pyplot as plt
#fig = plt.figure()
#ax = fig.add_subplot(111)
#ax.plot(hiddens[:, 0, :])
#fig.savefig('initial.png')
#del fig


args = (([X, Z], {}) for _ in itertools.repeat(()))
#X_minibatches = zeitgeist.data.minibatches(X, 100, d=1)
#Z_minibatches = zeitgeist.data.minibatches(Z, 100, d=1)
#minibatches = zip(X_minibatches, Z_minibatches)
#idxs = (random.randint(0, len(minibatches) - 1) for _ in itertools.count())
# cg_args = ((minibatches[i], {}) for i in idxs)
cg_args = args
kargs = (([X, Z], {}) for _ in itertools.repeat(()))
Hargs = (([X, Z], {}) for _ in itertools.repeat(()))

print '#pars:', P.data.size

ignore = ['args', 'kwargs', 'gradient', 'Hp', 'direction', 'step',
          'cg_minimum', 'basis', 'gradient_diff', 'coefficients', 'grad']
console_sink = chopmunk.prettyprint_sink()
console_sink = chopmunk.dontkeep(console_sink, ignore)

file_sink = chopmunk.file_sink('longterm.log')
file_sink = chopmunk.jsonify(file_sink)
file_sink = chopmunk.dontkeep(file_sink, ignore)

logger = chopmunk.broadcast(console_sink, file_sink)
logger = chopmunk.timify(logger)
logfunc = logger.send

optimizer = 'ksd'

if optimizer == 'ksd':
    givens.update({damping_factor: 0.})
    f_Hp = theano.function([par_sub, p, inpt, target], Hp,
                           givens=givens)
    opt = KrylovSubspaceDescent(
            # 50 bases seems to work well
        P.data, f, fprime, f_Hp, n_bases=50,
        args=args, logfunc=logfunc)
elif optimizer == 'rprop':
    opt = Rprop(P.data, f, fprime, args=args, logfunc=logfunc)
elif optimizer == 'hf':
    f_Hp = theano.function([par_sub, p, damping_factor, inpt, target], Hp, 
                           givens=givens)
    #line_search = WolfeLineSearch(P.data, f, fprime)
    opt = HessianFree(
        P.data, f, fprime, f_Hp, args=args, cg_args=cg_args,
        initial_damping=initial_damping,
        explicit_damping=True,
        logfunc=logfunc)

import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111)
for i, info in enumerate(opt):
    info['test-loss'] = f(P.data, TX, TZ)
    info['train-empirical'] = f_empirical(P.data, X, Z)
    info['test-empirical'] = f_empirical(P.data, TX, TZ)

    hiddens = f_hidden(P.data, X[:, 0:1, :])
    ax.plot(hiddens[:, 0, :])
    fig.savefig('%i.png' % i)
    ax.cla()
    predictions = f_predict(P.data, TX[:, :, :])
    errors = abs(predictions - TZ).mean(axis=1)
    array2d_pil(errors, 'errors-%i.png' % i)
    arr = np.hstack((predictions[:, n_samples / 2, :], TZ[:, n_samples / 2, :])).T
    array2d_pil(arr, 'predictions-%i.png' % i)
    print 'min max', predictions.min(), predictions.max()

    logfunc(info)
    if i > 10000:
        break
