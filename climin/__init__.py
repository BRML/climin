# http://stackoverflow.com/questions/15457786/ctrl-c-crashes-python-after-importing-scipy-stats
import sys
import os
import imp
import ctypes

if sys.platform == 'win32':
    basepath = imp.find_module('numpy')[1]
    ctypes.CDLL(os.path.join(basepath, 'core', 'libmmd.dll'))
    ctypes.CDLL(os.path.join(basepath, 'core', 'libifcoremd.dll'))

from bfgs import Bfgs, Lbfgs, Sbfgs
from cg import ConjugateGradient, NonlinearConjugateGradient
from gd import GradientDescent
from nes import Xnes
from rmsprop import RmsProp
from rprop import Rprop
from smd import Smd
from asgd import Asgd
from adadelta import Adadelta
from adam import Adam
