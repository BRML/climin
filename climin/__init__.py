from __future__ import absolute_import

# What follows is part of a hack to make control breaking work on windows even
# if scipy.stats ims imported. See:
# http://stackoverflow.com/questions/15457786/ctrl-c-crashes-python-after-importing-scipy-stats and
# https://github.com/numpy/numpy/issues/6923
import sys
import os
import imp
import ctypes

if sys.platform == 'win32':
    basepath = imp.find_module('numpy')[1]
    env = 'FOR_DISABLE_CONSOLE_CTRL_HANDLER'
    if env not in os.environ:
        os.environ[env] = '1'
    try:
        ctypes.CDLL(os.path.join(basepath, 'core', 'libmmd.dll'))
        ctypes.CDLL(os.path.join(basepath, 'core', 'libifcoremd.dll'))
    except Exception:
        pass

from .adadelta import Adadelta
from .adam import Adam
from .asgd import Asgd
from .bfgs import Bfgs, Lbfgs, Sbfgs
from .cg import ConjugateGradient, NonlinearConjugateGradient
from .gd import GradientDescent
from .nes import Xnes
from .rmsprop import RmsProp
from .rprop import Rprop
from .smd import Smd
