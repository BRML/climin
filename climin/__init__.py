from __future__ import absolute_import

# Control breaking does not work on Windows after e.g. scipy.stats is
# imported because certain Fortran libraries register their own signal handler
# See:
# http://stackoverflow.com/questions/15457786/ctrl-c-crashes-python-after-importing-scipy-stats
# and https://github.com/numpy/numpy/issues/6923
import sys
import os
import imp
import ctypes

if sys.platform == 'win32':
    # For setups where Intel Fortran compiler version >= 16.0 (This is the case
    # for Anaconda version 4.1.5 which comes with numpy version 1.10.4) is used,
    # the following flag allows to disable the additionally introduced signal
    # handler, older versios make no use of this environment variable
    env = 'FOR_DISABLE_CONSOLE_CTRL_HANDLER'
    if env not in os.environ:
        os.environ[env] = '1'
    # In setups with an older version, ensuring that the respective dlls are
    # loaded from the numpy core and not somewhere else (e.g. the Windows System
    # folder) helps
    basepath = imp.find_module('numpy')[1]
    # dll loading fails when Intel Fortran compiler version >= 16.0, therefore
    # use try/catch
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
