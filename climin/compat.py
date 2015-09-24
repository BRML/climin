import sys

if sys.version_info[0] == 2:
    range = xrange
else:
    from builtins import range

    basestring = str
