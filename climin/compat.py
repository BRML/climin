import sys

if sys.version_info[0] == 2:
    range = xrange
    # The following line looks weird, yes. However, if it is not added
    # ``basestring`` will not be part of the namespace of this module, and
    # henceforth not be importable.
    basestring = basestring
else:
    from builtins import range
    basestring = str
