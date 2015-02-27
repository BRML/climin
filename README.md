climin
------

climin is a Python package for optimization, heavily biased to machine learning
scenarios distributed under the BSD 3-clause license. It works on top of numpy
and (partially) gnumpy.

The project was started in winter 2011 by Christian Osendorfer and Justin Bayer.
Since then, Sarah Diot-Girard, Thomas Rueckstiess and Sebastian Urban have 
contributed. If you use climin in your (academic) work, please cite as 
(tech report is in preparation):

> J.Bayer and C.Osendorfer and S. Diot-Girard and T. Rückstiess and Sebastian Urban. climin - A pythonic framework for gradient-based function optimization. TUM Tech Report. 2015. 


Important links
---------------

 - Official repository of source: http://github.com/BRML/climin
 - Documentation: http://climin.readthedocs.org
 - Mailing list: climin@librelist.com (archive: http://librelist.com/browser/climin/)


Dependencies
------------

The software is tested under Python 2.7 with numpy 1.8, scipy 0.13. The tests
are run with nosetests.


Installation
------------

Use git to clone the official repository; then run `pip install --user -e .`
in the clone to intall in your local user space.


Testing
-------

From the download directory run ``nosetests tests/``.
