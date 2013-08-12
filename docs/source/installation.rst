Installation
============

Climin has been tested on `Python 2.7 <http://python.org>`_ with `numpy <http://numpy.org>`_ 1.8 and
`scipy <http://scipy.org>`_ 0.13. Currently, it is only available via git from
github::

    $ git clone https://github.com/BRML/climin.git

After that, `pip <http://www.pip-installer.org/>`_ can be used to install it on the Python path::

    $ cd climin
    $ pip install .

If you want to know whether everything works as we expect it, run the test
suite::

    $ nosetests tests/

for which `nose <https://nose.readthedocs.org/en/latest/>`_ is required.
