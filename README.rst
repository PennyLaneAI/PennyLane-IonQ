PennyLane-IonQ Plugin
#####################

.. image:: https://img.shields.io/github/workflow/status/PennyLaneAI/pennylane-ionq/Tests/master?logo=github&style=flat-square
    :alt: GitHub Workflow Status (branch)
    :target: https://github.com/PennyLaneAI/pennylane-ionq/actions?query=workflow%3ATests

.. image:: https://img.shields.io/codecov/c/github/PennyLaneAI/pennylane-ionq/master.svg?logo=codecov&style=flat-square
    :alt: Codecov coverage
    :target: https://codecov.io/gh/PennyLaneAI/pennylane-ionq

.. image:: https://img.shields.io/codefactor/grade/github/PennyLaneAI/pennylane-ionq/master?logo=codefactor&style=flat-square
    :alt: CodeFactor Grade
    :target: https://www.codefactor.io/repository/github/pennylaneai/pennylane-ionq

.. image:: https://img.shields.io/readthedocs/pennylane-ionq.svg?logo=read-the-docs&style=flat-square
    :alt: Read the Docs
    :target: https://pennylane-ionq.readthedocs.io

.. image:: https://img.shields.io/pypi/v/PennyLane-ionq.svg?style=flat-square
    :alt: PyPI
    :target: https://pypi.org/project/PennyLane-ionq

.. image:: https://img.shields.io/pypi/pyversions/PennyLane-ionq.svg?style=flat-square
    :alt: PyPI - Python Version
    :target: https://pypi.org/project/PennyLane-ionq

.. header-start-inclusion-marker-do-not-remove

The PennyLane-IonQ plugin provides the ability to use IonQ's ion-trap
quantum computing backends with PennyLane.

`PennyLane <https://pennylane.ai>`_ provides open-source tools for
quantum machine learning, quantum computing, quantum chemistry, and hybrid quantum-classical computing.

`IonQ <https://www.ionq.com>`_ is a ion-trap quantum computing
company offering access to quantum computing devices over the cloud.

.. header-end-inclusion-marker-do-not-remove

The plugin documentation can be found `here <https://pennylane-ionq.readthedocs.io/en/latest/>`__.

Features
========

* Provides two devices which can be used with IonQ's online API: ``"ionq.simulator"`` and ``"ionq.qpu"``.
  These provide access to an ideal ion-trap simulator as well as IonQ's quantum hardware, respectively.

* The plugin provides additional support for the IonQ's Ising-type gates.

* Supports core PennyLane operations such as qubit rotations, Hadamard, basis state preparations, etc.

.. installation-start-inclusion-marker-do-not-remove

Installation
============

PennyLane-IonQ only requires PennyLane for use, no additional external frameworks are needed.
The plugin can be installed via ``pip``:
::

    $ python3 -m pip install pennylane-ionq

Alternatively, you can install PennyLane-IonQ from the source code by navigating to the top directory and running
::

    $ python3 setup.py install


If you currently do not have Python 3 installed,
we recommend `Anaconda for Python 3 <https://www.anaconda.com/download/>`_, a distributed
version of Python packaged for scientific computation.

Software tests
~~~~~~~~~~~~~~

To ensure that PennyLane-IonQ is working correctly after installation, the test suite can be
run by navigating to the source code folder and running
::

    $ make test


Documentation
~~~~~~~~~~~~~

To build the HTML documentation, go to the top-level directory and run
::

    $ make docs

The documentation can then be found in the ``doc/_build/html/`` directory.

.. installation-end-inclusion-marker-do-not-remove

Getting started
===============

Once PennyLane is installed, the provided IonQ devices can be accessed straight
away in PennyLane. However, the user will need access credentials for the IonQ platform in order to
use these remote devices. These credentials should be provided to PennyLane via a
`configuration file or environment variable <https://pennylane.readthedocs.io/en/stable/introduction/configuration.html>`_.
Specifically, the variable ``IONQ_API_KEY`` must contain a valid access key for IonQ's online platform.

You can instantiate the IonQ devices for PennyLane as follows:

.. code-block:: python

    import pennylane as qml
    dev1 = qml.device('ionq.simulator', wires=2, shots=1000)
    dev2 = qml.device('ionq.qpu', wires=2, shots=1000)

These devices can then be used just like other devices for the definition and evaluation of
quantum circuits within PennyLane. For more details and ideas, see the
`PennyLane website <https://pennylane.ai>`_ and refer
to the `PennyLane documentation <https://pennylane.readthedocs.io>`_.


Contributing
============

We welcome contributions—simply fork the PennyLane-IonQ repository, and then make a
`pull request <https://help.github.com/articles/about-pull-requests/>`_ containing your contribution.
All contributers to PennyLane-IonQ will be listed as contributors on the releases.

We also encourage bug reports, suggestions for new features and enhancements, and even links to cool
projects or applications built on PennyLane and IonQ.


Contributors
============

PennyLane-IonQ is the work of many `contributors <https://github.com/PennyLaneAI/pennylane-ionq/graphs/contributors>`_.

If you are doing research using PennyLane, please cite our papers:

    Ville Bergholm, Josh Izaac, Maria Schuld, Christian Gogolin, M. Sohaib Alam, Shahnawaz Ahmed,
    Juan Miguel Arrazola, Carsten Blank, Alain Delgado, Soran Jahangiri, Keri McKiernan, Johannes Jakob Meyer,
    Zeyue Niu, Antal Száva, Nathan Killoran.
    *PennyLane: Automatic differentiation of hybrid quantum-classical computations.* 2018.
    `arXiv:1811.04968 <https://arxiv.org/abs/1811.04968>`_

    Maria Schuld, Ville Bergholm, Christian Gogolin, Josh Izaac, and Nathan Killoran.
    *Evaluating analytic gradients on quantum hardware.* 2018.
    `Phys. Rev. A 99, 032331 <https://journals.aps.org/pra/abstract/10.1103/PhysRevA.99.032331>`_

.. support-start-inclusion-marker-do-not-remove

Support
=======

- **Source Code:** https://github.com/PennyLaneAI/pennylane-ionq
- **Issue Tracker:** https://github.com/PennyLaneAI/pennylane-ionq/issues

If you are having issues, please let us know by posting the issue on our GitHub issue tracker.

.. support-end-inclusion-marker-do-not-remove
.. license-start-inclusion-marker-do-not-remove

License
=======

PennyLane-IonQ is **free** and **open source**, released under the Apache License, Version 2.0.

.. license-end-inclusion-marker-do-not-remove
