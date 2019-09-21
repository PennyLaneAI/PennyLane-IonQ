PennyLane IonQ Plugin
#####################

`IonQ Dewdrop <https://dewdrop.ionq.co>`_ is a trapped ion simulator and hardware cloud platform.

`PennyLane <https://pennylane.readthedocs.io>`_ is a machine learning library for optimization
and automatic differentiation of hybrid quantum-classical computations.


Features
========

* Provides an IonQ Dewdrop device, for accessing the IonQ cloud platform
  and running QML algorithms on the provider simulators and trapped-ion QPUs

* Supports all PennyLane qubit operations and observables

* Provides additional IonQ-specific quantum operations


Installation
============

PennyLane-IonQ requires PennyLane. It can be installed via ``pip``:

.. code-block:: bash

    $ python -m pip install pennylane-ionq


Getting started
===============

Once PennyLane-IonQ is installed, the provided IonQ device can be accessed straight
away in PennyLane.

You can instantiate this device for PennyLane as follows:

.. code-block:: python

    import pennylane as qml
    dev1 = qml.device('ionq.dewdrop', wires=2, target='qpu', shots=1024)

These devices can then be used just like other devices for the definition and evaluation of
QNodes within PennyLane. For more details, see the
`plugin usage guide <https://pennylane-ionq.readthedocs.io/en/latest/usage.html>`_ and refer
to the PennyLane documentation.


Contributing
============

We welcome contributions - simply fork the PennyLane-IonQ repository, and then make a
`pull request <https://help.github.com/articles/about-pull-requests/>`_ containing your contribution.
All contributers to PennyLane-SF will be listed as authors on the releases.

We also encourage bug reports, suggestions for new features and enhancements, and even links to cool
projects or applications built on PennyLane and IonQ.

If you are doing research using PennyLane, please cite our papers:

    Ville Bergholm, Josh Izaac, Maria Schuld, Christian Gogolin, and Nathan Killoran.
    *PennyLane: Automatic differentiation of hybrid quantum-classical computations.* 2018.
    `arXiv:1811.04968 <https://arxiv.org/abs/1811.04968>`_

    Maria Schuld, Ville Bergholm, Christian Gogolin, Josh Izaac, and Nathan Killoran.
    *Evaluating analytic gradients on quantum hardware.* 2018.
    `Phys. Rev. A 99, 032331 <https://journals.aps.org/pra/abstract/10.1103/PhysRevA.99.032331>`_


Support
=======

- **Source Code:** https://github.com/XanaduAI/plugin-name
- **Issue Tracker:** https://github.com/XanaduAI/plugin-namesf/issues

If you are having issues, please let us know by posting the issue on our GitHub issue tracker.


License
=======

PennyLane-IonQ is **free** and **open source**, released under the Apache License, Version 2.0.
