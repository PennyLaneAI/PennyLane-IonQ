PennyLane IonQ Plugin
#####################

This PennyLane plugin allows IonQ simulators/hardware to be used as PennyLane devices.


`IonQ Dewdrop <https://dewdrop.ionq.co>`_ is a trapped ion simulator and hardware cloud platform.

`PennyLane <https://pennylane.readthedocs.io>`_ is a machine learning library for optimization
and automatic differentiation of hybrid quantum-classical computations.


Features
========

* Provides two IonQ devices, ``ionq.simulator`` and ``ionq.qpu``, for accessing the
  IonQ cloud platform and running QML algorithms on the provider simulators and trapped-ion QPUs

* Supports most PennyLane qubit operations and observables

* Provides additional IonQ-specific quantum operations, including ``S``, ``T``, ``V``,
  ``CCNOT``, and the Ising coupling gates ``XX``, ``YY``, ``ZZ``.


Installation
============

PennyLane-IonQ requires PennyLane. It can be installed via ``pip``:

.. code-block:: bash

    $ python -m pip install pennylane-ionq


Getting started
===============

Once PennyLane-IonQ is installed, the provided IonQ devices can be accessed straight
away in PennyLane.

You can instantiate these device for PennyLane as follows:

.. code-block:: python

    import pennylane as qml
    dev1 = qml.device('ionq.dewdrop', wires=2, shots=1024)
    dev1 = qml.device('ionq.simulator', wires=3, shots=8192)

These devices can then be used just like other devices for the definition and evaluation of
QNodes within PennyLane. For more details, see the
`plugin usage guide <https://pennylane-ionq.readthedocs.io/en/latest/usage.html>`_ and refer
to the PennyLane documentation.

Supported operations
====================

All devices support all PennyLane `operations and observables <https://pennylane.readthedocs.io/en/latest/code/ops/qubit.html>`_, with the exception of:

- ``QubitStateVector``
- ``QubitUnitary``
- ``CRX``
- ``CRY``
- ``CRZ``
- ``CRot``
- ``CSWAP``
- ``PhaseShift``
- ``Hermitian``

In addition, the plugin provides the following framework-specific operations for PennyLane. These are all importable from :mod:`pennylane_ionq.ops <.ops>`.

These operations include:

.. autosummary::
    pennylane_ionq.ops.S
    pennylane_ionq.ops.Sdg
    pennylane_ionq.ops.T
    pennylane_ionq.ops.Tdg
    pennylane_ionq.ops.CCNOT
    pennylane_ionq.ops.V
    pennylane_ionq.ops.Vdg
    pennylane_ionq.ops.XX
    pennylane_ionq.ops.YY
    pennylane_ionq.ops.ZZ

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
