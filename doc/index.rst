PennyLane IonQ Plugin
#####################

:Release: |release|
:Date: |today|


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

To get started with the PennyLane IonQ plugin, follow the :ref:`installation steps <installation>`, then see the :ref:`usage <usage>` page.


Authors
=======

If you are doing research using PennyLane, please cite our papers:

    Ville Bergholm, Josh Izaac, Maria Schuld, Christian Gogolin, and Nathan Killoran.
    *PennyLane: Automatic differentiation of hybrid quantum-classical computations.* 2018.
    `arXiv:1811.04968 <https://arxiv.org/abs/1811.04968>`_

    Maria Schuld, Ville Bergholm, Christian Gogolin, Josh Izaac, and Nathan Killoran.
    *Evaluating analytic gradients on quantum hardware.* 2018.
    `Phys. Rev. A 99, 032331 <https://journals.aps.org/pra/abstract/10.1103/PhysRevA.99.032331>`_


Contents
========

.. rst-class:: contents local topic

.. toctree::
   :maxdepth: 2
   :caption: Getting started

   installing
   usage

.. rst-class:: contents local topic

.. toctree::
   :maxdepth: 1
   :caption: Code details

   code/ops
   code/dewdrop
   code/api_client
