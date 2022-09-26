PennyLane-IonQ Plugin
#####################

:Release: |release|

.. include:: ../README.rst
  :start-after:	header-start-inclusion-marker-do-not-remove
  :end-before: header-end-inclusion-marker-do-not-remove


Once the PennyLane-IonQ plugin is installed, the two provided IonQ devices can be accessed
straight away in PennyLane, without the need to import any additional packages.

Devices
=======

PennyLane-IonQ provides two IonQ devices for PennyLane:

.. title-card::
    :name: 'ionq.simulator'
    :description: Ideal noiseless trapped-ion simulator.
    :link: devices.html#simulator

.. title-card::
    :name: 'ionq.qpu'
    :description: Trapped-ion QPU
    :link: devices.html#qpu

.. raw:: html

    <div style='clear:both'></div>
    </br>

Both devices support the same operations, including IonQ's
custom :class:`.XX`, :class:`.YY`, and :class:`.ZZ` gates.

Remote backend access
=====================

The user will need access credentials for the IonQ platform in order to
use these remote devices. These credentials should be provided to PennyLane via a
`configuration file or environment variable <https://pennylane.readthedocs.io/en/stable/introduction/configuration.html>`_.
Specifically, the variable ``IONQ_API_KEY`` must contain a valid access key for IonQ's online platform.


.. toctree::
   :maxdepth: 2
   :titlesonly:
   :hidden:

   installation
   support

.. toctree::
   :maxdepth: 2
   :caption: Usage
   :hidden:

   devices

.. toctree::
   :maxdepth: 1
   :caption: API
   :hidden:

   code/__init__
   code/ops
