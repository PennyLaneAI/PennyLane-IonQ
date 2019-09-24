.. _usage:

Plugin usage
############

PennyLane-IonQ provides two devices for PennyLane:

* :class:`ionq.simulator <~SimulatorDevice>`: provides an PennyLane device for the IonQ cloud simulator

* :class:`ionq.qpu <~QPUDevice>`: provides an PennyLane device for the IonQ QPU hardware


Using the devices
=================

Once PennyLane-IonQ is installed, the two IonQ devices
can be accessed straight away in PennyLane.

You can instantiate these devices in PennyLane as follows:

>>> import pennylane as qml
>>> dev1 = qml.device('ionq.dewdrop', wires=2, shots=1024)
>>> dev2 = qml.device('ionq.simulator', wires=3, shots=8192)

These devices can then be used just like other devices for the definition and evaluation of QNodes within PennyLane.


Supported operations
====================

All devices support all PennyLane `operations and observables <https://pennylane.readthedocs.io/en/latest/code/ops/qubit.html>`_, with the exception of:

- :class:`QubitStateVector`
- :class:`QubitUnitary`
- :class:`CRX`
- :class:`CRY`
- :class:`CRZ`
- :class:`CRot`
- :class:`CSWAP`
- :class:`PhaseShift`
- :class:`Hermitian`

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
