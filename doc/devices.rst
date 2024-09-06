IonQ Devices
============

The PennyLane-IonQ plugin provides the ability for PennyLane to access
devices available via IonQ's online API.

Currently, access is available to two remote devices: one to access an ideal
trapped-ion simulator and another to access to IonQ's trapped-ion QPUs.

.. raw::html
    <section id="simulator">

Ideal trapped-ion simulator
------------------------

The :class:`~.pennylane_ionq.SimulatorDevice` provides an ideal noiseless trapped-ion simulation.
Once the plugin has been installed, you can use this device directly in PennyLane by specifying ``"ionq.simulator"``:

.. code-block:: python

    import pennylane as qml
    from pennylane_ionq import ops

    dev = qml.device("ionq.simulator", wires=2)

    @qml.qnode(dev)
    def circuit(x, y, z):
        qml.RX(w, wires=0)
        ops.YY(y, wires=[0,1])
        ops.ZZ(z, wires=[0,1])
        return qml.expval(qml.PauliZ(0))

.. raw::html
    </section>
    <section id="qpu">

Trapped-Ion QPU
---------------

The :class:`~.pennylane_ionq.QPUDevice` provides access to IonQ's trapped-ion QPUs. Once the plugin has been
installed, you can use this device directly in PennyLane by specifying ``"ionq.qpu"`` with a
``"backend"`` from `available backends <https://docs.ionq.com/#tag/jobs>`_:

.. code-block:: python

    import pennylane as qml
    from pennylane_ionq import ops

    dev = qml.device("ionq.qpu", backend="harmony", wires=2)

    @qml.qnode(dev)
    def circuit(x, y):
        qml.XX(x, wires=[0, 1])
        ops.YY(y, wires=[0, 1])
        return qml.expval(qml.PauliZ(0))

Both devices support the same set of operations.

.. raw::html
    </section>

IonQ Operations
---------------

PennyLane-IonQ provides three gates specific to IonQ's ion-trap API:

.. autosummary::

    ~pennylane_ionq.ops.XX
    ~pennylane_ionq.ops.YY
    ~pennylane_ionq.ops.ZZ

These three gates can be imported from :mod:`pennylane_ionq.ops <~.ops>`.

Remote backend access
---------------------

Access credentials will be needed for the IonQ platform in order to
use these remote devices. These credentials should be provided to PennyLane via a
`configuration file or environment variable <https://pennylane.readthedocs.io/en/stable/introduction/configuration.html>`_.
Specifically, the variable ``IONQ_API_KEY`` must contain a valid access key for IonQ's online platform.
