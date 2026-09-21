IonQ Devices
============

The PennyLane-IonQ plugin provides the ability for PennyLane to access
devices available via IonQ's online API.

Currently, access is available to two remote devices: one to access an ideal
trapped-ion simulator and another to access to IonQ's trapped-ion QPUs.

.. raw::html
    <section id="simulator">

Trapped-ion simulator
------------------------

The :class:`~.pennylane_ionq.SimulatorDevice` provides a trapped-ion simulation.
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

Hardware noise model simulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The simulator supports hardware-aware noise models that approximate the noise characteristics
of IonQ's trapped-ion QPUs. To enable a noise model, pass the ``noise_model`` parameter:

.. code-block:: python

    dev = qml.device("ionq.simulator", wires=2, noise_model="aria-1")

Available noise models are ``"ideal"`` (default, noiseless), ``"harmony"``, ``"aria-1"``,
``"aria-2"``, ``"forte-1"``, and ``"forte-enterprise-1"``. For reproducible results, you can
also set a ``noise_seed``:

.. code-block:: python

    dev = qml.device("ionq.simulator", wires=2, noise_model="aria-1", noise_seed=42)

See the `IonQ noise model documentation <https://docs.ionq.com/guides/simulation-with-noise-models>`_
for details on each noise model.

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

    dev = qml.device("ionq.qpu", backend="aria-1", wires=2)

    @qml.qnode(dev)
    def circuit(x, y):
        qml.XX(x, wires=[0, 1])
        ops.YY(y, wires=[0, 1])
        return qml.expval(qml.PauliZ(0))

Both devices support the same set of operations.

.. raw::html
    </section>

Mid-circuit measurements and qubit reset
----------------------------------------

Both devices support mid-circuit measurements via
:func:`qp.measure <pennylane.measure>`. When a circuit contains a mid-circuit
measurement, the plugin converts it to an OpenQASM 3.0 program and submits it
through IonQ's ``ionq.qasm3.v1`` job type instead of the flat gate-list format:

.. code-block:: python

    import pennylane as qp

    dev = qp.device("ionq.simulator", wires=1)

    @qp.set_shots(1024)
    @qp.qnode(dev)
    def circuit():
        qp.Hadamard(wires=0)
        qp.measure(0, reset=True)
        qp.PauliX(wires=0)
        return qp.probs(wires=[0])

Measuring with ``reset=True`` emits an explicit ``reset`` statement in the
generated OpenQASM 3.0 program.

The following restrictions apply:

* Postselection (``qp.measure(0, postselect=1)``) is not supported.

* Classically controlled operations (:func:`qp.cond <pennylane.cond>`) are not
  supported. Apply :func:`qp.defer_measurements <pennylane.defer_measurements>`
  to the circuit (or use ``mcm_method="deferred"``) to convert conditionals to
  controlled gates.

* Circuits with mid-circuit measurements must be submitted one at a time; they
  cannot be part of a batch submission.

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
