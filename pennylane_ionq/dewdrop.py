# Copyright 2019 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
IonQ device classes
===================

**Module name:** :mod:`pennylane_ionq.dewdrop`

.. currentmodule:: pennylane_ionq.dewdrop

This module contains the PennyLane :class:`Device` classes for the IonQ
simulator and QPU.

Classes
-------

.. autosummary::
   DewdropDevice
   SimulatorDevice
   QPUDevice

Code details
~~~~~~~~~~~~
"""
import itertools
import functools
from time import sleep

# we always import NumPy directly
import numpy as np

from pennylane import Device, DeviceError

from .api_client import Job
from ._version import __version__


class DewdropDevice(Device):
    r"""Abstract Framework device for IonQ Dewdrop.

    Args:
        wires (int): the number of modes to initialize the device in
        target (str): the target device, either ``"simulator"`` or ``"qpu"``
        shots (int): number of circuit evaluations/random samples used
            to estimate expectation values of observables
    """
    pennylane_requires = ">=0.5.0"
    version = __version__
    author = "XanaduAI"

    _operation_map = {
        # native PennyLane operations also native to IonQ
        "PauliX": "x",
        "PauliY": "y",
        "PauliZ": "z",
        "Hadamard": "h",
        "CNOT": "cnot",
        "SWAP": "swap",
        "RX": "rx",
        "RY": "ry",
        "RZ": "rz",
        # operations not natively implemented in IonQ
        "Rot": None,
        "BasisState": None,
        "CZ": None,
        "CCNOT": None,
        # additional operations not native to PennyLane but present in IonQ
        "S": "s",
        "Sdg": "si",
        "T": "t",
        "Tdg": "ti",
        "V": "v",
        "Vdg": "vi",
        "XX": "xx",
        "YY": "yy",
        "ZZ": "zz",
    }

    observables = {"PauliX", "PauliY", "PauliZ", "Identity", "Hadamard"}
    _eigs = {}

    def __init__(self, wires, *, target="simulator", shots=1024):
        super().__init__(wires, shots)
        self.target = target
        self.reset()

    def reset(self):
        """Reset the device"""
        self.prob = None
        self._first_operation = True
        self.circuit = {"qubits": self.num_wires, "circuit": []}
        self.job = {"lang": "json", "body": self.circuit, "target": self.target}

    @property
    def operations(self):
        """Get the supported set of operations.

        Returns:
            set[str]: the set of PennyLane operation names the device supports
        """
        return set(self._operation_map.keys())

    def apply(self, operation, wires, par):
        if operation == "BasisState":
            if not self._first_operation:
                raise DeviceError(
                    "Operation {} cannot be used after other Operations have already been applied "
                    "on a {} device.".format(operation, self.short_name)
                )
            self._first_operation = False
            for w, p in enumerate(par[0]):
                if p == 1:
                    self.apply("PauliX", [w], [])
            return

        self._first_operation = False

        if operation == "Rot":
            self.apply("RZ", wires, [par[0]])
            self.apply("RY", wires, [par[1]])
            self.apply("RZ", wires, [par[2]])
            return

        if operation == "CZ":
            self.apply("Hadamard", [wires[1]], [])
            self.apply("CNOT", wires, [])
            self.apply("Hadamard", [wires[1]], [])
            return

        if operation == "CCNOT":
            self.apply("Hadamard", [wires[2]], [])
            self.apply("CNOT", [wires[1], wires[2]], [])
            self.apply("Tdg", [wires[2]], [])
            self.apply("CNOT", [wires[0], wires[2]], [])
            self.apply("T", [wires[2]], [])
            self.apply("CNOT", [wires[1], wires[2]], [])
            self.apply("Tdg", [wires[2]], [])
            self.apply("CNOT", [wires[0], wires[2]], [])
            self.apply("T", [wires[1]], [])
            self.apply("T", [wires[2]], [])
            self.apply("CNOT", [wires[0], wires[1]], [])
            self.apply("Hadamard", [wires[2]], [])
            self.apply("T", [wires[0]], [])
            self.apply("Tdg", [wires[1]], [])
            self.apply("CNOT", [wires[0], wires[1]], [])
            return

        op_name = self._operation_map[operation]
        gate = {"gate": op_name}
        if len(wires) == 2:
            if operation in {"SWAP", "XX", "YY", "ZZ"}:
                # these gates takes two targets
                gate["targets"] = wires
            else:
                gate["control"] = wires[0]
                gate["target"] = wires[1]
        else:
            gate["target"] = wires[0]

        if par:
            gate["rotation"] = par[0]

        self.circuit["circuit"].append(gate)

    def post_apply(self):
        for e in self.obs_queue:
            # Add unitaries prior to measurement if the expectation
            # depending on the observable to be measured
            self.rotate_basis(e.name, e.wires, e.parameters)

    def pre_measure(self):
        job = Job()

        # send job for exection
        print(self.job)
        job.manager.create(**self.job)

        # retrieve results
        while not job.is_complete:
            sleep(0.01)
            job.reload()

        job.manager.get(job.id.value)

        histogram = job.data.value["histogram"]
        self.prob = np.zeros([2 ** self.num_wires])
        self.prob[np.array([int(i) for i in histogram.keys()])] = list(histogram.values())

    def expval(self, observable, wires, par):
        eigvals = self.eigvals(observable, wires, par)
        probs = self.probabilities(wires)
        return (eigvals @ probs).real

    def var(self, observable, wires, par):
        eigvals = self.eigvals(observable, wires, par)
        probs = self.probabilities(wires)
        return (eigvals ** 2) @ probs - (eigvals @ probs).real ** 2

    def probabilities(self, wires=None):
        """Marginal probabilities of each computational basis
        state from the last run of the device.

        Args:
            wires (Sequence[int]): Sequence of wires to return
                marginal probabilities for. Wires not provided
                are traced out of the system.

        Returns:
            array[float]: array containing the probability values
            of measuring each computational basis state
        """
        if self.prob is None:
            return None

        wires = wires or range(self.num_wires)
        wires = np.hstack(wires)

        #basis_states = itertools.product(range(2), repeat=len(wires))
        inactive_wires = list(set(range(self.num_wires)) - set(wires))

        prob = self.prob.reshape([2] * self.num_wires)
        return np.apply_over_axes(np.sum, prob.T, inactive_wires).flatten()

    def rotate_basis(self, observable, wires, par):
        """Rotates the specified wires such that they
        are in the eigenbasis of the provided observable.

        Args:
            observable (str): the name of an observable
            wires (List[int]): wires the observable is measured on
            par (List[Any]): parameters of the observable
        """
        if observable == "PauliX":
            # X = H.Z.H
            self.apply("Hadamard", wires=wires, par=[])

        elif observable == "PauliY":
            # Y = (HS^)^.Z.(HS^) and S^=SZ
            self.apply("PauliZ", wires=wires, par=[])
            self.apply("S", wires=wires, par=[])
            self.apply("Hadamard", wires=wires, par=[])

        elif observable == "Hadamard":
            # H = Ry(-pi/4)^.Z.Ry(-pi/4)
            self.apply("RY", wires, [-np.pi / 4])

        # TODO: Uncomment the following when arbitrary hermitian
        # observables and arbitrary qubit unitaries are supported.
        #
        # elif observable == "Hermitian":
        #     # For arbitrary Hermitian matrix H, let U be the unitary matrix
        #     # that diagonalises it, and w_i be the eigenvalues.
        #     Hmat = par[0]
        #     Hkey = tuple(Hmat.flatten().tolist())
        #
        #     if Hkey in self._eigs:
        #         # retrieve eigenvectors
        #         U = self._eigs[Hkey]["eigvec"]
        #     else:
        #         # store the eigenvalues corresponding to H
        #         # in a dictionary, so that they do not need to
        #         # be calculated later
        #         w, U = np.linalg.eigh(Hmat)
        #         self._eigs[Hkey] = {"eigval": w, "eigvec": U}
        #
        #     # Perform a change of basis before measuring by applying U^ to the circuit
        #     self.apply("QubitUnitary", wires, [U.conj().T])

    def eigvals(self, observable, wires, par):
        """Determine the eigenvalues of observable(s).

        Args:
            observable (str, List[str]): the name of an observable,
                or a list of observables representing a tensor product
            wires (List[int]): wires the observable(s) is measured on
            par (List[Any]): parameters of the observable(s)

        Returns:
            array[float]: an array of size ``(2**len(wires),)`` containing the
            eigenvalues of the observable
        """
        # the standard observables all share a common eigenbasis {1, -1}
        # with the Pauli-Z gate/computational basis measurement
        standard_observables = {"PauliX", "PauliY", "PauliZ", "Hadamard"}

        # observable should be Z^{\otimes n}
        eigvals = pauli_eigs(len(wires))

        if isinstance(observable, list):
            # tensor product of observables

            # check if there are any non-standard observables (such as Identity, Hadamard)
            if set(observable) - standard_observables:
                # Tensor product of observables contains a mixture
                # of standard and non-standard observables
                eigvals = np.array([1])

                # group the observables into subgroups, depending on whether
                # they are in the standard observables or not.
                for k, g in itertools.groupby(
                    zip(observable, wires, par), lambda x: x[0] in standard_observables
                ):
                    if k:
                        # Subgroup g contains only standard observables.
                        # Determine the size of the subgroup, by transposing
                        # the list, flattening it, and determining the length.
                        n = len([w for sublist in list(zip(*g))[1] for w in sublist])
                        eigvals = np.kron(eigvals, pauli_eigs(n))
                    else:
                        # Subgroup g contains only non-standard observables.
                        for ns_obs in g:
                            # loop through all non-standard observables
                            if ns_obs[0] == "Identity":
                                # Identity observable has eigenvalues (1, 1)
                                eigvals = np.kron(eigvals, np.array([1, 1]))

                            # TODO: Uncomment the following when arbitrary hermitian
                            # observables and arbitrary qubit unitaries are supported.
                            # elif ns_obs[0] == "Hermitian":
                            #     # Hermitian observable has pre-computed eigenvalues
                            #     p = ns_obs[2]
                            #     Hkey = tuple(p[0].flatten().tolist())
                            #     eigvals = np.kron(eigvals, self._eigs[Hkey]["eigval"])


        # TODO: Uncomment the following when arbitrary hermitian
        # observables and arbitrary qubit unitaries are supported.
        # elif observable == "Hermitian":
        #     # single wire Hermitian observable
        #     Hkey = tuple(par[0].flatten().tolist())
        #     eigvals = self._eigs[Hkey]["eigval"]

        elif observable == "Identity":
            # single wire identity observable
            eigvals = np.ones(2 ** len(wires))

        return eigvals


@functools.lru_cache()
def pauli_eigs(n):
    r"""Returns the eigenvalues for :math:`Z^{\otimes n}`.

    Args:
        n (int): number of wires

    Returns:
        array[int]: eigenvalues of :math:`Z^{\otimes n}`
    """
    if n == 1:
        return np.array([1, -1])
    return np.concatenate([pauli_eigs(n - 1), -pauli_eigs(n - 1)])


class SimulatorDevice(DewdropDevice):
    r"""Simulator device for IonQ.

    Args:
        wires (int): the number of modes to initialize the device in
        shots (int): number of circuit evaluations/random samples used
            to estimate expectation values of observables
    """
    name = "IonQ Simulator PennyLane plugin"
    short_name = "ionq.simulator"

    def __init__(self, wires, *, shots=1024):
        super().__init__(wires=wires, target="simulator", shots=shots)


class QPUDevice(DewdropDevice):
    r"""QPU device for IonQ.

    Args:
        wires (int): the number of modes to initialize the device in
        shots (int): number of circuit evaluations/random samples used
            to estimate expectation values of observables
    """
    name = "IonQ QPU PennyLane plugin"
    short_name = "ionq.qpu"

    def __init__(self, wires, *, shots=1024):
        super().__init__(wires=wires, target="qpu", shots=shots)
